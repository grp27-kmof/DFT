#!/usr/bin/env python3
"""
mof_screen.py — Production DFT screening for MOF anode materials
===================================================================
Calculates four key properties using CP2K (PBE+D3 / DZVP-MOLOPT-SR-GTH):

  1. Electronic bandgap       (single-point DFT)
  2. Ion adsorption energy    (GEO_OPT of MOF + ion)
  3. Formation energy         (vs self-consistent elemental references)
  4. Volume expansion         (from CELL_OPT or fixed-cell GEO_OPT)

Key features
------------
  - Live progress bar + SCF step counter that updates every second
  - Each result is printed AS SOON AS that calculation finishes
  - Full restart/checkpoint support (safe to kill and rerun)
  - MPI + OpenMP support
  - Dynamic k-point grid (Gamma for MOFs, MP grid for small references)
  - Correct spin/multiplicity for reference molecules
  - Self-consistent elemental references via --compute-refs
  - Supports Li, Na, K, Mg, Ca, Zn, Al (--ion)

Usage
-----
  conda activate dft_env
  python mof_screen.py mof_relaxed.cif --cores 16
  python mof_screen.py mof_relaxed.cif --ion K --n-ions 4 --cell-opt
  python mof_screen.py mof_relaxed.cif --cores 16 --compute-refs
  python mof_screen.py mof_relaxed.cif --cores 4 --mpi-ranks 4
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from ase import Atoms
from ase.build import bulk, molecule
from ase.io import read, write

# ============================================================================
# LOGGING  — stdout + results/run.log
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("mof_dft")
os.environ["CP2K_DATA_DIR"] = "/home/srsarvanthikha/miniconda/envs/dft_env/share/cp2k/data"

# ============================================================================
# CONSTANTS
# ============================================================================
HARTREE_TO_EV = 27.211386245988

CP2K_BINARY_NAMES = ["cp2k.psmp", "cp2k.ssmp", "cp2k.popt", "cp2k"]

BASIS_SETS = {
    "default":       "DZVP-MOLOPT-SR-GTH",
    "high_accuracy": "TZV2P-MOLOPT-GTH",
}

# Hubbard U values (U_eff = U-J, eV, d-orbitals)
# Source: Jain et al., APL Materials 1, 011002 (2013)
HUBBARD_U: Dict[str, float] = {
    "Ti": 3.5, "V":  3.1, "Cr": 3.5, "Mn": 3.9, "Fe": 4.0,
    "Co": 3.3, "Ni": 6.4, "Cu": 4.0, "Zn": 0.0,
    "Zr": 2.0, "Mo": 4.38,"W":  4.0, "Nb": 1.5,
    "Ru": 3.0, "Rh": 3.0, "Pd": 2.0, "Ag": 1.5,
    "Ce": 5.0, "Nd": 5.0, "Sm": 5.0, "Eu": 5.0,
}

# Unpaired electrons per site (heuristic, capped at 5 sites per element)
UNPAIRED_PER_SITE: Dict[str, int] = {
    "Ti": 0, "V":  2, "Cr": 3, "Mn": 5, "Fe": 4,
    "Co": 3, "Ni": 2, "Cu": 1, "Zn": 0,
    "Zr": 0, "Mo": 0, "W":  0, "Nb": 0,
    "Ru": 0, "Rh": 0, "Pd": 0, "Ag": 0,
    "Ce": 1, "Nd": 3, "Sm": 5, "Eu": 6,
    # Alkali / Alkaline-earth / other metals
    "Li": 1, "Na": 1, "K": 1, "Mg": 0, "Ca": 0, "Al": 1,
}

# FIX6: Correct multiplicities for reference molecules
# O2=triplet(3), everything else is singlet/closed-shell
MOLECULE_MULTIPLICITY: Dict[str, int] = {
    "H2": 1, "N2": 1, "O2": 3, "Cl2": 1, "F2": 1,
}

# Ionic radii (Å) — used to scale min_dist in void detection
IONIC_RADII: Dict[str, float] = {
    "Li": 0.76, "Na": 1.02, "K": 1.38, "Mg": 0.72,
    "Ca": 1.00, "Zn": 0.74, "Al": 0.54,
}

# Charge state of each ion (number of electrons transferred per ion)
ION_CHARGE: Dict[str, int] = {
    "Li": 1, "Na": 1, "K": 1,
    "Mg": 2, "Ca": 2, "Zn": 2, "Al": 3,
}

# Approximate PBE bulk reference energies (eV/atom) — used as fallback
APPROX_ION_REFS: Dict[str, float] = {
    "Li": -1.900, "Na": -1.310, "K": -1.000,
    "Mg": -1.500, "Ca": -1.900, "Zn": -1.200, "Al": -3.500,
}

# Bulk structure info for alkali/alkaline-earth reference calculations
ALKALI_METAL_INFO: Dict[str, dict] = {
    "Li": {"structure": "bcc", "lattice_a": 3.51,  "multiplicity": 2},
    "Na": {"structure": "bcc", "lattice_a": 4.29,  "multiplicity": 2},
    "K":  {"structure": "bcc", "lattice_a": 5.23,  "multiplicity": 2},
}

# Cutoff settings
DEFAULT_CUTOFF     = 600
DEFAULT_REL_CUTOFF = 60
FAST_CUTOFF        = 400
FAST_REL_CUTOFF    = 40

# K-point target period (Å). For any cell axis > this, use Gamma only.
# MOFs are always Gamma-only. References with small cells get real k-grids.
KPOINT_MAX_CELL_LENGTH = 15.0
KPOINT_TARGET_PERIOD   = 25.0


# ============================================================================
# DATA CLASSES
# ============================================================================
@dataclass
class BandgapResult:
    bandgap_ev:      Optional[float] = None
    homo_ev:         Optional[float] = None
    lumo_ev:         Optional[float] = None
    scf_converged:   bool            = False
    total_energy_ev: Optional[float] = None
    classification:  str             = "UNKNOWN"
    elapsed_min:     float           = 0.0


@dataclass
class AdsorptionResult:
    e_ads_ev:     Optional[float] = None
    e_mof_ion_ev: Optional[float] = None   # renamed from e_mof_li_ev
    n_ions:       int             = 1
    ion_symbol:   str             = "Li"
    relaxed:      bool            = False
    elapsed_min:  float           = 0.0


@dataclass
class FormationResult:
    e_form_ev:          Optional[float] = None
    e_form_per_atom_ev: Optional[float] = None
    refs_complete:      bool            = True
    missing_elements:   List[str]       = field(default_factory=list)


@dataclass
class VolumeResult:
    v_before_A3:   Optional[float] = None
    v_after_A3:    Optional[float] = None
    expansion_pct: Optional[float] = None
    cell_relaxed:  bool            = False


@dataclass
class OCVResult:
    """Open-circuit voltage derived from adsorption energy."""
    ocv_v:          Optional[float] = None   # Volts
    z_charge:       int             = 1      # ion charge (Li/Na/K=1, Mg/Ca/Zn=2, Al=3)
    n_ions:         int             = 1
    ion_symbol:     str             = "Li"


@dataclass
class DiffusionBarrierResult:
    """
    Diffusion barrier (eV) from a pre-computed NEB/CI-NEB barrier file.
    The file should contain a single float (eV) or a CP2K NEB output.
    """
    barrier_ev:     Optional[float] = None
    source_file:    str             = ""
    available:      bool            = False


@dataclass
class DOSResult:
    """Density of states info parsed from CP2K PDOS files."""
    pdos_files:     List[str]       = field(default_factory=list)
    fermi_ev:       Optional[float] = None
    n_pdos_files:   int             = 0
    parsed:         bool            = False


@dataclass
class AllResults:
    formula:           str                   = ""
    n_atoms:           int                   = 0
    cif_path:          str                   = ""
    ion_symbol:        str                   = "Li"
    bandgap:           BandgapResult         = field(default_factory=BandgapResult)
    adsorption:        AdsorptionResult      = field(default_factory=AdsorptionResult)
    formation:         FormationResult       = field(default_factory=FormationResult)
    volume:            VolumeResult          = field(default_factory=VolumeResult)
    ocv:               OCVResult             = field(default_factory=OCVResult)
    diffusion_barrier: DiffusionBarrierResult = field(default_factory=DiffusionBarrierResult)
    dos:               DOSResult             = field(default_factory=DOSResult)


# ============================================================================
# LIVE PROGRESS MONITOR
# ============================================================================
class ProgressMonitor:
    """
    Watches a CP2K output file in a background thread and prints a live
    progress bar + SCF step counter to the terminal every second.

    What it tracks:
      - SCF steps completed (from "Step number" lines)
      - Outer SCF cycles (from "outer SCF" lines)
      - GEO_OPT steps (from "STEP| " lines)
      - Elapsed time
      - Whether SCF has converged
    """

    BAR_WIDTH = 35

    def __init__(self, out_file: str, label: str,
                 max_scf: int = 300, max_geo: int = 300):
        self.out_file   = out_file
        self.label      = label
        self.max_scf    = max_scf
        self.max_geo    = max_geo
        self._stop      = threading.Event()
        self._thread    = threading.Thread(target=self._run, daemon=True)
        self._t0        = time.time()

    def start(self):
        self._t0 = time.time()
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=3)
        # Clear the progress line
        sys.stdout.write("\r" + " " * 100 + "\r")
        sys.stdout.flush()

    def _parse(self) -> dict:
        """Read the output file and extract progress counters."""
        data = {"scf_steps": 0, "outer_scf": 0, "geo_steps": 0,
                "converged": False, "energy": None, "last_force": None}
        if not os.path.exists(self.out_file):
            return data
        try:
            with open(self.out_file, "r") as f:
                for line in f:
                    # SCF step lines: "    Step number    1 ..."
                    if re.search(r"Step number\s+\d+", line):
                        m = re.search(r"Step number\s+(\d+)", line)
                        if m:
                            data["scf_steps"] = int(m.group(1))
                    # Outer SCF
                    if "outer SCF" in line and "converged" not in line.lower():
                        m = re.search(r"outer SCF\s+(\d+)", line)
                        if m:
                            data["outer_scf"] = int(m.group(1))
                    # GEO_OPT step
                    if "STEP|" in line:
                        m = re.search(r"STEP\|\s+(\d+)", line)  # FIX: escaped |
                        if m:
                            data["geo_steps"] = int(m.group(1))
                    # Convergence
                    if "SCF run converged" in line:
                        data["converged"] = True
                    # Latest energy
                    if "ENERGY| Total FORCE_EVAL" in line:
                        m = re.search(r"(-?[\d.]+(?:[Ee][+-]?\d+)?)\s*$",
                                      line.strip())
                        if m:
                            data["energy"] = float(m.group(1)) * HARTREE_TO_EV
                    # Max force (GEO_OPT convergence indicator)
                    if "Max. gradient" in line or "Max. step" in line:
                        m = re.search(r"([\d.]+(?:[Ee][+-]?\d+)?)\s*$",
                                      line.strip())
                        if m:
                            data["last_force"] = float(m.group(1))
        except Exception:
            pass
        return data

    def _bar(self, current: int, total: int) -> str:
        frac   = min(current / max(total, 1), 1.0)
        filled = int(self.BAR_WIDTH * frac)
        bar    = "█" * filled + "░" * (self.BAR_WIDTH - filled)
        return f"[{bar}] {current}/{total}"

    def _run(self):
        while not self._stop.is_set():
            data    = self._parse()
            elapsed = time.time() - self._t0
            mins    = int(elapsed // 60)
            secs    = int(elapsed % 60)

            # Choose what to display based on what's happening
            if data["geo_steps"] > 0:
                bar      = self._bar(data["geo_steps"], self.max_geo)
                phase    = f"GEO step {data['geo_steps']}"
                scf_info = f"| SCF {data['scf_steps']}/{self.max_scf}"
            else:
                bar   = self._bar(data["scf_steps"], self.max_scf)
                phase = f"SCF {data['scf_steps']}"
                if data["outer_scf"] > 0:
                    phase += f" (outer {data['outer_scf']})"
                scf_info = ""

            energy_str = ""
            if data["energy"] is not None:
                energy_str = f"| E={data['energy']:.4f} eV"

            line = (f"\r  ⚙ {self.label:20s} {bar}  "
                    f"{phase:25s}{scf_info}  "
                    f"{energy_str}  {mins:3d}m{secs:02d}s")

            sys.stdout.write(line[:120])  # truncate to terminal width
            sys.stdout.flush()
            time.sleep(1.0)


# ============================================================================
# STRUCTURE HELPERS
# ============================================================================
def detect_transition_metals(atoms: Atoms) -> Dict[str, float]:
    """Return {element: U_eff_eV} for TM elements present with U > 0."""
    syms = set(atoms.get_chemical_symbols())
    return {s: HUBBARD_U[s] for s in syms if s in HUBBARD_U and HUBBARD_U[s] > 0.0}


def estimate_multiplicity(atoms: Atoms) -> int:
    """
    Estimate spin multiplicity = 2S+1.
    Sums unpaired electrons across TM sites, capped at 5 sites per element
    to avoid absurd values for MOFs with many identical metal nodes.
    """
    counts = Counter(atoms.get_chemical_symbols())
    total  = 0
    for elem, n in counts.items():
        per = UNPAIRED_PER_SITE.get(elem, 0)
        if per > 0:
            total += per * min(n, 5)
    return total + 1


def is_metallic_system(atoms: Atoms) -> bool:
    """
    Heuristic: return True if the system likely needs smearing.
    True only for pure metals or TM bulk references.
    MOFs always return False.
    """
    non_metal = {"H", "C", "N", "O", "F", "Cl", "Br", "I", "P", "S", "Se"}
    # If more than 80% of atoms are non-metal elements, it's not a metal
    nm_count = sum(1 for s in atoms.get_chemical_symbols() if s in non_metal)
    return nm_count / len(atoms) < 0.2


# FIX5: K-point grid — only for small cells (elemental references)
def compute_kpoint_grid(cell_lengths: np.ndarray) -> Tuple[int, int, int]:
    """
    Return Monkhorst-Pack grid. For any cell axis > KPOINT_MAX_CELL_LENGTH,
    the whole grid becomes 1x1x1 (Gamma only). This prevents k-points from
    being applied to large MOF cells where they are unnecessary and costly.
    """
    if np.any(cell_lengths > KPOINT_MAX_CELL_LENGTH):
        return (1, 1, 1)
    grid = []
    for L in cell_lengths:
        k = max(1, int(round(KPOINT_TARGET_PERIOD / L)))
        grid.append(k)
    return tuple(grid)


def find_void_sites(atoms: Atoms, ion: str = "Li", n_sites: int = 1,
                    grid_pts: int = 12) -> List[np.ndarray]:
    """
    Find the n_sites largest void positions using a fractional-coordinate grid.
    Uses minimum image convention so ions are never placed near periodic images.
    Ensures inserted ions are also at least min_dist apart from each other.
    Minimum distance is scaled by ionic radius of the ion being inserted.
    """
    # Scale min_dist by ionic radius relative to Li
    base_min_dist = 1.8
    scale    = IONIC_RADII.get(ion, 1.0) / IONIC_RADII.get("Li", 0.76)
    min_dist = max(1.5, base_min_dist * scale)

    cell     = atoms.get_cell()
    cell_inv = np.linalg.inv(cell.T)
    pos      = atoms.get_positions()

    lin       = np.linspace(0.05, 0.95, grid_pts)
    grid_frac = np.array([[i, j, k] for i in lin for j in lin for k in lin])
    grid_cart = grid_frac @ cell  # (N,3)

    # Min distance from each grid point to any framework atom (periodic)
    min_dists = np.full(len(grid_cart), np.inf)
    for p in pos:
        diff      = grid_cart - p
        df        = diff @ cell_inv.T
        df       -= np.round(df)
        dists     = np.linalg.norm(df @ cell, axis=1)
        min_dists = np.minimum(min_dists, dists)

    order = np.argsort(-min_dists)
    sites: List[np.ndarray] = []

    for idx in order:
        if len(sites) >= n_sites:
            break
        cand      = grid_cart[idx]
        void_size = min_dists[idx]
        if void_size < min_dist:
            continue
        # Check distance to already placed ions (periodic)
        too_close = False
        for placed in sites:
            df  = (cand - placed) @ cell_inv.T
            df -= np.round(df)
            if np.linalg.norm(df @ cell) < min_dist:
                too_close = True
                break
        if not too_close:
            sites.append(cand)
            log.info(f"    Void site {len(sites)}: {np.round(cand,2)} Å  "
                     f"(nearest atom {void_size:.2f} Å)")

    if len(sites) < n_sites:
        log.warning(f"    Only {len(sites)}/{n_sites} sites found with "
                    f"min_dist>{min_dist:.2f} Å — try reducing min_dist or grid_pts.")
    return sites


def build_loaded_structure(atoms: Atoms, ion: str = "Li",
                            n: int = 1) -> Atoms:
    """Insert n ions into the MOF at the largest void sites."""
    sites = find_void_sites(atoms, ion=ion, n_sites=n)
    if not sites:
        raise RuntimeError(
            "No void sites found. Ensure CIF is a relaxed structure with open pores.")
    loaded = atoms.copy()
    for pos in sites:
        loaded += Atoms(ion, positions=[pos],
                        cell=atoms.get_cell(), pbc=True)
    return loaded


# ============================================================================
# CP2K INPUT BUILDER  (helpers defined BEFORE the main generator — FIX8)
# ============================================================================
def _plus_u_block(metals: Dict[str, float]) -> str:
    if not metals:
        return ""
    lines = ["      &PLUS_U"]
    for elem, u in sorted(metals.items()):
        lines += [f"        &KIND {elem}",
                  f"          L 2",
                  f"          U_MINUS_J [eV] {u:.4f}",
                  f"        &END KIND"]
    lines.append("      &END PLUS_U")
    return "\n".join(lines)


def _subsys_block(atoms: Atoms, basis: str) -> str:
    cell  = atoms.get_cell()
    lines = ["  &SUBSYS", "    &CELL"]
    for name, vec in zip("ABC", cell):
        lines.append(f"      {name}  {vec[0]:.10f}  {vec[1]:.10f}  {vec[2]:.10f}")
    lines += ["      PERIODIC XYZ", "    &END CELL", "    &COORD"]
    for a in atoms:
        p = a.position
        lines.append(f"      {a.symbol:4s}  {p[0]:.10f}  {p[1]:.10f}  {p[2]:.10f}")
    lines.append("    &END COORD")
    for sym in sorted(set(atoms.get_chemical_symbols())):
        lines += [f"    &KIND {sym}",
                  f"      BASIS_SET {basis}",
                  f"      POTENTIAL GTH-PBE",
                  f"    &END KIND"]
    lines.append("  &END SUBSYS")
    return "\n".join(lines)


def _kpoints_block(k1: int, k2: int, k3: int) -> str:
    if k1 == 1 and k2 == 1 and k3 == 1:
        return ""  # Gamma-only: no block needed
    return (f"\n    &KPOINTS\n"
            f"      SCHEME MONKHORST-PACK {k1} {k2} {k3}\n"
            f"    &END KPOINTS\n")


def _motion_block(cell_opt: bool) -> str:
    constraint = "NONE" if cell_opt else "FIXED"
    return f"""
&MOTION
  &GEO_OPT
    TYPE MINIMIZATION
    MAX_ITER 300
    OPTIMIZER BFGS
    MAX_DR    0.0030
    MAX_FORCE 0.0005
    RMS_DR    0.0015
    RMS_FORCE 0.0003
  &END GEO_OPT
  &CELL_OPT
    TYPE DIRECT_CELL_OPT
    KEEP_SYMMETRY .FALSE.
    KEEP_ANGLES   .FALSE.
    CONSTRAINT {constraint}
    MAX_ITER 100
  &END CELL_OPT
  &PRINT
    &RESTART
      BACKUP_COPIES 3
      &EACH
        GEO_OPT 5
      &END EACH
    &END RESTART
    &TRAJECTORY
      FORMAT XYZ
      &EACH
        GEO_OPT 1
      &END EACH
    &END TRAJECTORY
  &END PRINT
&END MOTION
"""


def _mo_print_block() -> str:
    """MO eigenvalue printing — only for single-point (ENERGY) runs."""
    return """
    &PRINT
      &MO
        EIGENVALUES      .TRUE.
        OCCUPATION_NUMBERS .TRUE.
        &EACH
          QS_SCF 0
        &END EACH
      &END MO
      &PDOS
        NLUMO 0
        COMPONENTS .TRUE.
      &END PDOS
    &END PRINT"""


def generate_cp2k_input(atoms: Atoms,
                         metals: Dict[str, float],
                         multiplicity: int,
                         label: str,
                         run_type: str       = "ENERGY",
                         use_high_accuracy: bool = False,
                         cell_opt: bool      = False,
                         use_smearing: bool  = False,
                         fast_mode: bool     = False) -> str:
    """
    Generate a complete, correct CP2K input string.

    Key correctness points:
    - FIX3: &EXT_RESTART goes in &GLOBAL, handled separately in run_cp2k()
    - FIX5: K-points only for small cells (refs), Gamma for MOFs
    - FIX10: STRESS_TENSOR only for CELL_OPT
    - FIX1: All &EACH blocks properly closed with &END EACH
    - FIX2: Smearing only when use_smearing=True AND system is metallic
    """
    basis      = BASIS_SETS["high_accuracy"] if use_high_accuracy else BASIS_SETS["default"]
    use_uks    = multiplicity > 1
    cutoff     = FAST_CUTOFF     if fast_mode else DEFAULT_CUTOFF
    rel_cutoff = FAST_REL_CUTOFF if fast_mode else DEFAULT_REL_CUTOFF

    # FIX5: K-points — Gamma for any large cell
    k1, k2, k3 = compute_kpoint_grid(atoms.cell.lengths())
    kpts        = _kpoints_block(k1, k2, k3)
    if k1 > 1 or k2 > 1 or k3 > 1:
        log.info(f"    K-point grid: {k1}×{k2}×{k3} "
                 f"(cell {atoms.cell.lengths()[0]:.1f}×"
                 f"{atoms.cell.lengths()[1]:.1f}×"
                 f"{atoms.cell.lengths()[2]:.1f} Å)")

    # FIX2: Smearing — only when explicitly requested AND system is metallic
    smear_block = ""
    if use_smearing and is_metallic_system(atoms):
        smear_block = """
      &SMEAR
        METHOD FERMI_DIRAC
        ELECTRONIC_TEMPERATURE [K] 300
      &END SMEAR"""

    # Motion block only for geometry optimisation
    motion = _motion_block(cell_opt) if run_type in ("GEO_OPT", "CELL_OPT") else ""

    # MO printing only for single-point
    mo_print = _mo_print_block() if run_type == "ENERGY" else ""

    # FIX10: Stress tensor only when relaxing the cell
    stress = "STRESS_TENSOR ANALYTICAL" if cell_opt else ""

    return f"""! CP2K input — mof_screen.py
! Label: {label} | Run: {run_type} | Basis: {basis}
! UKS: {use_uks} | Mult: {multiplicity} | DFT+U: {bool(metals)}
! Smearing: {use_smearing and is_metallic_system(atoms)} | K-pts: {k1}×{k2}×{k3}
{motion}
&GLOBAL
  RUN_TYPE    {run_type}
  PROJECT_NAME {label}
  PRINT_LEVEL MEDIUM
  WALLTIME    86000
&END GLOBAL

&FORCE_EVAL
  METHOD Quickstep
  {stress}
  &DFT
    BASIS_SET_FILE_NAME BASIS_MOLOPT
    POTENTIAL_FILE_NAME GTH_POTENTIALS
    UKS         {'TRUE' if use_uks else 'FALSE'}
    MULTIPLICITY {multiplicity}
    CHARGE      0
    &MGRID
      CUTOFF     {cutoff}
      REL_CUTOFF {rel_cutoff}
      NGRIDS     5
    &END MGRID
    &QS
      METHOD      GPW
      EPS_DEFAULT 1.0E-12
      EPS_PGF_ORB 1.0E-30
    &END QS
    &POISSON
      PERIODIC XYZ
    &END POISSON
    &SCF
      MAX_SCF    300
      EPS_SCF    1.0E-7
      ADDED_MOS  {0 if len(atoms) < 10 else 50}
      SCF_GUESS  ATOMIC
      &MIXING
        METHOD  BROYDEN_MIXING
        ALPHA   0.1
        BETA    1.5
        NBUFFER 15
      &END MIXING
      &DIAGONALIZATION
        ALGORITHM STANDARD
      &END DIAGONALIZATION
      &OUTER_SCF ON
        MAX_SCF   50
        EPS_SCF   1.0E-7
        OPTIMIZER DIIS
      &END OUTER_SCF
      {smear_block}
      &PRINT
        &RESTART
          BACKUP_COPIES 2
          &EACH
            QS_SCF 20
          &END EACH
        &END RESTART
      &END PRINT
    &END SCF
    &XC
      &XC_FUNCTIONAL PBE
      &END XC_FUNCTIONAL
      &VDW_POTENTIAL
        POTENTIAL_TYPE PAIR_POTENTIAL
        &PAIR_POTENTIAL
          TYPE                  DFTD3
          REFERENCE_FUNCTIONAL  PBE
          PARAMETER_FILE_NAME   dftd3.dat
          R_CUTOFF              15.0
        &END PAIR_POTENTIAL
      &END VDW_POTENTIAL
      &XC_GRID
        XC_DERIV    SPLINE2
        XC_SMOOTH_RHO NN50
      &END XC_GRID
    &END XC
{_plus_u_block(metals)}
{kpts}
{mo_print}
  &END DFT
{_subsys_block(atoms, basis)}
&END FORCE_EVAL
"""


# ============================================================================
# CP2K RUNNER
# ============================================================================
def find_cp2k() -> Optional[str]:
    for name in CP2K_BINARY_NAMES:
        path = shutil.which(name)
        if path:
            return path
    return None


def run_cp2k(inp_file: str, out_file: str,
             num_cores: int, mpi_ranks: int = 1,
             restart_wfn:   Optional[str] = None,
             restart_coord: Optional[str] = None,
             label: str = "",
             max_scf: int = 300,
             max_geo: int = 300) -> bool:
    """
    Run CP2K with live progress bar.
    Streams output directly to file (no RAM buffering).
    Supports WFN and coordinate restarts.
    FIX3: &EXT_RESTART is injected into &GLOBAL, not &DFT.
    """
    cp2k = find_cp2k()
    if cp2k is None:
        log.error("CP2K not found. Run: conda activate dft_env")
        return False

    # ---- Patch WFN restart ----
    if restart_wfn and os.path.exists(restart_wfn):
        log.info(f"    Restarting WFN from: {restart_wfn}")
        txt = Path(inp_file).read_text()
        txt = txt.replace(
            "    SCF_GUESS  ATOMIC",
            f"    SCF_GUESS  RESTART\n    WFN_RESTART_FILE_NAME {restart_wfn}")
        Path(inp_file).write_text(txt)

    # FIX3: &EXT_RESTART belongs in &GLOBAL, not &DFT
    if restart_coord and os.path.exists(restart_coord):
        log.info(f"    Restarting coords from: {restart_coord}")
        txt = Path(inp_file).read_text()
        if "&EXT_RESTART" not in txt:
            ext = (f"\n&EXT_RESTART\n"
                   f"  RESTART_FILE_NAME {restart_coord}\n"
                   f"  RESTART_COUNTERS .FALSE.\n"
                   f"  RESTART_POS .TRUE.\n"
                   f"  RESTART_CELL .FALSE.\n"
                   f"&END EXT_RESTART\n")
            # Insert after &END GLOBAL
            txt = txt.replace("&END GLOBAL", "&END GLOBAL\n" + ext)
        Path(inp_file).write_text(txt)

    # ---- Build command ----
    mpi_cmd = ["mpirun", "-np", str(mpi_ranks)] if mpi_ranks > 1 else []
    cmd     = mpi_cmd + [cp2k, "-i", inp_file, "-o", out_file]
    log.info(f"    CMD: {' '.join(cmd)}")

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(num_cores)
    env["OMP_PROC_BIND"]   = "close"
    env["OMP_PLACES"]      = "cores"

    # ---- Start progress monitor ----
    monitor = ProgressMonitor(out_file, label=label or Path(inp_file).stem,
                               max_scf=max_scf, max_geo=max_geo)
    monitor.start()
    t0 = time.time()

    try:
        with open(out_file, "w") as fout, \
             open(out_file + ".stderr", "w") as ferr:
            proc = subprocess.run(cmd, stdout=fout, stderr=ferr,
                                  env=env, check=False)
    except Exception as e:
        monitor.stop()
        log.error(f"    Failed to launch CP2K: {e}")
        return False

    monitor.stop()
    elapsed = time.time() - t0
    log.info(f"    Finished in {elapsed/60:.1f} min (exit {proc.returncode})")

    if proc.returncode != 0:
        log.error(f"    Non-zero exit. Check {out_file}.stderr")
        return False
    return True


# ============================================================================
# OUTPUT PARSERS
# ============================================================================
_ENERGY_RE = re.compile(
    r"ENERGY\|\s+Total FORCE_EVAL.*?:\s*(-?[\d.]+(?:[Ee][+-]?\d+)?)")


def extract_total_energy(out_file: str) -> Optional[float]:
    """
    Extract the last total DFT energy (eV) from a CP2K output.
    Handles both fixed-point and scientific notation.
    """
    if not os.path.exists(out_file):
        return None
    e_au = None
    with open(out_file) as f:
        for line in f:
            if "ENERGY| Total FORCE_EVAL" in line:
                m = _ENERGY_RE.search(line)
                if m:
                    e_au = float(m.group(1))
    return None if e_au is None else e_au * HARTREE_TO_EV


def extract_bandgap_results(out_file: str) -> BandgapResult:
    """
    Parse CP2K single-point output for bandgap, HOMO, LUMO, convergence.

    Primary source : 'MO| Band gap:  X.XXX a.u.  Y.YYY eV'
    Fallback       : scan MO eigenvalue lines for last occ->virt transition
                     (FIX4 — restored the fallback that deepseek.py dropped)
    """
    result = BandgapResult()
    if not os.path.exists(out_file):
        return result

    _ev_re = re.compile(r"(-?[\d.]+(?:[Ee][+-]?\d+)?)\s+eV\s*$")
    _mo_re = re.compile(
        r"^\s*MO\|\s+\d+\s+(-?[\d.]+)\s+(-?[\d.]+)\s+([\d.]+)\s*$")

    # Track MO eigenvalue scanning: we want the LAST complete occ->virt pair
    current_homo: Optional[float] = None
    last_homo:    Optional[float] = None
    last_lumo:    Optional[float] = None
    lumo_found_for_current = False

    with open(out_file) as f:
        for line in f:
            # Convergence
            if "SCF run converged" in line:
                result.scf_converged = True

            # Primary: explicit band gap line
            if "MO| Band gap:" in line:
                m = _ev_re.search(line)
                if m:
                    result.bandgap_ev = float(m.group(1))

            # Fermi energy (proxy for HOMO in closed-shell systems)
            if "MO| E(Fermi):" in line:
                m = _ev_re.search(line)
                if m:
                    result.homo_ev = float(m.group(1))

            # FIX4: MO eigenvalue line scanning for HOMO/LUMO
            m = _mo_re.match(line)
            if m:
                e_ev = float(m.group(2))
                occ  = float(m.group(3))
                if occ >= 0.5:
                    current_homo           = e_ev
                    lumo_found_for_current = False   # reset on each occupied MO
                elif current_homo is not None and not lumo_found_for_current:
                    last_homo              = current_homo
                    last_lumo              = e_ev
                    lumo_found_for_current = True

    # Fallback: use eigenvalue-derived gap if explicit line missing
    if result.bandgap_ev is None and last_homo is not None and last_lumo is not None:
        result.bandgap_ev = last_lumo - last_homo
        log.info("    Gap from MO eigenvalues (fallback)")

    if result.homo_ev is None:
        result.homo_ev = last_homo
    result.lumo_ev = last_lumo

    result.total_energy_ev = extract_total_energy(out_file)

    # Classification
    if result.bandgap_ev is not None:
        g = result.bandgap_ev
        if g < 0.01:   result.classification = "METALLIC"
        elif g < 0.5:  result.classification = "SEMI-METAL"
        elif g < 1.5:  result.classification = "NARROW-GAP SEMICONDUCTOR"
        elif g < 3.0:  result.classification = "SEMICONDUCTOR"
        elif g < 5.0:  result.classification = "WIDE-GAP SEMICONDUCTOR"
        else:          result.classification = "INSULATOR"

    if not result.scf_converged:
        log.warning("    SCF did NOT converge — bandgap unreliable!")

    return result


def extract_final_volume(out_file: str) -> Optional[float]:
    """Extract last CELL| Volume line from CP2K output (Å³)."""
    if not os.path.exists(out_file):
        return None
    vol = None
    with open(out_file) as f:
        for line in f:
            if "CELL| Volume" in line:
                m = re.search(r"([\d.]+(?:[Ee][+-]?\d+)?)\s*$", line.strip())
                if m:
                    vol = float(m.group(1))
    return vol


def check_scf_converged(out_file: str) -> bool:
    if not os.path.exists(out_file):
        return False
    with open(out_file) as f:
        return any("SCF run converged" in l for l in f)


# ============================================================================
# RESULT PRINTER  (called immediately when each calc finishes)
# ============================================================================
def _box(title: str, lines: List[str]):
    """Print a neat box to stdout as soon as a result is available."""
    w = 62
    print("\n" + "┌" + "─" * w + "┐")
    print(f"│  ✓  {title:<{w-4}}│")
    print("├" + "─" * w + "┤")
    for l in lines:
        print(f"│  {l:<{w-2}}│")
    print("└" + "─" * w + "┘")
    sys.stdout.flush()


def print_bandgap(r: BandgapResult):
    lines = [
        f"Gap (PBE)        : {r.bandgap_ev:.4f} eV" if r.bandgap_ev else "Gap              : NOT FOUND",
        f"Classification   : {r.classification}",
        f"HOMO             : {r.homo_ev:.4f} eV" if r.homo_ev else "HOMO             : N/A",
        f"LUMO             : {r.lumo_ev:.4f} eV" if r.lumo_ev else "LUMO             : N/A",
        f"SCF converged    : {r.scf_converged}",
        f"Elapsed          : {r.elapsed_min:.1f} min",
        "",
        "NOTE: PBE underestimates real gap by ~30-50%.",
    ]
    _box("BANDGAP RESULT", lines)


def print_adsorption(r: AdsorptionResult):
    if r.e_ads_ev is None:
        _box("ADSORPTION ENERGY RESULT", ["FAILED — check adsorption.out"])
        return
    e = r.e_ads_ev
    if e < -1.0:   verdict = f"STRONG — excellent {r.ion_symbol} storage"
    elif e < -0.3: verdict = f"MODERATE — {r.ion_symbol} is stable"
    elif e < 0.0:  verdict = f"WEAK — {r.ion_symbol} barely binds"
    else:          verdict = f"UNFAVORABLE — {r.ion_symbol} does not stay in MOF"
    _box("ADSORPTION ENERGY RESULT", [
        f"E_ads ({r.n_ions}x {r.ion_symbol})    : {e:.4f} eV",
        f"Assessment       : {verdict}",
        f"Relaxed          : {r.relaxed}",
        f"Elapsed          : {r.elapsed_min:.1f} min",
        "",
        f"E_ads = E(MOF+{r.ion_symbol}) - E(MOF) - n*E({r.ion_symbol}_bulk)",
        "More negative = stronger binding = better anode.",
    ])


def print_formation(r: FormationResult):
    if r.e_form_ev is None:
        _box("FORMATION ENERGY RESULT", ["FAILED — E(MOF) not available"])
        return
    e = r.e_form_per_atom_ev
    if e < -1.0:   verdict = "HIGHLY STABLE"
    elif e < 0.0:  verdict = "STABLE"
    else:          verdict = "UNSTABLE (positive E_form)"
    warn = "  ⚠ Refs approximate" if not r.refs_complete else ""
    _box("FORMATION ENERGY RESULT", [
        f"E_form           : {r.e_form_ev:.4f} eV",
        f"E_form/atom      : {r.e_form_per_atom_ev:.4f} eV/atom",
        f"Assessment       : {verdict}",
        f"Refs complete    : {r.refs_complete}{warn}",
    ] + ([f"Missing refs     : {r.missing_elements}"] if r.missing_elements else []))


def print_volume(r: VolumeResult, ion: str = "Li"):
    if r.expansion_pct is None:
        _box("VOLUME EXPANSION RESULT", ["FAILED — check adsorption.out"])
        return
    p = r.expansion_pct
    if abs(p) < 1.0:   verdict = "EXCELLENT — rigid framework"
    elif p < 5.0:      verdict = "VERY GOOD — minimal change"
    elif p < 10.0:     verdict = "GOOD — acceptable for anode"
    elif p < 20.0:     verdict = "MODERATE — may limit cycle life"
    else:              verdict = "HIGH — poor cycle stability expected"
    _box("VOLUME EXPANSION RESULT", [
        f"V(bare MOF)      : {r.v_before_A3:.3f} Å³",
        f"V(MOF + {ion})   : {r.v_after_A3:.3f} Å³",
        f"Expansion        : {p:.3f} %",
        f"Cell relaxed     : {r.cell_relaxed}",
        f"Assessment       : {verdict}",
    ])


def print_ocv(r: OCVResult):
    if r.ocv_v is None:
        _box("OPEN-CIRCUIT VOLTAGE RESULT", ["NOT COMPUTED — requires adsorption energy"])
        return
    v = r.ocv_v
    if v > 2.0:   verdict = "HIGH — may cause electrolyte decomposition"
    elif v > 1.0: verdict = "MODERATE — check electrolyte window"
    elif v > 0.0: verdict = "LOW — suitable for anode (no dendrite risk)"
    else:         verdict = "NEGATIVE — thermodynamically unfavorable"
    _box("OPEN-CIRCUIT VOLTAGE RESULT", [
        f"OCV              : {v:.4f} V  (vs {r.ion_symbol}+/{r.ion_symbol})",
        f"Ion charge (z)   : {r.z_charge}",
        f"n_ions           : {r.n_ions}",
        f"Assessment       : {verdict}",
        "",
        f"OCV = -E_ads / (n × z × e)  [eV→V]",
    ])


def print_diffusion_barrier(r: DiffusionBarrierResult):
    if not r.available or r.barrier_ev is None:
        _box("DIFFUSION BARRIER RESULT", [
            "NOT AVAILABLE — run a separate CI-NEB calculation.",
            "Provide the result file via --barrier-file <path>.",
        ])
        return
    b = r.barrier_ev
    if b < 0.3:   verdict = "EXCELLENT — fast ion diffusion"
    elif b < 0.5: verdict = "GOOD — acceptable kinetics"
    elif b < 0.8: verdict = "MODERATE — may limit rate capability"
    else:         verdict = "HIGH — slow ion transport expected"
    _box("DIFFUSION BARRIER RESULT", [
        f"Barrier          : {b:.4f} eV",
        f"Source           : {r.source_file}",
        f"Assessment       : {verdict}",
        "",
        "Computed from pre-run CI-NEB or NEB barrier file.",
    ])


def print_dos(r: DOSResult):
    if not r.parsed:
        _box("DENSITY OF STATES RESULT", [
            "PDOS not parsed — run with --dos to extract DOS.",
        ])
        return
    lines = [
        f"PDOS files found : {r.n_pdos_files}",
        f"Fermi energy     : {r.fermi_ev:.4f} eV" if r.fermi_ev else "Fermi energy     : N/A",
    ]
    lines += [f"  {p}" for p in r.pdos_files[:8]]
    if len(r.pdos_files) > 8:
        lines.append(f"  ... and {len(r.pdos_files)-8} more")
    _box("DENSITY OF STATES RESULT", lines)


# ============================================================================
# CALCULATION 5 — OPEN-CIRCUIT VOLTAGE  (derived, no DFT)
# ============================================================================
def calc_ocv(ads: AdsorptionResult) -> OCVResult:
    """
    OCV (V) = -E_ads / (n_ions × z × e)
    z = ion charge (1 for Li/Na/K, 2 for Mg/Ca/Zn, 3 for Al).
    E_ads is already in eV, so dividing by z gives volts directly.
    """
    log.info("")
    log.info("━" * 62)
    log.info("  CALC 5/7 — OPEN-CIRCUIT VOLTAGE  (derived from E_ads)")
    log.info("━" * 62)

    z = ION_CHARGE.get(ads.ion_symbol, 1)
    result = OCVResult(z_charge=z, n_ions=ads.n_ions, ion_symbol=ads.ion_symbol)

    if ads.e_ads_ev is None:
        log.warning("  E_ads not available — OCV cannot be computed.")
        print_ocv(result)
        return result

    result.ocv_v = -ads.e_ads_ev / (ads.n_ions * z)
    log.info(f"  E_ads     = {ads.e_ads_ev:.6f} eV")
    log.info(f"  z (charge)= {z}")
    log.info(f"  OCV       = {result.ocv_v:.6f} V")

    print_ocv(result)
    return result


# ============================================================================
# CALCULATION 6 — DIFFUSION BARRIER  (reads pre-computed NEB file)
# ============================================================================
def calc_diffusion_barrier(barrier_file: Optional[str]) -> DiffusionBarrierResult:
    """
    Load diffusion barrier (eV) from a pre-computed NEB barrier file.

    Accepted file formats:
      1. Plain text with a single float on the first non-blank line.
      2. CP2K NEB output — scans for 'Activation energy:' lines.
    """
    log.info("")
    log.info("━" * 62)
    log.info("  CALC 6/7 — DIFFUSION BARRIER  (from NEB barrier file)")
    log.info("━" * 62)

    result = DiffusionBarrierResult()

    if not barrier_file:
        log.info("  No --barrier-file provided. Skipping.")
        print_diffusion_barrier(result)
        return result

    bf = Path(barrier_file)
    if not bf.exists():
        log.error(f"  Barrier file not found: {barrier_file}")
        print_diffusion_barrier(result)
        return result

    result.source_file = str(bf)
    text = bf.read_text()

    # Try CP2K NEB pattern first
    m = re.search(r"Activation energy:\s*(-?[\d.]+(?:[Ee][+-]?\d+)?)\s*eV", text)
    if m:
        result.barrier_ev = float(m.group(1))
        result.available  = True
        log.info(f"  Barrier (CP2K NEB): {result.barrier_ev:.4f} eV")
        print_diffusion_barrier(result)
        return result

    # Fallback: plain float on first non-blank line
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            try:
                result.barrier_ev = float(line)
                result.available  = True
                log.info(f"  Barrier (plain text): {result.barrier_ev:.4f} eV")
                break
            except ValueError:
                pass

    if not result.available:
        log.error("  Could not parse barrier from file.")

    print_diffusion_barrier(result)
    return result


# ============================================================================
# CALCULATION 7 — DENSITY OF STATES  (parse CP2K PDOS files)
# ============================================================================
def calc_dos(out_dir: Path, bandgap_result: BandgapResult) -> DOSResult:
    """
    Parse CP2K PDOS output files produced by the &PDOS block in _mo_print_block().
    Files are named  mof_bandgap-PDOS-k1-1.pdos  etc.
    Each file has a header line: '# Fermi energy [eV]:  X.XXX'
    """
    log.info("")
    log.info("━" * 62)
    log.info("  CALC 7/7 — DENSITY OF STATES  (parse CP2K PDOS files)")
    log.info("━" * 62)

    result = DOSResult()
    pdos_files = sorted(out_dir.glob("*.pdos"))

    if not pdos_files:
        log.info("  No .pdos files found in output directory.")
        log.info("  PDOS files are written by the &PDOS block during the bandgap calc.")
        print_dos(result)
        return result

    result.pdos_files  = [str(p) for p in pdos_files]
    result.n_pdos_files = len(pdos_files)

    # Extract Fermi energy from any pdos file header
    _fermi_re = re.compile(r"Fermi energy[\s\[\w]*\]?:?\s*(-?[\d.]+(?:[Ee][+-]?\d+)?)",
                           re.IGNORECASE)
    for pf in pdos_files:
        try:
            with open(pf) as fh:
                for line in fh:
                    if not line.startswith("#"):
                        break
                    mm = _fermi_re.search(line)
                    if mm:
                        result.fermi_ev = float(mm.group(1))
                        break
        except Exception:
            pass
        if result.fermi_ev is not None:
            break

    # Fallback: use HOMO from bandgap result as Fermi energy
    if result.fermi_ev is None and bandgap_result.homo_ev is not None:
        result.fermi_ev = bandgap_result.homo_ev
        log.info("  Fermi energy: using HOMO from bandgap calc (fallback)")

    result.parsed = True
    log.info(f"  Found {result.n_pdos_files} PDOS file(s).")
    if result.fermi_ev is not None:
        log.info(f"  Fermi energy : {result.fermi_ev:.4f} eV")
    for pf in pdos_files:
        log.info(f"    {pf.name}")

    print_dos(result)
    return result


# ============================================================================
# CALCULATION 1 — BANDGAP
# ============================================================================
def calc_bandgap(atoms: Atoms, out_dir: Path,
                 num_cores: int, mpi_ranks: int,
                 use_high_accuracy: bool, fast_mode: bool,
                 user_mult: Optional[int]) -> BandgapResult:
    log.info("")
    log.info("━" * 62)
    log.info("  CALC 1/4 — BANDGAP  (single-point DFT)")
    log.info("━" * 62)

    metals = detect_transition_metals(atoms)
    mult   = user_mult if user_mult else estimate_multiplicity(atoms)
    log.info(f"  Spin multiplicity : {mult} "
             f"({'user' if user_mult else 'auto'})")
    log.info(f"  DFT+U metals      : {list(metals.keys()) or 'none'}")

    label    = "mof_bandgap"
    inp_file = str(out_dir / "bandgap.inp")
    out_file = str(out_dir / "bandgap.out")
    wfn      = str(out_dir / f"{label}-RESTART.wfn")

    if Path(out_file).exists() and check_scf_converged(out_file):
        log.info("  bandgap.out already converged — skipping.")
    else:
        inp = generate_cp2k_input(
            atoms, metals, mult, label,
            run_type="ENERGY",
            use_high_accuracy=use_high_accuracy,
            use_smearing=False,
            fast_mode=fast_mode)
        Path(inp_file).write_text(inp)
        run_cp2k(inp_file, out_file, num_cores, mpi_ranks,
                 restart_wfn=wfn if os.path.exists(wfn) else None,
                 label="Bandgap", max_scf=300)

    t0     = time.time()
    result = extract_bandgap_results(out_file)
    result.elapsed_min = (time.time() - t0) / 60.0

    # Print result IMMEDIATELY
    print_bandgap(result)
    return result


# ============================================================================
# CALCULATION 3 — FORMATION ENERGY (runs right after calc 1)
# ============================================================================
def calc_formation_energy(atoms: Atoms, e_mof: Optional[float],
                           ref_energies: Dict[str, float]) -> FormationResult:
    log.info("")
    log.info("━" * 62)
    log.info("  CALC 3/4 — FORMATION ENERGY  (instant, uses Calc 1 energy)")
    log.info("━" * 62)

    result = FormationResult()
    if e_mof is None:
        log.error("  E(MOF) not available.")
        print_formation(result)
        return result

    counts  = Counter(atoms.get_chemical_symbols())
    n_atoms = len(atoms)
    ref_sum = 0.0
    missing = []

    for elem, cnt in sorted(counts.items()):
        if elem not in ref_energies:
            missing.append(elem)
            result.refs_complete = False
            log.warning(f"  Missing reference: {elem}")
            continue
        e_ref   = ref_energies[elem]
        contrib = cnt * e_ref
        ref_sum += contrib
        log.info(f"  {elem:3s}: {cnt:4d} × {e_ref:.6f} = {contrib:.6f} eV")

    result.missing_elements   = missing
    result.e_form_ev          = e_mof - ref_sum
    result.e_form_per_atom_ev = result.e_form_ev / n_atoms

    log.info(f"  E(MOF)       = {e_mof:.6f} eV")
    log.info(f"  Σ refs       = {ref_sum:.6f} eV")
    log.info(f"  E_form       = {result.e_form_ev:.6f} eV")
    log.info(f"  E_form/atom  = {result.e_form_per_atom_ev:.6f} eV/atom")

    # Print result IMMEDIATELY
    print_formation(result)
    return result


# ============================================================================
# CALCULATION 2 — ADSORPTION ENERGY
# ============================================================================
def calc_adsorption_energy(atoms: Atoms, e_mof: Optional[float],
                            out_dir: Path, num_cores: int, mpi_ranks: int,
                            use_high_accuracy: bool, fast_mode: bool,
                            n_ions: int,
                            ion_symbol: str,
                            cell_opt: bool,
                            ion_ref_ev: float,
                            user_mult: Optional[int]) -> AdsorptionResult:
    log.info("")
    log.info("━" * 62)
    log.info(f"  CALC 2/4 — {ion_symbol.upper()} ADSORPTION ENERGY  ({n_ions}× {ion_symbol})")
    log.info("━" * 62)

    result = AdsorptionResult(n_ions=n_ions, ion_symbol=ion_symbol)
    if e_mof is None:
        log.error("  E(MOF) not available — skipping.")
        print_adsorption(result)
        return result

    # Build MOF+ion structure
    log.info(f"  Searching for {n_ions} void site(s) for {ion_symbol}...")
    try:
        loaded = build_loaded_structure(atoms, ion_symbol, n_ions)
    except RuntimeError as e:
        log.error(f"  {e}")
        print_adsorption(result)
        return result

    write(str(out_dir / f"mof_with_{ion_symbol.lower()}.cif"), loaded)

    metals   = detect_transition_metals(loaded)
    mult     = user_mult if user_mult else estimate_multiplicity(loaded)
    run_type = "CELL_OPT" if cell_opt else "GEO_OPT"
    label    = f"mof_{ion_symbol.lower()}"
    inp_file = str(out_dir / "adsorption.inp")
    out_file = str(out_dir / "adsorption.out")
    wfn      = str(out_dir / f"{label}-RESTART.wfn")
    coord_rs = str(out_dir / f"{label}-1.restart")

    t0 = time.time()

    if Path(out_file).exists() and extract_total_energy(out_file) is not None:
        log.info("  adsorption.out has energy — skipping.")
    else:
        # FIX2: smearing OFF for closed-shell MOF (is_metallic_system returns False)
        inp = generate_cp2k_input(
            loaded, metals, mult, label,
            run_type=run_type,
            use_high_accuracy=use_high_accuracy,
            cell_opt=cell_opt,
            use_smearing=True,   # will only activate if is_metallic_system() == True
            fast_mode=fast_mode)
        Path(inp_file).write_text(inp)

        ok = run_cp2k(
            inp_file, out_file, num_cores, mpi_ranks,
            restart_wfn=wfn if os.path.exists(wfn) else None,
            restart_coord=coord_rs if os.path.exists(coord_rs) else None,
            label=f"GEO_OPT({n_ions}{ion_symbol})",
            max_scf=300, max_geo=300)

        if not ok:
            log.warning("  GEO_OPT failed — trying single-point fallback...")
            inp = generate_cp2k_input(
                loaded, metals, mult, label,
                run_type="ENERGY",
                use_high_accuracy=use_high_accuracy,
                fast_mode=fast_mode)
            Path(inp_file).write_text(inp)
            run_cp2k(inp_file, out_file, num_cores, mpi_ranks,
                     label=f"SP({n_ions}{ion_symbol})", max_scf=300)
            result.relaxed = False
        else:
            result.relaxed = True

    result.elapsed_min  = (time.time() - t0) / 60.0
    e_mof_ion           = extract_total_energy(out_file)
    result.e_mof_ion_ev = e_mof_ion

    if e_mof_ion is None:
        log.error(f"  Could not extract E(MOF+{ion_symbol}).")
        print_adsorption(result)
        return result

    result.e_ads_ev = e_mof_ion - e_mof - n_ions * ion_ref_ev
    log.info(f"  E(MOF)                = {e_mof:.6f} eV")
    log.info(f"  E(MOF+{n_ions}{ion_symbol})       = {e_mof_ion:.6f} eV")
    log.info(f"  E({ion_symbol} ref/atom)   = {ion_ref_ev:.6f} eV")
    log.info(f"  E_ads                 = {result.e_ads_ev:.6f} eV")

    # Print result IMMEDIATELY
    print_adsorption(result)
    return result


# ============================================================================
# CALCULATION 4 — VOLUME EXPANSION (runs right after calc 2)
# ============================================================================
def calc_volume_expansion(atoms: Atoms, out_dir: Path,
                           cell_opt: bool,
                           ion_symbol: str = "Li") -> VolumeResult:
    log.info("")
    log.info("━" * 62)
    log.info("  CALC 4/4 — VOLUME EXPANSION  (instant, uses Calc 2 output)")
    log.info("━" * 62)

    result             = VolumeResult(cell_relaxed=cell_opt)
    result.v_before_A3 = atoms.get_volume()
    out_file           = str(out_dir / "adsorption.out")

    v_after = extract_final_volume(out_file)

    # Fallback: read from trajectory file (ion-symbol-aware)
    if v_after is None:
        for traj in [out_dir / f"mof_{ion_symbol.lower()}-pos-1.xyz",
                     out_dir / "cp2k-pos-1.xyz"]:
            if traj.exists():
                try:
                    v_after = read(str(traj), index=-1).get_volume()
                    log.info(f"  Volume from trajectory: {traj.name}")
                    break
                except Exception:
                    pass

    if v_after is None and not cell_opt:
        log.info("  Fixed-cell GEO_OPT — volume expansion = 0% by definition.")
        log.info("  Use --cell-opt for true volume relaxation.")
        v_after = result.v_before_A3

    if v_after is None:
        log.error("  Could not determine relaxed volume.")
        print_volume(result, ion_symbol)
        return result

    result.v_after_A3    = v_after
    result.expansion_pct = (v_after - result.v_before_A3) / result.v_before_A3 * 100.0

    log.info(f"  V(bare)    = {result.v_before_A3:.4f} Å³")
    log.info(f"  V(with {ion_symbol}) = {v_after:.4f} Å³")
    log.info(f"  Expansion  = {result.expansion_pct:.3f} %")

    # Print result IMMEDIATELY
    print_volume(result, ion_symbol)
    return result


# ============================================================================
# ELEMENTAL REFERENCE CALCULATIONS
# ============================================================================
def _reference_structure(elem: str) -> Optional[Tuple[Atoms, int]]:
    """
    Return (Atoms, multiplicity) for an elemental reference calculation.
    FIX6: Correct multiplicities for O2(3), etc.
    Supports Li, Na, K, Mg, Ca, Zn, Al in addition to standard elements.
    """
    try:
        if elem in ("H", "N", "O", "Cl", "F"):
            mol_map  = {"H": "H2", "N": "N2", "O": "O2", "Cl": "Cl2", "F": "F2"}
            ref      = molecule(mol_map[elem])
            ref.set_cell([15.0, 16.0, 17.0])
            ref.center()
            ref.set_pbc(True)
            mult = MOLECULE_MULTIPLICITY.get(mol_map[elem], 1)
        elif elem == "C":
            ref  = bulk("C", "diamond", a=3.567)
            mult = 1
        elif elem in ("Li", "Na", "K"):
            # Alkali metals: BCC structure
            a = {"Li": 3.51, "Na": 4.29, "K": 5.23}.get(elem, 3.5)
            ref  = bulk(elem, "bcc", a=a)
            mult = 2
        elif elem in ("Mg", "Ca", "Zn", "Al"):
            # Alkaline-earth / post-transition metals
            ref  = bulk(elem, orthorhombic=True)
            mult = UNPAIRED_PER_SITE.get(elem, 0) + 1
        else:
            ref  = bulk(elem, orthorhombic=True)
            mult = estimate_multiplicity(ref)
        return ref, mult
    except Exception as e:
        log.error(f"  Cannot build reference for {elem}: {e}")
        return None


def compute_elemental_reference(elem: str, ref_dir: Path,
                                 num_cores: int, mpi_ranks: int,
                                 use_high_accuracy: bool,
                                 fast_mode: bool) -> Optional[float]:
    """Run CP2K on reference structure, return eV/atom."""
    ret = _reference_structure(elem)
    if ret is None:
        return None
    ref_atoms, mult = ret

    label    = f"ref_{elem}"
    inp_file = str(ref_dir / f"{label}.inp")
    out_file = str(ref_dir / f"{label}.out")

    # Skip if already computed
    if Path(out_file).exists():
        e = extract_total_energy(out_file)
        if e is not None:
            per_atom = e / len(ref_atoms)
            log.info(f"  {elem}: {per_atom:.6f} eV/atom (cached)")
            return per_atom

    metals = detect_transition_metals(ref_atoms)
    inp    = generate_cp2k_input(
        ref_atoms, metals, mult, label,
        run_type="ENERGY",
        use_high_accuracy=use_high_accuracy,
        fast_mode=fast_mode)
    Path(inp_file).write_text(inp)

    ok = run_cp2k(inp_file, out_file, num_cores, mpi_ranks,
                  label=f"Ref-{elem}", max_scf=300)
    if not ok:
        log.error(f"  Reference calc for {elem} failed.")
        return None

    energy_ev = extract_total_energy(out_file)
    if energy_ev is None:
        return None

    per_atom = energy_ev / len(ref_atoms)
    log.info(f"  {elem}: {per_atom:.6f} eV/atom")
    return per_atom


def compute_all_references(elements: set, out_dir: Path,
                            num_cores: int, mpi_ranks: int,
                            use_high_accuracy: bool,
                            fast_mode: bool) -> Dict[str, float]:
    ref_dir = out_dir / "elemental_refs"
    ref_dir.mkdir(exist_ok=True)
    refs: Dict[str, float] = {}

    log.info(f"  Computing references for: {sorted(elements)}")
    for elem in sorted(elements):
        log.info(f"  --- Reference: {elem} ---")
        e = compute_elemental_reference(
            elem, ref_dir, num_cores, mpi_ranks, use_high_accuracy, fast_mode)
        if e is not None:
            refs[elem] = e

    json_path = ref_dir / "ref_energies.json"
    json_path.write_text(json.dumps(refs, indent=2))
    log.info(f"  References saved: {json_path}")
    return refs


def load_reference_energies(ref_file: Path) -> Dict[str, float]:
    if not ref_file.exists():
        return {}
    return json.loads(ref_file.read_text())


# ============================================================================
# MAIN SCREENING WORKFLOW
# ============================================================================
def run_screening(cif_path: str,
                  num_cores: int          = 16,
                  mpi_ranks: int          = 1,
                  use_high_accuracy: bool = False,
                  fast_mode: bool         = False,
                  ion_symbol: str         = "Li",
                  n_ions: int             = 1,
                  cell_opt: bool          = False,
                  user_mult: Optional[int] = None,
                  ref_energies_file: Optional[str] = None,
                  compute_refs: bool      = False,
                  ion_ref_ev: Optional[float] = None,
                  cp2k_data_dir: Optional[str] = None,
                  barrier_file: Optional[str] = None,
                  compute_dos: bool        = False) -> AllResults:

    # Validate ion type
    supported_ions = ["Li", "Na", "K", "Mg", "Ca", "Zn", "Al"]
    if ion_symbol not in supported_ions:
        raise ValueError(f"ion_symbol must be one of {supported_ions}, got '{ion_symbol}'")

    if cp2k_data_dir:
        os.environ["CP2K_DATA_DIR"] = cp2k_data_dir

    cif_path = os.path.abspath(cif_path)
    out_dir  = Path(os.path.dirname(cif_path)) / "results"
    out_dir.mkdir(exist_ok=True)

    # Add file log handler (only once)
    log_path = out_dir / "run.log"
    if not any(isinstance(h, logging.FileHandler) for h in log.handlers):
        fh = logging.FileHandler(str(log_path))
        fh.setFormatter(logging.Formatter(
            "%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S"))
        log.addHandler(fh)

    log.info("═" * 65)
    log.info(f"  MOF DFT SCREENING — {ion_symbol.upper()}-ION ANODE PROPERTIES")
    log.info("═" * 65)
    log.info(f"  CIF          : {cif_path}")
    log.info(f"  Output dir   : {out_dir}")
    log.info(f"  OMP threads  : {num_cores}  |  MPI ranks: {mpi_ranks}")
    log.info(f"  Accuracy     : {'TZV2P' if use_high_accuracy else 'DZVP'}"
             f"{'  [FAST MODE]' if fast_mode else ''}")
    log.info(f"  Ion type     : {ion_symbol}  |  Ions: {n_ions}  |  Cell opt: {cell_opt}")
    log.info(f"  Log file     : {log_path}")

    # ---- Load and validate structure ----
    atoms   = read(cif_path)
    formula = atoms.get_chemical_formula()
    n_atoms = len(atoms)

    log.info(f"  Formula      : {formula}  ({n_atoms} atoms)")
    log.info(f"  Cell (Å)     : {np.round(atoms.cell.lengths(), 3)}")
    log.info(f"  Volume       : {atoms.get_volume():.3f} Å³")
    log.info(f"  Elements     : {', '.join(sorted(set(atoms.get_chemical_symbols())))}")

    if atoms.get_volume() <= 0:
        raise ValueError("CIF has zero or negative volume — aborting.")
    min_d = atoms.get_all_distances()[np.nonzero(atoms.get_all_distances())].min()
    if min_d < 0.8:
        log.warning(f"  Very short bond: {min_d:.3f} Å — CIF may be unrelaxed!")

    # ---- Reference energies ----
    refs: Dict[str, float] = {}

    if compute_refs:
        elements = set(atoms.get_chemical_symbols()) | {ion_symbol}
        log.info(f"  Computing self-consistent references for {sorted(elements)}...")
        refs = compute_all_references(
            elements, out_dir, num_cores, mpi_ranks, use_high_accuracy, fast_mode)
    else:
        # Try loading from file argument, then default location
        rf = (Path(ref_energies_file) if ref_energies_file
              else out_dir / "elemental_refs" / "ref_energies.json")
        refs = load_reference_energies(rf)
        if refs:
            log.info(f"  Loaded {len(refs)} reference energies from {rf}")

    # Resolve ion reference energy
    if ion_symbol in refs:
        ion_ref = refs[ion_symbol]
        log.info(f"  {ion_symbol} reference : {ion_ref:.6f} eV/atom (from refs)")
    elif ion_ref_ev is not None:
        ion_ref = ion_ref_ev
        log.info(f"  {ion_symbol} reference : {ion_ref:.6f} eV/atom (user override)")
    else:
        # FIX7: loud error instead of silent fallback
        ion_ref = APPROX_ION_REFS.get(ion_symbol, -1.5)
        log.error(f"  ⚠ NO {ion_symbol} reference energy found!")
        log.error("    Adsorption energy will be approximate.")
        log.error(f"    Run with --compute-refs or --ion-ref-ev for accuracy.")
        log.warning(f"  Using APPROXIMATE {ion_symbol} ref = {ion_ref:.3f} eV/atom")

    results = AllResults(formula=formula, n_atoms=n_atoms, cif_path=cif_path,
                         ion_symbol=ion_symbol)

    # ================================================================
    # SEQUENTIAL EXECUTION WITH IMMEDIATE RESULT PRINTING
    # Calc 3 runs right after calc 1 (no DFT needed)
    # Calc 4 runs right after calc 2 (no DFT needed)
    # ================================================================

    log.info("")
    log.info("  Calculation plan:")
    log.info("    1. Bandgap (DFT single-point, ~10-60 min)")
    log.info("    → 3. Formation energy (instant, uses calc 1)")
    log.info(f"    2. {ion_symbol} Adsorption energy (GEO_OPT, ~30-120 min)")
    log.info("    → 4. Volume expansion (instant, uses calc 2)")
    log.info("    → 5. Open-circuit voltage (instant, derived from E_ads)")
    log.info("    6. Diffusion barrier (from NEB file, if provided)")
    log.info("    7. Density of states (parse PDOS files, if available)")
    log.info("")

    # ---- CALC 1: BANDGAP ----
    results.bandgap = calc_bandgap(
        atoms, out_dir, num_cores, mpi_ranks,
        use_high_accuracy, fast_mode, user_mult)

    # ---- CALC 3: FORMATION ENERGY (instant — runs right after calc 1) ----
    results.formation = calc_formation_energy(
        atoms, results.bandgap.total_energy_ev, refs)

    # ---- CALC 2: ADSORPTION ENERGY ----
    results.adsorption = calc_adsorption_energy(
        atoms, results.bandgap.total_energy_ev,
        out_dir, num_cores, mpi_ranks,
        use_high_accuracy, fast_mode,
        n_ions, ion_symbol, cell_opt, ion_ref, user_mult)

    # ---- CALC 4: VOLUME EXPANSION (instant — runs right after calc 2) ----
    results.volume = calc_volume_expansion(atoms, out_dir, cell_opt, ion_symbol)

    # ---- CALC 5: OPEN-CIRCUIT VOLTAGE (instant — derived from E_ads) ----
    results.ocv = calc_ocv(results.adsorption)

    # ---- CALC 6: DIFFUSION BARRIER (reads pre-computed NEB file) ----
    results.diffusion_barrier = calc_diffusion_barrier(barrier_file)

    # ---- CALC 7: DENSITY OF STATES (parse PDOS files from calc 1) ----
    if compute_dos:
        results.dos = calc_dos(out_dir, results.bandgap)
    else:
        log.info("  DOS parsing skipped (use --dos to enable).")

    # ---- Save JSON summary ----
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(asdict(results), indent=2, default=str))
    log.info(f"\n  All results saved to: {summary_path}")

    # ---- Final compact summary ----
    print("\n" + "═" * 65)
    print("  FINAL SUMMARY")
    print("═" * 65)
    print(f"  MOF: {formula}  ({n_atoms} atoms)  |  Ion: {ion_symbol}×{n_ions}")
    print(f"  ─────────────────────────────────────────────────────────────")

    bg = results.bandgap
    if bg.bandgap_ev is not None:
        print(f"  Bandgap        : {bg.bandgap_ev:.4f} eV  [{bg.classification}]")
    else:
        print("  Bandgap        : NOT FOUND")

    ad = results.adsorption
    if ad.e_ads_ev is not None:
        print(f"  E_ads ({ion_symbol}×{n_ions})   : {ad.e_ads_ev:.4f} eV")
    else:
        print("  E_ads          : NOT COMPUTED")

    fm = results.formation
    if fm.e_form_per_atom_ev is not None:
        flag = " ⚠ approx" if not fm.refs_complete else ""
        print(f"  E_form/atom    : {fm.e_form_per_atom_ev:.4f} eV/atom{flag}")
    else:
        print("  E_form/atom    : NOT COMPUTED")

    vl = results.volume
    if vl.expansion_pct is not None:
        print(f"  Volume exp.    : {vl.expansion_pct:.3f} %")
    else:
        print("  Volume exp.    : NOT COMPUTED")

    ov = results.ocv
    if ov.ocv_v is not None:
        print(f"  OCV            : {ov.ocv_v:.4f} V  (vs {ion_symbol}+/{ion_symbol})")
    else:
        print("  OCV            : NOT COMPUTED")

    db = results.diffusion_barrier
    if db.available and db.barrier_ev is not None:
        print(f"  Diff. barrier  : {db.barrier_ev:.4f} eV")
    else:
        print("  Diff. barrier  : NOT PROVIDED (use --barrier-file)")

    ds = results.dos
    if ds.parsed:
        fermi_str = f"{ds.fermi_ev:.4f} eV" if ds.fermi_ev is not None else "N/A"
        print(f"  DOS (Fermi)    : {fermi_str}  [{ds.n_pdos_files} PDOS files]")
    else:
        print("  DOS            : NOT PARSED (use --dos)")

    print(f"  ─────────────────────────────────────────────────────────────")
    print(f"  Output dir     : {out_dir}")
    print("═" * 65)

    return results


# ============================================================================
# CLI
# ============================================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Production DFT screening of MOFs for alkali/alkaline-earth ion anode materials.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python mof_screen.py mof_relaxed.cif --cores 16
  python mof_screen.py mof_relaxed.cif --ion K --n-ions 4 --cell-opt
  python mof_screen.py mof_relaxed.cif --cores 4 --mpi-ranks 4
  python mof_screen.py mof_relaxed.cif --cores 16 --compute-refs
  python mof_screen.py mof_relaxed.cif --cores 16 --fast
        """)

    parser.add_argument("cif_file", help="Path to relaxed CIF file")
    parser.add_argument("--cores", "-n", type=int, default=16,
                        help="OMP_NUM_THREADS per process (default: 16)")
    parser.add_argument("--mpi-ranks", type=int, default=1,
                        help="Number of MPI ranks (default: 1)")
    parser.add_argument("--high-accuracy", action="store_true",
                        help="Use TZV2P basis set (3-5x slower, publication quality)")
    parser.add_argument("--fast", action="store_true",
                        help="Fast mode: lower cutoffs (400 Ry) for quick screening")
    parser.add_argument("--ion", type=str, default="Li",
                        choices=["Li", "Na", "K", "Mg", "Ca", "Zn", "Al"],
                        help="Ion to insert (default: Li)")
    parser.add_argument("--n-ions", type=int, default=1,
                        help="Number of ions to insert (default: 1)")
    parser.add_argument("--n-li", type=int, dest="n_ions_deprecated",
                        help=argparse.SUPPRESS)
    parser.add_argument("--li-ref-ev", type=float, dest="ion_ref_ev_deprecated",
                        help=argparse.SUPPRESS)
    parser.add_argument("--ion-ref-ev", type=float,
                        help="Bulk reference energy eV/atom for the chosen ion")
    parser.add_argument("--cell-opt", action="store_true",
                        help="Relax cell vectors during adsorption")
    parser.add_argument("--multiplicity", type=int,
                        help="Spin multiplicity override (default: auto-detect)")
    parser.add_argument("--ref-energies", type=str,
                        help="JSON file with pre-computed elemental reference energies")
    parser.add_argument("--compute-refs", action="store_true",
                        help="Run CP2K to compute self-consistent elemental references")
    parser.add_argument("--barrier-file", type=str, default=None,
                        help="Path to pre-computed NEB barrier file (plain float eV or CP2K NEB output)")
    parser.add_argument("--dos", action="store_true",
                        help="Parse CP2K PDOS files for density of states (produced by bandgap calc)")

    args = parser.parse_args()

    n_ions = args.n_ions
    if args.n_ions_deprecated is not None:
        log.warning("--n-li is deprecated, use --n-ions instead.")
        n_ions = args.n_ions_deprecated

    ion_ref_ev = args.ion_ref_ev
    if args.ion_ref_ev_deprecated is not None:
        log.warning("--li-ref-ev is deprecated, use --ion-ref-ev instead.")
        ion_ref_ev = args.ion_ref_ev_deprecated

    if not os.path.exists(args.cif_file):
        log.error(f"CIF not found: {args.cif_file}")
        sys.exit(1)

    run_screening(
        cif_path          = args.cif_file,
        num_cores         = args.cores,
        mpi_ranks         = args.mpi_ranks,
        use_high_accuracy = args.high_accuracy,
        fast_mode         = args.fast,
        ion_symbol        = args.ion,
        n_ions            = n_ions,
        cell_opt          = args.cell_opt,
        user_mult         = args.multiplicity,
        ref_energies_file = args.ref_energies,
        compute_refs      = args.compute_refs,
        ion_ref_ev        = ion_ref_ev,
        barrier_file      = args.barrier_file,
        compute_dos       = args.dos,
    )