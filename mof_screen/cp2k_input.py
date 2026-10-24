"""
mof_screen/cp2k_input.py
=========================
CP2K input file generator for the MOF K⁺-anode screening pipeline.

All functions are pure (no I/O, no global state) — they return strings.
The caller is responsible for writing them to disk.

Key correctness points (preserved from _core.py):
  - K-points only for small reference cells; Gamma-only for MOFs
  - Stress tensor only for CELL_OPT
  - Smearing only when explicitly requested AND system is metallic
  - DFT+U (Hubbard) applied per TM element
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
from ase import Atoms

from mof_screen.constants import (
    BASIS_SETS,
    DEFAULT_CUTOFF,
    DEFAULT_REL_CUTOFF,
    FAST_CUTOFF,
    FAST_REL_CUTOFF,
    MAX_SCF_STEPS,
    KPOINT_MAX_CELL_LENGTH,
    KPOINT_TARGET_PERIOD,
    SCREENING_CUTOFF,
    SCREENING_REL_CUTOFF,
    SCREENING_EPS_SCF,
    SCREENING_MAX_SCF,
    SCREENING_MAX_GEO,
    SCREENING_MAX_FORCE,
    SCREENING_MAX_DR,
)

from mof_screen.structure import is_metallic_system

log = logging.getLogger("mof_dft.cp2k_input")


# ---------------------------------------------------------------------------
# K-point grid
# ---------------------------------------------------------------------------
def compute_kpoint_grid(cell_lengths: np.ndarray) -> Tuple[int, int, int]:
    """
    Return Monkhorst-Pack grid dimensions.

    K-points are applied ONLY when ALL cell axes are below KPOINT_MAX_CELL_LENGTH
    (i.e. small elemental reference cells like Zr at ~3 Å, K-BCC at ~5 Å).
    Any MOF-sized cell (>10 Å axes) returns (1,1,1) → Gamma only.

    This avoids the CP2K 2024.2 crash with ADDED_MOS + UKS + k-points.
    """
    if np.any(cell_lengths >= KPOINT_MAX_CELL_LENGTH):
        return (1, 1, 1)
    grid = [max(1, int(round(KPOINT_TARGET_PERIOD / L))) for L in cell_lengths]
    return tuple(grid)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Sub-blocks
# ---------------------------------------------------------------------------
def _kpoints_block(k1: int, k2: int, k3: int) -> str:
    if k1 == 1 and k2 == 1 and k3 == 1:
        return ""
    return (
        f"\n    &KPOINTS\n"
        f"      SCHEME MONKHORST-PACK {k1} {k2} {k3}\n"
        f"    &END KPOINTS\n"
    )


def _subsys_block(atoms: Atoms, basis: str, metals: Dict[str, float] = None) -> str:
    if metals is None:
        metals = {}
    cell  = atoms.get_cell()
    lines = ["  &SUBSYS", "    &CELL"]
    for name, vec in zip("ABC", cell):
        lines.append(
            f"      {name}  {vec[0]:.10f}  {vec[1]:.10f}  {vec[2]:.10f}"
        )
    lines += ["      PERIODIC XYZ", "    &END CELL", "    &COORD"]
    for a in atoms:
        p = a.position
        lines.append(f"      {a.symbol:4s}  {p[0]:.10f}  {p[1]:.10f}  {p[2]:.10f}")
    lines.append("    &END COORD")
    for sym in sorted(set(atoms.get_chemical_symbols())):
        lines += [
            f"    &KIND {sym}",
            f"      BASIS_SET {basis}",
            f"      POTENTIAL GTH-PBE",
        ]
        if sym in metals:
            lines += [
                f"      &DFT_PLUS_U",
                f"        L 2",
                f"        U_MINUS_J [eV] {metals[sym]:.4f}",
                f"      &END DFT_PLUS_U",
            ]
        lines.append(f"    &END KIND")
    lines.append("  &END SUBSYS")
    return "\n".join(lines)


def _motion_block(
    cell_opt: bool,
    max_geo: int    = 300,
    max_force: float = 0.0005,
    max_dr: float    = 0.0030,
) -> str:
    """
    Generate the &MOTION block for GEO_OPT / CELL_OPT.

    Parameters
    ----------
    cell_opt  : relax cell vectors (CELL_OPT) vs fixed-cell (GEO_OPT)
    max_geo   : maximum number of geometry optimisation steps
    max_force : convergence criterion on maximum force (Ha/Bohr)
                  0.0005  = production (tight)
                  0.001   = screening  (loose, ~30-50% fewer GEO steps)
    max_dr    : convergence criterion on maximum atomic displacement (Bohr)
                  0.003   = production
                  0.006   = screening
    """
    constraint = "NONE"
    rms_force  = max_force * 0.6     # RMS is always 60% of MAX criterion
    rms_dr     = max_dr    * 0.5
    return f"""
&MOTION
  &GEO_OPT
    TYPE MINIMIZATION
    MAX_ITER {max_geo}
    OPTIMIZER BFGS
    MAX_DR    {max_dr:.4f}
    MAX_FORCE {max_force:.4f}
    RMS_DR    {rms_dr:.4f}
    RMS_FORCE {rms_force:.4f}
  &END GEO_OPT
  &CELL_OPT
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


# ---------------------------------------------------------------------------
# Main input generator
# ---------------------------------------------------------------------------
def generate_cp2k_input(
    atoms: Atoms,
    metals: Dict[str, float],
    multiplicity: int,
    label: str,
    run_type: str           = "ENERGY",
    use_high_accuracy: bool = False,
    cell_opt: bool          = False,
    use_smearing: bool      = False,
    fast_mode: bool         = False,
    screening_mode: bool    = False,
) -> str:
    """
    Generate a complete, correct CP2K input string.

    Parameters
    ----------
    atoms           : ASE Atoms (MOF with or without inserted ion)
    metals          : {element: U_eff_eV} from detect_transition_metals()
    multiplicity    : spin multiplicity (auto or user-supplied)
    label           : CP2K project name (used for output file names)
    run_type        : "ENERGY" | "GEO_OPT" | "CELL_OPT"
    use_high_accuracy : True → TZV2P basis, False → DZVP (default)
    cell_opt        : True → include STRESS_TENSOR + CELL_OPT block
    use_smearing    : True → Fermi smearing (only activates for metallic systems)
    fast_mode       : True → 400 Ry cutoff only (quick single-step estimates)
    screening_mode  : True → ALL 5 speed-up settings applied:
                       - 400 Ry cutoff  (biggest speedup, ~30% per SCF step)
                       - EPS_SCF 1E-6   (saves 10-20 SCF iters per GEO step)
                       - MAX_FORCE 0.001 (50% fewer GEO_OPT steps typically)
                       - MAX_GEO 150    (hard cap)
                       - MAX_SCF 150    (hard cap, safety net)
                      Expected speedup vs production: 40-60%
                      E_ads accuracy: ±0.05-0.10 eV (vs ±0.01 eV production)
                      Good for: ranking/filtering MOFs
                      NOT for: publication or final validation
    """
    basis      = BASIS_SETS["high_accuracy"] if use_high_accuracy else BASIS_SETS["default"]
    use_uks    = multiplicity > 1

    # ---- Resolve numerical settings based on mode priority ----
    # Priority: screening_mode > fast_mode > production defaults
    if screening_mode:
        cutoff     = SCREENING_CUTOFF
        rel_cutoff = SCREENING_REL_CUTOFF
        eps_scf    = SCREENING_EPS_SCF
        max_scf    = SCREENING_MAX_SCF
        max_geo    = SCREENING_MAX_GEO
        max_force  = SCREENING_MAX_FORCE
        max_dr     = SCREENING_MAX_DR
        log.info("    [SCREENING MODE] 400 Ry | EPS_SCF=1E-6 | MAX_FORCE=0.001 | MAX_GEO=150")
    elif fast_mode:
        cutoff     = FAST_CUTOFF
        rel_cutoff = FAST_REL_CUTOFF
        eps_scf    = 1.0e-7      # keep tight SCF even in fast mode
        max_scf    = MAX_SCF_STEPS
        max_geo    = 300
        max_force  = 0.0005
        max_dr     = 0.003
        log.info("    [FAST MODE] 400 Ry cutoff only")
    else:
        cutoff     = DEFAULT_CUTOFF
        rel_cutoff = DEFAULT_REL_CUTOFF
        eps_scf    = 1.0e-7
        max_scf    = MAX_SCF_STEPS
        max_geo    = 300
        max_force  = 0.0005
        max_dr     = 0.003

    # K-points — Gamma for large (MOF) cells
    k1, k2, k3 = compute_kpoint_grid(atoms.cell.lengths())
    if metals:
        # CP2K does not support DFT+U with k-points, so force Gamma point
        k1, k2, k3 = 1, 1, 1
        log.info("    [INFO] Forcing Gamma point (1x1x1) because DFT+U is active.")
        
    kpts        = _kpoints_block(k1, k2, k3)
    if k1 > 1 or k2 > 1 or k3 > 1:
        log.info(
            f"    K-point grid: {k1}×{k2}×{k3} "
            f"(cell {atoms.cell.lengths()[0]:.1f}×"
            f"{atoms.cell.lengths()[1]:.1f}×"
            f"{atoms.cell.lengths()[2]:.1f} Å)"
        )

    # Smearing — only for genuinely metallic systems
    use_smearing_active = use_smearing and is_metallic_system(atoms)
    smear_block = ""
    if use_smearing_active:
        smear_block = """
      &SMEAR
        METHOD FERMI_DIRAC
        ELECTRONIC_TEMPERATURE [K] 300
      &END SMEAR"""

    # ── SCF algorithm selector ────────────────────────────────────────────────
    # OT (Orbital Transformation) + CG (Conjugate Gradient) is unconditionally
    # stable for gapped (insulating/semiconducting) MOFs. CG + LINESEARCH 2PNT
    # guarantees monotonic energy decrease — eliminates DIIS oscillation failures.
    # OT is INCOMPATIBLE with fractional occupations (Fermi smearing), so we
    # fall back to Broyden+Diagonalization only when smearing is active (metallic).
    if use_smearing_active:
        # Metallic system — must use Diagonalization (OT does not support ADDED_MOS)
        log.info("    [SCF] Metallic system detected → Broyden/Diagonalization")
        added_mos = 50 if run_type != "ENERGY" else 0
        scf_algo_block = f"""      ADDED_MOS  {added_mos}
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
        EPS_SCF   {eps_scf:.1E}
        OPTIMIZER DIIS
      &END OUTER_SCF
      {smear_block}"""
    else:
        # Insulating/open-shell MOF — OT + CG guarantees monotonic convergence
        log.info("    [SCF] Insulating/open-shell MOF → OT + Conjugate Gradient")
        scf_algo_block = """      SCF_GUESS  ATOMIC
      &OT ON
        MINIMIZER    CG
        LINESEARCH   2PNT
        PRECONDITIONER FULL_SINGLE_INVERSE
        ENERGY_GAP   0.05
      &END OT"""

    # Motion block (geometry optimisation) — passes screening convergence params
    motion = (
        _motion_block(cell_opt, max_geo=max_geo, max_force=max_force, max_dr=max_dr)
        if run_type in ("GEO_OPT", "CELL_OPT")
        else ""
    )

    # Stress tensor only for cell optimisation
    stress = "STRESS_TENSOR ANALYTICAL" if cell_opt else ""

    # PDOS + Cube + Hirshfeld: print only for ENERGY runs (avoids huge files on every GEO step)
    # All three live inside a single &DFT / &PRINT block to avoid parse errors.
    is_energy_run = (run_type == "ENERGY")
    dft_print_block = ""
    if is_energy_run:
        dft_print_block = """
    &PRINT
      &HIRSHFELD ON
        SELF_CONSISTENT TRUE
      &END HIRSHFELD
      &MULLIKEN ON
      &END MULLIKEN
      &DOS
        NLUMO      -1
      &END DOS
      &E_DENSITY_CUBE
        STRIDE 2 2 2
      &END E_DENSITY_CUBE
    &END PRINT"""

    outer_eps = eps_scf   # outer SCF uses same threshold as inner

    return f"""! CP2K input — mof_screen package
! Label: {label} | Run: {run_type} | Basis: {basis}
! Mode: {'SCREENING' if screening_mode else 'FAST' if fast_mode else 'PRODUCTION'}
! UKS: {use_uks} | Mult: {multiplicity} | DFT+U: {bool(metals)}
! Smearing: {use_smearing and is_metallic_system(atoms)} | K-pts: {k1}x{k2}x{k3}
! EPS_SCF: {eps_scf:.0e} | MAX_SCF: {max_scf} | MAX_GEO: {max_geo} | MAX_FORCE: {max_force}
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
      MAX_SCF    {max_scf}
      EPS_SCF    {eps_scf:.1E}
      IGNORE_CONVERGENCE_FAILURE TRUE
{scf_algo_block}
      &PRINT
        &RESTART
          BACKUP_COPIES 2
          &EACH
            QS_SCF 20
          &END EACH
        &END RESTART
      &END PRINT
    &END SCF
{dft_print_block}
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
{kpts}
  &END DFT
{_subsys_block(atoms, basis, metals)}
&END FORCE_EVAL
"""
