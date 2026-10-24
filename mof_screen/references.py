"""
mof_screen/references.py
========================
Elemental reference energy computations.

Provides both a self-consistent DFT path (--compute-refs) and an
approximate fallback (APPROX_ION_REFS). Results are cached in
  <results_dir>/elemental_refs/ref_energies.json
so they are re-used across multiple MOF screenings in the same project.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

from ase import Atoms
from ase.build import bulk, molecule

from mof_screen.constants import (
    APPROX_ION_REFS,
    MOLECULE_MULTIPLICITY,
    UNPAIRED_PER_SITE,
)
from mof_screen.cp2k_input import generate_cp2k_input
from mof_screen.cp2k_runner import run_cp2k
from mof_screen.parsers import extract_total_energy
from mof_screen.structure import detect_transition_metals, estimate_multiplicity

log = logging.getLogger("mof_dft.references")


# ---------------------------------------------------------------------------
# Reference structure builder
# ---------------------------------------------------------------------------
def _reference_structure(elem: str) -> Optional[Tuple[Atoms, int]]:
    """
    Build an ASE Atoms object and spin multiplicity for a standard
    elemental reference calculation.

    Supports: H, C, N, O, Cl, F (diatomic molecules)
              Li, Na, K (BCC bulk)
              Mg, Ca, Zn, Al (bulk)
              All other elements: orthorhombic bulk via ASE

    Returns (atoms, multiplicity) or None on failure.
    """
    try:
        if elem in ("H", "N", "O", "Cl", "F"):
            mol_map = {"H": "H2", "N": "N2", "O": "O2", "Cl": "Cl2", "F": "F2"}
            ref = molecule(mol_map[elem])
            ref.set_cell([15.0, 16.0, 17.0])
            ref.center()
            ref.set_pbc(True)
            mult = MOLECULE_MULTIPLICITY.get(mol_map[elem], 1)

        elif elem == "C":
            ref  = bulk("C", "diamond", a=3.567)
            mult = 1

        elif elem in ("Li", "Na", "K"):
            a    = {"Li": 3.51, "Na": 4.29, "K": 5.23}[elem]
            ref  = bulk(elem, "bcc", a=a)
            mult = 2

        elif elem in ("Mg", "Ca", "Zn", "Al"):
            ref  = bulk(elem, orthorhombic=True)
            mult = UNPAIRED_PER_SITE.get(elem, 0) + 1

        else:
            ref  = bulk(elem, orthorhombic=True)
            mult = estimate_multiplicity(ref)

        return ref, mult

    except Exception as exc:
        log.error(f"  Cannot build reference structure for {elem}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Single-element reference calculation
# ---------------------------------------------------------------------------
def compute_elemental_reference(
    elem: str,
    ref_dir: Path,
    num_cores: int,
    mpi_ranks: int,
    use_high_accuracy: bool,
    fast_mode: bool,
) -> Optional[float]:
    """
    Run CP2K on the elemental reference structure and return eV/atom.
    Skips the DFT run if a valid cached output already exists.
    """
    ret = _reference_structure(elem)
    if ret is None:
        return None
    ref_atoms, mult = ret

    label    = f"ref_{elem}"
    inp_file = str(ref_dir / f"{label}.inp")
    out_file = str(ref_dir / f"{label}.out")

    # ---- Cache check ----
    if Path(out_file).exists():
        e = extract_total_energy(out_file)
        if e is not None:
            per_atom = e / len(ref_atoms)
            log.info(f"  {elem}: {per_atom:.6f} eV/atom  (cached)")
            return per_atom

    # Elemental bulk references always use k-points.
    # CP2K strictly forbids OT with k-points → must use DIAG + Fermi smearing.
    metals = {}
    inp    = generate_cp2k_input(
        ref_atoms, metals, mult, label,
        run_type="ENERGY",
        use_high_accuracy=use_high_accuracy,
        fast_mode=fast_mode,
        screening_mode=True,
        use_smearing=True,   # forces DIAG solver; OT is incompatible with k-points
    )
    Path(inp_file).write_text(inp)

    ok = run_cp2k(
        inp_file, out_file,
        num_cores, mpi_ranks,
        label=f"Ref-{elem}",
    )
    if not ok:
        log.error(f"  Reference calculation for {elem} failed.")
        return None

    energy_ev = extract_total_energy(out_file)
    if energy_ev is None:
        return None

    per_atom = energy_ev / len(ref_atoms)
    log.info(f"  {elem}: {per_atom:.6f} eV/atom")
    return per_atom


# ---------------------------------------------------------------------------
# Batch reference computation
# ---------------------------------------------------------------------------
def compute_all_references(
    elements: Set[str],
    out_dir: Path,
    num_cores: int,
    mpi_ranks: int,
    use_high_accuracy: bool,
    fast_mode: bool,
) -> Dict[str, float]:
    """
    Compute DFT reference energies (eV/atom) for all elements in the set.
    Results are saved to <out_dir>/elemental_refs/ref_energies.json.
    """
    ref_dir = out_dir / "elemental_refs"
    ref_dir.mkdir(exist_ok=True)
    refs: Dict[str, float] = {}

    log.info(f"  Computing references for: {sorted(elements)}")
    for elem in sorted(elements):
        log.info(f"  --- Reference: {elem} ---")
        e = compute_elemental_reference(
            elem, ref_dir, num_cores, mpi_ranks, use_high_accuracy, fast_mode
        )
        if e is not None:
            refs[elem] = e

    json_path = ref_dir / "ref_energies.json"
    json_path.write_text(json.dumps(refs, indent=2))
    log.info(f"  References saved: {json_path}")
    return refs


# ---------------------------------------------------------------------------
# Load pre-computed references
# ---------------------------------------------------------------------------
def load_reference_energies(ref_file: Path) -> Dict[str, float]:
    """Load reference energies from a JSON file. Returns {} if not found."""
    if not ref_file.exists():
        return {}
    try:
        return json.loads(ref_file.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        log.warning(f"  Could not load reference file {ref_file}: {exc}")
        return {}


# ---------------------------------------------------------------------------
# Ion reference energy resolver
# ---------------------------------------------------------------------------
def resolve_ion_reference(
    ion_symbol: str,
    refs: Dict[str, float],
    ion_ref_ev: Optional[float] = None,
) -> float:
    """
    Resolve the bulk reference energy (eV/atom) for the chosen ion.

    Priority:
      1. Self-consistent DFT value from refs dict (--compute-refs)
      2. User-provided override (--ion-ref-ev)
      3. Approximate tabulated value (APPROX_ION_REFS) — loud warning
    """
    if ion_symbol in refs:
        val = refs[ion_symbol]
        log.info(f"  {ion_symbol} reference: {val:.6f} eV/atom  (from DFT refs)")
        return val

    if ion_ref_ev is not None:
        log.info(f"  {ion_symbol} reference: {ion_ref_ev:.6f} eV/atom  (user override)")
        return ion_ref_ev

    approx = APPROX_ION_REFS.get(ion_symbol, -1.5)
    log.error(f"  ⚠ NO {ion_symbol} reference energy found!")
    log.error(  "    Adsorption energy will be approximate.")
    log.error(  "    Run with --compute-refs or --ion-ref-ev for accuracy.")
    log.warning(f"  Using APPROXIMATE {ion_symbol} ref = {approx:.3f} eV/atom")
    return approx
