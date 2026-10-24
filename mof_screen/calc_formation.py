"""
mof_screen/calc_formation.py
=============================
PARAMETER: Formation Energy  (eV and eV/atom)

Calculation type : Instant — reuses E(MOF) already in adsorption.out
Standalone       : YES — needs adsorption.json or an explicit e_mof value
Cache file       : <results_dir>/formation.json

Formula
-------
    E_form = E(MOF) − Σ_i  n_i × E(element_i per atom)

    Stable MOFs have E_form < 0.  More negative = more thermodynamically stable.

Usage (standalone)
------------------
    from mof_screen.calc_formation import calc_formation_energy
    result = calc_formation_energy(
        atoms=atoms, e_mof=e_mof,
        ref_energies={"Cu": -3.72, "C": -9.21, ...},
        cache=cache, dashboard=dashboard,
    )
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from typing import Dict, Optional

from ase import Atoms

from mof_screen.dataclasses_ import FormationResult
from mof_screen.printer import print_formation
from mof_screen.progress import ProgressDashboard
from mof_screen.state import StateCache

log = logging.getLogger("mof_dft.formation")

_PARAM = "formation"


def calc_formation_energy(
    atoms: Atoms,
    e_mof: Optional[float],
    ref_energies: Dict[str, float],
    dashboard: Optional[ProgressDashboard] = None,
    cache: Optional[StateCache]            = None,
) -> FormationResult:
    """
    Compute the MOF formation energy from elemental DFT reference energies.

    Instant — no DFT call needed; reuses E(MOF) from the adsorption run.
    Missing element references are flagged as warnings (not fatal).
    """
    log.info("")
    log.info("━" * 64)
    log.info("  FORMATION ENERGY  (instant — reuses E(MOF))")
    log.info("━" * 64)

    result = FormationResult()

    # ---- Cache hit ----
    if cache and cache.is_done(_PARAM):
        log.info("  ✓ Loading formation from cache.")
        data = cache.load(_PARAM) or {}
        result.e_form_ev          = data.get("e_form_ev")
        result.e_form_per_atom_ev = data.get("e_form_per_atom_ev")
        result.refs_complete      = data.get("refs_complete", True)
        result.missing_elements   = data.get("missing_elements", [])
        if dashboard:
            dashboard.mark_cached(_PARAM)
        print_formation(result)
        return result

    if dashboard:
        dashboard.mark_running(_PARAM)

    t0 = time.time()

    if e_mof is None:
        log.error("  E(MOF) not available — formation energy cannot be computed.")
        if dashboard:
            dashboard.mark_failed(_PARAM)
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
            log.warning(f"  Missing elemental reference: {elem}")
            continue
        e_ref   = ref_energies[elem]
        contrib = cnt * e_ref
        ref_sum += contrib
        log.info(f"  {elem:3s}: {cnt:4d} × {e_ref:.6f} = {contrib:.6f} eV")

    result.missing_elements   = missing
    result.e_form_ev          = e_mof - ref_sum
    result.e_form_per_atom_ev = result.e_form_ev / n_atoms

    log.info(f"  E(MOF)      = {e_mof:.6f} eV")
    log.info(f"  Σ refs      = {ref_sum:.6f} eV")
    log.info(f"  E_form      = {result.e_form_ev:.6f} eV")
    log.info(f"  E_form/atom = {result.e_form_per_atom_ev:.6f} eV/atom")

    elapsed = (time.time() - t0) / 60.0
    log.info(f"  Elapsed     = {elapsed * 60:.2f} s  (instant)")

    if cache:
        cache.save(_PARAM, result)
    if dashboard:
        dashboard.mark_done(_PARAM)

    print_formation(result)
    return result
