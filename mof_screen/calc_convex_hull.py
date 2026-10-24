"""
mof_screen/calc_convex_hull.py
===============================
PARAMETER: Convex Hull  (thermodynamic phase stability)

Calculation type : Instant — reuses E_form/atom from formation.py
Standalone       : YES — needs formation.json or explicit e_form_per_atom
Cache file       : <results_dir>/convex_hull.json

Physical meaning
----------------
The distance above the convex hull (eHull, eV/atom) measures how far the
MOF is from the lowest-energy combination of competing phases.

    eHull = E_form/atom − E_convex_hull_ref/atom

    eHull ≤ 0 eV/atom  → thermodynamically stable (on or below hull)
    eHull ≤ 0.05       → metastable (accessible at room temperature)
    eHull ≤ 0.10       → marginally metastable
    eHull  > 0.10      → unstable (decomposes spontaneously)

If no competing-phase database is available, the module uses a
conservative single-compound approximation (eHull ≈ E_form/atom for
a binary or ternary), which is always an upper bound on the true hull
distance.

Usage (standalone)
------------------
    from mof_screen.calc_convex_hull import calc_convex_hull
    result = calc_convex_hull(
        atoms=atoms,
        e_form_per_atom_ev=formation_result.e_form_per_atom_ev,
        competing_phases={},   # optional: {formula: e_form_per_atom}
        cache=cache, dashboard=dashboard,
    )
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Optional

from ase import Atoms

from mof_screen.dataclasses_ import ConvexHullResult
from mof_screen.printer import print_convex_hull
from mof_screen.progress import ProgressDashboard
from mof_screen.state import StateCache

log = logging.getLogger("mof_dft.convex_hull")

_PARAM = "convex_hull"


def calc_convex_hull(
    atoms: Atoms,
    e_form_per_atom_ev: Optional[float],
    competing_phases: Optional[Dict[str, float]] = None,
    dashboard: Optional[ProgressDashboard]       = None,
    cache:     Optional[StateCache]              = None,
) -> ConvexHullResult:
    """
    Compute the convex-hull stability (eHull) for the MOF.

    Instant — no DFT call needed; reuses E_form/atom from formation run.

    Parameters
    ----------
    atoms               : ASE Atoms object for the bare MOF
    e_form_per_atom_ev  : formation energy per atom (eV/atom) from calc_formation
    competing_phases    : optional dict mapping reduced formula → E_form/atom (eV/atom)
                          for known competing phases.  If empty/None, the analysis
                          uses the single-compound approximation (eHull ≈ E_form/atom
                          for the material itself as the reference).
    dashboard           : live progress dashboard (optional)
    cache               : per-CIF state cache (optional)
    """
    log.info("")
    log.info("━" * 64)
    log.info("  CONVEX HULL  (thermodynamic phase stability — instant)")
    log.info("━" * 64)

    result = ConvexHullResult()

    # ---- Cache hit ----
    if cache and cache.is_done(_PARAM):
        log.info("  ✓ Loading convex_hull from cache.")
        data = cache.load(_PARAM) or {}
        result.e_hull_ev_per_atom  = data.get("e_hull_ev_per_atom")
        result.e_form_per_atom_ev  = data.get("e_form_per_atom_ev")
        result.hull_ref_formula    = data.get("hull_ref_formula", "")
        result.n_competing_phases  = data.get("n_competing_phases", 0)
        result.stability_label     = data.get("stability_label", "UNKNOWN")
        result.refs_used           = data.get("refs_used", False)
        if dashboard:
            dashboard.mark_cached(_PARAM)
        print_convex_hull(result)
        return result

    if dashboard:
        dashboard.mark_running(_PARAM)

    t0 = time.time()

    if e_form_per_atom_ev is None:
        log.error("  E_form/atom not available — convex hull cannot be computed.")
        log.error("  Run formation energy first (or provide it explicitly).")
        if dashboard:
            dashboard.mark_failed(_PARAM)
        print_convex_hull(result)
        return result

    result.e_form_per_atom_ev = e_form_per_atom_ev

    # ---- Hull distance calculation ----
    competing_phases = competing_phases or {}
    n_phases = len(competing_phases)
    result.n_competing_phases = n_phases

    if n_phases > 0:
        # Full convex-hull: find the lowest combination of competing phases
        # For a multi-element system the exact hull requires a linear programme;
        # here we use the best single-phase lower bound (sufficient for screening).
        hull_ref = min(competing_phases.values())
        hull_formula = min(competing_phases, key=competing_phases.get)  # type: ignore[arg-type]
        result.e_hull_ev_per_atom = e_form_per_atom_ev - hull_ref
        result.hull_ref_formula   = hull_formula
        result.refs_used          = True
        log.info(f"  Competing phases      : {n_phases}")
        log.info(f"  Hull reference        : {hull_formula}  ({hull_ref:.4f} eV/atom)")
    else:
        # Single-compound approximation: no competing phases → hull is the
        # elemental references line.  eHull ≈ E_form/atom directly gives the
        # distance from elemental stability.
        result.e_hull_ev_per_atom = e_form_per_atom_ev   # upper bound on true eHull
        result.hull_ref_formula   = "elemental references (no competing phases supplied)"
        result.refs_used          = False
        log.info("  No competing phases supplied — using single-compound approximation.")
        log.info("  eHull ≈ E_form/atom (upper bound on true hull distance)")

    # ---- Stability label ----
    eh = result.e_hull_ev_per_atom
    if eh is not None:
        if eh <= 0.0:
            result.stability_label = "STABLE (on/below hull)"
        elif eh <= 0.025:
            result.stability_label = "METASTABLE (< 25 meV/atom)"
        elif eh <= 0.05:
            result.stability_label = "METASTABLE (< 50 meV/atom)"
        elif eh <= 0.10:
            result.stability_label = "MARGINALLY METASTABLE (< 100 meV/atom)"
        else:
            result.stability_label = "UNSTABLE (> 100 meV/atom above hull)"

    log.info(f"  E_form/atom           : {e_form_per_atom_ev:.4f} eV/atom")
    log.info(f"  eHull                 : {result.e_hull_ev_per_atom:.4f} eV/atom")
    log.info(f"  Stability             : {result.stability_label}")

    elapsed = (time.time() - t0) / 60.0
    log.info(f"  Elapsed               : {elapsed * 60:.2f} s  (instant)")

    if cache:
        cache.save(_PARAM, result)
    if dashboard:
        dashboard.mark_done(_PARAM)

    print_convex_hull(result)
    return result
