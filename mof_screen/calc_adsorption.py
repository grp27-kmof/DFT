"""
mof_screen/calc_adsorption.py
==============================
PARAMETER: Adsorption Energy  (K⁺ ion binding energy, eV)

Calculation type : GEO_OPT of MOF + K⁺ (most expensive DFT step)
Standalone       : YES — only needs the CIF and K reference energy
Cache file       : <results_dir>/adsorption.json

Formula
-------
    E_ads = E(MOF+K) − E(MOF) − n_ions × E(K_bulk per atom)

    Negative E_ads = K⁺ binds favourably to the MOF → good anode.

Usage (standalone)
------------------
    from mof_screen.calc_adsorption import calc_adsorption_energy
    result = calc_adsorption_energy(
        atoms=atoms, e_mof=e_mof,
        out_dir=out_dir, num_cores=16, mpi_ranks=1,
        n_ions=1, ion_symbol="K", cell_opt=False,
        ion_ref_ev=-1.0,
        dashboard=None, cache=cache,
    )
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

from ase import Atoms
from ase.io import write

from mof_screen.cp2k_input import generate_cp2k_input
from mof_screen.cp2k_runner import run_cp2k
from mof_screen.dataclasses_ import AdsorptionResult
from mof_screen.parsers import check_geo_opt_completed, extract_total_energy
from mof_screen.printer import print_adsorption
from mof_screen.progress import ProgressDashboard
from mof_screen.state import StateCache
from mof_screen.structure import (
    build_loaded_structure,
    detect_transition_metals,
    estimate_multiplicity,
    find_candidate_sites_all_types,
)

log = logging.getLogger("mof_dft.adsorption")

_PARAM = "adsorption"


def calc_adsorption_energy(
    atoms: Atoms,
    e_mof: Optional[float],
    out_dir: Path,
    num_cores: int,
    mpi_ranks: int,
    use_high_accuracy: bool,
    fast_mode: bool,
    screening_mode: bool,
    n_ions: int,
    ion_symbol: str,
    cell_opt: bool,
    ion_ref_ev: float,
    user_mult: Optional[int]              = None,
    dashboard: Optional[ProgressDashboard] = None,
    cache: Optional[StateCache]            = None,
) -> AdsorptionResult:
    """
    Compute K⁺ adsorption energy by geometry-optimising the ion-loaded MOF.

    If a valid cache exists the DFT run is skipped entirely.
    Progress is streamed to the dashboard (SCF / GEO step counters).
    """
    log.info("")
    log.info("━" * 64)
    log.info(f"  ADSORPTION ENERGY  ({n_ions}× {ion_symbol})  —  GEO_OPT (DFT)")
    log.info("━" * 64)

    result = AdsorptionResult(n_ions=n_ions, ion_symbol=ion_symbol)

    # ---- Cache hit ----
    if cache and cache.is_done(_PARAM):
        log.info("  ✓ Loading adsorption from cache.")
        data = cache.load(_PARAM) or {}
        result.e_ads_ev     = data.get("e_ads_ev")
        result.e_mof_ion_ev = data.get("e_mof_ion_ev")
        result.relaxed      = data.get("relaxed", False)
        result.elapsed_min  = data.get("elapsed_min", 0.0)
        if dashboard:
            dashboard.mark_cached(_PARAM)
        print_adsorption(result)
        return result

    if dashboard:
        dashboard.mark_running(_PARAM)

    if e_mof is None:
        log.error("  E(MOF) not available — skipping adsorption.")
        if dashboard:
            dashboard.mark_failed(_PARAM)
        print_adsorption(result)
        return result

    # ---- Multi-site pre-screening: find the best K-binding site ----
    run_type = "CELL_OPT" if cell_opt else "GEO_OPT"
    label    = f"mof_{ion_symbol.lower()}"
    inp_file = str(out_dir / "adsorption.inp")
    out_file = str(out_dir / "adsorption.out")

    t0 = time.time()

    # ---- Return early if a fully converged output already exists ----
    geo_done = check_geo_opt_completed(out_file)
    if geo_done:
        log.info("  adsorption.out: GEO_OPT completed — skipping DFT run.")
        result.relaxed = True
        result.elapsed_min  = (time.time() - t0) / 60.0
        e_mof_ion           = extract_total_energy(out_file)
        result.e_mof_ion_ev = e_mof_ion
        if e_mof_ion is None:
            log.error(f"  Could not extract E(MOF+{ion_symbol}).")
            if dashboard: dashboard.mark_failed(_PARAM)
            print_adsorption(result)
            return result
        result.e_ads_ev = e_mof_ion - e_mof - n_ions * ion_ref_ev
        log.info(f"  E(MOF)              = {e_mof:.6f} eV")
        log.info(f"  E(MOF+{n_ions}{ion_symbol})      = {e_mof_ion:.6f} eV")
        log.info(f"  E({ion_symbol} ref/atom)  = {ion_ref_ev:.6f} eV")
        log.info(f"  E_ads               = {result.e_ads_ev:.6f} eV")
        if cache: cache.save(_PARAM, result)
        if dashboard: dashboard.mark_done(_PARAM)
        print_adsorption(result)
        return result

    # ---- Generate chemistry-aware candidate sites ----
    log.info(f"  Generating chemistry-aware binding site candidates for {ion_symbol}…")
    try:
        candidates = find_candidate_sites_all_types(atoms, ion=ion_symbol, max_per_type=1)
    except Exception as exc:
        log.warning(f"  Site detection failed ({exc}) — falling back to single void site.")
        try:
            loaded = build_loaded_structure(atoms, ion_symbol, n_ions)
            candidates = [(loaded.get_positions()[-n_ions:][0], "void_geometric", 99.0)]
        except RuntimeError as exc2:
            log.error(f"  {exc2}")
            if dashboard: dashboard.mark_failed(_PARAM)
            print_adsorption(result)
            return result

    # ---- Pre-screen each candidate site with a cheap short GEO_OPT ----
    best_site_pos     = None
    best_site_type    = "unknown"
    best_site_e       = None
    best_prescreen_wfn: Optional[str] = None   # WFN from winning pre-screen

    if len(candidates) > 1:
        log.info(f"  Pre-screening {len(candidates)} candidate site(s)…")
        for site_idx, (site_pos, site_type, _score) in enumerate(candidates):
            ps_label    = f"prescreen_{site_idx+1}_{site_type}"
            ps_inp_file = str(out_dir / f"{ps_label}.inp")
            ps_out_file = str(out_dir / f"{ps_label}.out")
            # WFN files are written to the process CWD (not the results dir)
            ps_wfn_cwd  = str(Path(".").resolve() / f"{ps_label}-RESTART.wfn")
            ps_wfn_res  = str(out_dir / f"{ps_label}-RESTART.wfn")
            ps_wfn      = ps_wfn_cwd if os.path.exists(ps_wfn_cwd) else (
                          ps_wfn_res  if os.path.exists(ps_wfn_res)  else None)

            # Skip if this pre-screen already ran (cache reuse on re-run)
            ps_cached_e = extract_total_energy(ps_out_file)
            if ps_cached_e is not None:
                log.info(f"    [{site_idx+1}/{len(candidates)}] [{site_type}] E = {ps_cached_e:.4f} eV  (cached)")
                if ps_cached_e < (best_site_e or float("inf")):
                    best_site_e       = ps_cached_e
                    best_site_pos     = site_pos
                    best_site_type    = site_type
                    best_prescreen_wfn = ps_wfn
                continue

            # Build trial structure with K at this candidate site
            trial = atoms.copy()
            from ase import Atoms as _Atoms
            trial += _Atoms(ion_symbol, positions=[site_pos], cell=atoms.get_cell(), pbc=True)

            metals_trial = detect_transition_metals(trial)
            mult_trial   = user_mult if user_mult else estimate_multiplicity(trial)

            # Cheap pre-screen input (30-step loose GEO_OPT)
            ps_inp = generate_cp2k_input(
                trial, metals_trial, mult_trial, ps_label,
                run_type="GEO_OPT",
                use_high_accuracy=False,
                cell_opt=False,
                use_smearing=True,
                fast_mode=False,
                screening_mode=True,
            )
            Path(ps_inp_file).write_text(ps_inp)
            log.info(f"    [{site_idx+1}/{len(candidates)}] Pre-screening [{site_type}]…")

            run_cp2k(
                ps_inp_file, ps_out_file,
                num_cores, mpi_ranks,
                label=f"PreSCR-{site_type}",
                param_name=_PARAM,
                dashboard=dashboard,
            )
            # Refresh WFN path after run
            ps_wfn = ps_wfn_cwd if os.path.exists(ps_wfn_cwd) else (
                     ps_wfn_res  if os.path.exists(ps_wfn_res)  else None)

            ps_e = extract_total_energy(ps_out_file)
            log.info(
                f"    [{site_idx+1}/{len(candidates)}] [{site_type}] E = "
                f"{ps_e:.4f} eV" if ps_e else
                f"    [{site_idx+1}/{len(candidates)}] [{site_type}] — energy extraction failed"
            )
            if ps_e is not None and (best_site_e is None or ps_e < best_site_e):
                best_site_e       = ps_e
                best_site_pos     = site_pos
                best_site_type    = site_type
                best_prescreen_wfn = ps_wfn
        log.info(f"  Best pre-screen site: [{best_site_type}]  E = {best_site_e:.4f} eV")
        if best_prescreen_wfn:
            log.info(f"  Prescreen WFN for warm-start: {best_prescreen_wfn}")
        else:
            log.warning("  No prescreen WFN found — production will start from ATOMIC guess.")
    else:
        # Only one candidate — skip pre-screening
        best_site_pos  = candidates[0][0]
        best_site_type = candidates[0][1]
        log.info(f"  Single candidate site [{best_site_type}] — skipping pre-screen.")

    # ---- Build production structure with properly distributed ions ----
    if best_site_pos is None:
        best_site_pos  = candidates[0][0]
        best_site_type = candidates[0][1]
        log.warning("  Pre-screen did not converge for any site — using top-ranked candidate.")

    from ase import Atoms as _Atoms2
    from mof_screen.structure import find_binding_sites, find_void_sites

    if n_ions == 1:
        # Single ion: place at the pre-screened best site
        loaded = atoms.copy()
        loaded += _Atoms2(ion_symbol, positions=[best_site_pos],
                          cell=atoms.get_cell(), pbc=True)
        log.info(f"  Production: 1× {ion_symbol} at [{best_site_type}] site.")
    else:
        # Multiple ions: distribute across n_ions distinct sites.
        # Using find_binding_sites() which enforces minimum K–K separation
        # (never stacks ions at the same position → avoids 0-Å distances and abort).
        # This matches the published approach for high-loading MOF studies
        # (e.g. ACS JPCC 2021, doi:10.1021/acs.jpcc.1c01657).
        log.info(
            f"  Distributing {n_ions}× {ion_symbol} across distinct sites "
            f"(chemistry-aware + void-fill)…"
        )
        try:
            sites = find_binding_sites(atoms, ion=ion_symbol, n_sites=n_ions)
        except Exception as exc:
            log.warning(f"  Chemistry site search failed ({exc}); using grid void search.")
            sites = [(s, "void_geometric", 99.0)
                     for s in find_void_sites(atoms, ion=ion_symbol,
                                              n_sites=n_ions, grid_pts=20)]

        n_placed = len(sites)
        if n_placed < n_ions:
            log.warning(
                f"  Only {n_placed}/{n_ions} distinct void sites found. "
                f"Running with {n_placed}× {ion_symbol} instead. "
                f"For very high loading, increase grid_pts or use a supercell."
            )

        loaded = atoms.copy()
        site_labels = []
        for site_pos, site_type, _ in sites:
            loaded += _Atoms2(ion_symbol, positions=[site_pos],
                              cell=atoms.get_cell(), pbc=True)
            site_labels.append(site_type)
        log.info(
            f"  Placed {n_placed}× {ion_symbol} at sites: "
            + ", ".join(f"[{t}]" for t in site_labels)
        )

    log.info(f"  Production GEO_OPT will use [{best_site_type}] as primary site.")


    # Physical sanity check: warn if n_ions exceeds realistic pore capacity.
    # K⁺ at close-packing: ~17 Å³/ion.  MOF pore fraction ≈ 0.40–0.55.
    # Conservative estimate: pore_vol = 0.40 × cell_vol → max = pore_vol / 17
    _pore_vol_est = 0.40 * atoms.get_volume()
    _max_safe     = max(1, int(_pore_vol_est / 17.0))
    if n_ions > _max_safe:
        log.warning(
            f"  ⚠ n_ions={n_ions} exceeds estimated pore capacity "
            f"(cell={atoms.get_volume():.0f} Å³, pore≈{_pore_vol_est:.0f} Å³ "
            f"→ max ~{_max_safe} K at close-packing). "
            f"GEO_OPT may struggle to converge at this loading."
        )
    else:
        log.info(
            f"  Pore capacity check: {n_ions}/{_max_safe} K ions "
            f"(cell={atoms.get_volume():.0f} Å³) — within estimated limit."
        )

    write(str(out_dir / f"mof_with_{ion_symbol.lower()}.cif"), loaded)

    metals   = detect_transition_metals(loaded)
    mult     = user_mult if user_mult else estimate_multiplicity(loaded)

    # ── WFN restart logic ────────────────────────────────────────────────────
    # CRITICAL: The prescreen WFN is computed with exactly 1 ion.
    # Its MO matrix has N_electrons = N_MOF + Z_ion (e.g. 19 for K).
    # The production WFN needs N_electrons = N_MOF + n_ions × Z_ion.
    # If n_ions > 1, dimensions mismatch → CP2K SIGABRT (exit -6, 0.0 min).
    # → ONLY use prescreen WFN when n_ions == 1.
    prod_wfn_cwd = str(Path(".").resolve() / f"{label}-RESTART.wfn")
    prod_wfn_res = str(out_dir / f"{label}-RESTART.wfn")
    prod_wfn     = prod_wfn_cwd if os.path.exists(prod_wfn_cwd) else (
                   prod_wfn_res  if os.path.exists(prod_wfn_res)  else None)

    has_partial_production = extract_total_energy(out_file) is not None
    if has_partial_production and prod_wfn:
        # Continue an interrupted production run (same n_ions → same dimensions)
        restart_wfn_path = prod_wfn
        log.info(f"  Continuing partial production run. WFN: {restart_wfn_path}")
    elif n_ions == 1 and best_prescreen_wfn and os.path.exists(best_prescreen_wfn):
        # n_ions==1: prescreen WFN has same electron count → safe warm-start
        restart_wfn_path = best_prescreen_wfn
        log.info(f"  Using prescreen WFN warm-start: {restart_wfn_path}")
    else:
        # n_ions > 1: prescreen WFN has wrong dimensions → MUST start fresh
        restart_wfn_path = None
        if n_ions > 1 and best_prescreen_wfn:
            log.info(
                f"  n_ions={n_ions} > 1: skipping prescreen WFN (dimensions differ from "
                f"1-K prescreen). Starting from ATOMIC guess."
            )
        else:
            log.info("  No compatible WFN restart — starting from ATOMIC guess.")


    # Coordinate restart: ONLY use if we have a partial production output (not
    # from a different run or failed SCF). Stale coord restarts cause exit -6.
    coord_rs_cwd = str(Path(".").resolve() / f"{label}-1.restart")
    coord_rs_res = str(out_dir / f"{label}-1.restart")
    raw_coord_rs = coord_rs_cwd if os.path.exists(coord_rs_cwd) else (
                   coord_rs_res  if os.path.exists(coord_rs_res)  else None)
    coord_rs = raw_coord_rs if (has_partial_production and raw_coord_rs) else None

    if Path(out_file).exists():
        log.info("  adsorption.out exists but GEO_OPT did not complete — re-running.")

    inp = generate_cp2k_input(
        loaded, metals, mult, label,
        run_type=run_type,
        use_high_accuracy=use_high_accuracy,
        cell_opt=cell_opt,
        use_smearing=True,
        fast_mode=fast_mode,
        screening_mode=screening_mode,
    )
    Path(inp_file).write_text(inp)

    ok = run_cp2k(
        inp_file, out_file,
        num_cores, mpi_ranks,
        restart_wfn   = restart_wfn_path if restart_wfn_path and os.path.exists(restart_wfn_path) else None,
        restart_coord = coord_rs if coord_rs and os.path.exists(coord_rs) else None,
        label         = f"GEO_OPT({n_ions}{ion_symbol})",
        param_name    = _PARAM,
        dashboard     = dashboard,
    )

    if not ok:
        log.warning("  GEO_OPT failed — falling back to single-point energy.")
        inp = generate_cp2k_input(
            loaded, metals, mult, label,
            run_type="ENERGY",
            use_high_accuracy=use_high_accuracy,
            fast_mode=fast_mode,
            screening_mode=False,
        )
        Path(inp_file).write_text(inp)
        run_cp2k(
            inp_file, out_file, num_cores, mpi_ranks,
            label="SP fallback", param_name=_PARAM, dashboard=dashboard,
        )
        result.relaxed = False
    else:
        result.relaxed = True

    result.elapsed_min  = (time.time() - t0) / 60.0
    e_mof_ion           = extract_total_energy(out_file)
    result.e_mof_ion_ev = e_mof_ion

    if e_mof_ion is None:
        log.error(f"  Could not extract E(MOF+{ion_symbol}).")
        if dashboard:
            dashboard.mark_failed(_PARAM)
        print_adsorption(result)
        return result

    e_ads = e_mof_ion - e_mof - n_ions * ion_ref_ev
    
    result.e_ads_ev = e_ads
    
    log.info(f"  E(MOF)              = {e_mof:.6f} eV")
    log.info(f"  E(MOF+{n_ions}{ion_symbol})      = {e_mof_ion:.6f} eV")
    log.info(f"  E({ion_symbol} ref/atom)  = {ion_ref_ev:.6f} eV")
    log.info(f"  E_ads               = {result.e_ads_ev:.6f} eV")
    log.info(f"  Adsorption site     = [{best_site_type}]")

    if cache:
        cache.save(_PARAM, result)
    if dashboard:
        dashboard.mark_done(_PARAM)

    print_adsorption(result)
    return result
