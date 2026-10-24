"""
mof_screen/workflow.py
=======================
Main DAG orchestrator for the MOF K⁺-anode DFT screening pipeline.

run_screening() executes all (or a selected subset of) parameters in
the correct dependency order, with live progress dashboard, per-parameter
JSON caching, and per-CIF log file.

Dependency order
----------------
    adsorption  →  formation  (needs E(MOF) from adsorption.out)
    adsorption  →  volume     (reads adsorption.out cell)
    adsorption  →  ocv        (needs E_ads)
    adsorption  →  ionic_conductivity  (needs E_ads; barrier optional)
    diffusion   →  (standalone, feeds ionic_conductivity optionally)
    dos         →  (standalone, reads .pdos files)
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
from ase import Atoms

from mof_screen.calc_adsorption import calc_adsorption_energy
from mof_screen.calc_bader import calc_bader_charges
from mof_screen.calc_convex_hull import calc_convex_hull
from mof_screen.calc_diffusion import calc_diffusion_barrier
from mof_screen.calc_neb import calc_neb_barrier
from mof_screen.calc_dos import calc_dos
from mof_screen.calc_formation import calc_formation_energy
from mof_screen.calc_ionic_conductivity import calc_ionic_conductivity
from mof_screen.calc_ocv import calc_ocv
from mof_screen.calc_volume import calc_volume_expansion
from mof_screen.constants import (
    DEFAULT_TEMPERATURE,
    SUPPORTED_IONS,
)
from mof_screen.dataclasses_ import AllResults
from mof_screen.parsers import extract_total_energy
from mof_screen.progress import ProgressDashboard
from mof_screen.references import (
    compute_all_references,
    load_reference_energies,
    resolve_ion_reference,
)
from mof_screen.state import StateCache
from mof_screen.structure import validate_cif

log = logging.getLogger("mof_dft.workflow")

import sys as _sys
_cp2k_data_default = str(
    Path(_sys.executable).parent.parent / "share" / "cp2k" / "data"
)
os.environ.setdefault("CP2K_DATA_DIR", _cp2k_data_default)


# ---------------------------------------------------------------------------
# All recognised parameter names
# ---------------------------------------------------------------------------
ALL_PARAMS: List[str] = [
    "adsorption",
    "formation",
    "volume",
    "ocv",
    "diffusion",
    "dos",
    "ionic_conductivity",
    "convex_hull",
    "bader",
]


# ---------------------------------------------------------------------------
# Main workflow entry point
# ---------------------------------------------------------------------------
def run_screening(
    cif_path: str,
    # Execution control
    only: Optional[List[str]]    = None,     # run only these params
    force_rerun: Optional[List[str]] = None, # clear cache and re-run these
    dry_run: bool                = False,
    # DFT settings
    num_cores: int               = 16,
    mpi_ranks: int               = 1,
    use_high_accuracy: bool      = False,
    fast_mode: bool              = False,
    screening_mode: bool         = False,
    # Ion
    ion_symbol: str              = "K",
    n_ions: int                  = 1,
    cell_opt: bool               = False,
    user_mult: Optional[int]     = None,
    temperature_k: float         = DEFAULT_TEMPERATURE,
    # References
    ref_energies_file: Optional[str] = None,
    compute_refs: bool           = False,
    ion_ref_ev: Optional[float]  = None,
    # Optional inputs
    barrier_file: Optional[str]  = None,
    neb_images: Optional[str]    = None,
    nproc_rep: int               = 2,
    compute_dos: bool            = False,
    cp2k_data_dir: Optional[str] = None,
    # Convex hull
    competing_phases: Optional[Dict[str, float]] = None,
    # Bader
    compute_bader: bool          = False,
) -> AllResults:
    """
    Run the MOF K⁺-anode DFT screening pipeline.

    Parameters
    ----------
    cif_path        : absolute or relative path to the MOF CIF file
    only            : if set, run ONLY the listed parameters
    force_rerun     : clear cache for these params and re-run them
    dry_run         : print the execution plan without running any DFT
    num_cores       : OMP_NUM_THREADS per MPI rank
    mpi_ranks       : number of MPI ranks
    use_high_accuracy : True → TZV2P basis; False → DZVP
    fast_mode       : True → lower cutoffs for quick screening
    ion_symbol      : which ion to insert (default "K")
    n_ions          : number of ions to insert per unit cell
    cell_opt        : relax cell vectors during adsorption GEO_OPT
    user_mult       : spin multiplicity override (None = auto-detect)
    temperature_k   : temperature for Nernst-Einstein σ (default 300 K)
    ref_energies_file : path to a pre-computed elemental_refs JSON
    compute_refs    : run CP2K to compute self-consistent element refs
    ion_ref_ev      : override bulk K reference energy (eV/atom)
    barrier_file    : path to pre-computed NEB barrier file
    compute_dos     : parse PDOS files after adsorption run
    cp2k_data_dir   : override CP2K_DATA_DIR env variable
    competing_phases: dict {formula: E_form/atom (eV/atom)} for convex hull
    compute_bader   : True → parse Bader/Hirshfeld charges from output
    """
    # ---- Validate ion ----
    if ion_symbol not in SUPPORTED_IONS:
        raise ValueError(
            f"ion_symbol must be one of {SUPPORTED_IONS}, got '{ion_symbol}'"
        )

    if cp2k_data_dir:
        os.environ["CP2K_DATA_DIR"] = cp2k_data_dir

    cif_path = os.path.abspath(cif_path)
    cif_stem = Path(cif_path).stem
    out_dir  = Path(os.path.dirname(cif_path)) / "results" / cif_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- File logger (per-CIF) ----
    log_path = out_dir / "run.log"
    if not any(isinstance(h, logging.FileHandler) for h in log.handlers):
        fh = logging.FileHandler(str(log_path))
        fh.setFormatter(
            logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s",
                              datefmt="%H:%M:%S")
        )
        log.addHandler(fh)

    log.info("═" * 65)
    log.info(f"  MOF K⁺-ANODE SCREENING — {ion_symbol.upper()} — {cif_stem}")
    log.info("═" * 65)
    log.info(f"  CIF         : {cif_path}")
    log.info(f"  Output dir  : {out_dir}")
    log.info(f"  Cores/MPI   : {num_cores} OMP × {mpi_ranks} MPI")
    log.info(f"  Basis       : {'TZV2P' if use_high_accuracy else 'DZVP'}"
             f"{'  [SCREENING]' if screening_mode else '  [FAST]' if fast_mode else ''}")
    log.info(f"  Ion         : {ion_symbol}  |  n={n_ions}  |  cell_opt={cell_opt}")
    log.info(f"  Temperature : {temperature_k} K")

    # ---- Load and validate CIF ----
    atoms   = validate_cif(cif_path)
    formula = atoms.get_chemical_formula()
    n_atoms = len(atoms)
    log.info(f"  Formula     : {formula}  ({n_atoms} atoms)")
    log.info(f"  Cell (Å)    : {np.round(atoms.cell.lengths(), 3)}")
    log.info(f"  Volume      : {atoms.get_volume():.3f} Å³")
    log.info(f"  Elements    : {', '.join(sorted(set(atoms.get_chemical_symbols())))}")

    # ---- Determine which params to run ----
    active_params: Set[str] = set(only) if only else set(ALL_PARAMS)
    if not compute_dos:
        active_params.discard("dos")
    if not barrier_file and not neb_images:
        active_params.discard("diffusion")
    if not compute_bader:
        active_params.discard("bader")

    # ---- Dry-run: print plan and exit ----
    if dry_run:
        _print_dry_run(active_params, cif_path, ion_symbol, n_ions, barrier_file)
        return AllResults(formula=formula, n_atoms=n_atoms,
                          cif_path=cif_path, ion_symbol=ion_symbol)

    # ---- State cache ----
    cache = StateCache(out_dir)

    # Clear forced re-runs
    if force_rerun:
        for p in force_rerun:
            cache.clear(p)
            log.info(f"  Cache cleared for: {p}  (--force-rerun)")

    # ---- Reference energies ----
    refs: Dict[str, float] = {}
    if compute_refs:
        elements = set(atoms.get_chemical_symbols()) | {ion_symbol}
        log.info(f"  Computing DFT refs for: {sorted(elements)}")
        refs = compute_all_references(
            elements, out_dir, num_cores, mpi_ranks, use_high_accuracy, fast_mode
        )
    else:
        rf   = (Path(ref_energies_file) if ref_energies_file
                else out_dir / "elemental_refs" / "ref_energies.json")
        refs = load_reference_energies(rf)
        if refs:
            log.info(f"  Loaded {len(refs)} ref energies from {rf}")

    ion_ref = resolve_ion_reference(ion_symbol, refs, ion_ref_ev)

    # ---- Build results container ----
    results = AllResults(
        formula=formula, n_atoms=n_atoms,
        cif_path=cif_path, ion_symbol=ion_symbol,
    )

    # ---- Initialise progress dashboard ----
    dashboard = ProgressDashboard(
        params     = list(active_params),
        mof_name   = formula,
        ion_symbol = ion_symbol,
    )

    # Mark skipped params
    for p in set(ALL_PARAMS) - active_params:
        dashboard.mark_skipped(p)

    dashboard.start()
    t_total = time.time()

    try:
        # ================================================================
        # STEP 1 — ADSORPTION ENERGY  (most expensive DFT calc)
        # ================================================================
        if "adsorption" in active_params:
            e_mof = _get_e_mof(
                atoms, out_dir, cache, num_cores, mpi_ranks,
                use_high_accuracy, fast_mode, screening_mode,
                cell_opt=cell_opt,
            )
            results.adsorption = calc_adsorption_energy(
                atoms=atoms, e_mof=e_mof,
                out_dir=out_dir,
                num_cores=num_cores, mpi_ranks=mpi_ranks,
                use_high_accuracy=use_high_accuracy,
                fast_mode=fast_mode,
                screening_mode=screening_mode,
                n_ions=n_ions, ion_symbol=ion_symbol,
                cell_opt=cell_opt,
                ion_ref_ev=ion_ref,
                user_mult=user_mult,
                dashboard=dashboard, cache=cache,
            )
        else:
            # Try to load from cache for downstream calcs
            _load_adsorption_from_cache(results, cache)

        e_ads  = results.adsorption.e_ads_ev
        e_mof  = _get_e_mof(
            atoms, out_dir, cache, num_cores, mpi_ranks,
            use_high_accuracy, fast_mode, screening_mode,
            cell_opt=cell_opt,
        )

        # ================================================================
        # STEP 2 — FORMATION ENERGY  (instant)
        # ================================================================
        if "formation" in active_params:
            results.formation = calc_formation_energy(
                atoms=atoms, e_mof=e_mof,
                ref_energies=refs,
                dashboard=dashboard, cache=cache,
            )

        # ================================================================
        # STEP 3 — VOLUME EXPANSION  (instant)
        # ================================================================
        if "volume" in active_params:
            results.volume = calc_volume_expansion(
                atoms=atoms, out_dir=out_dir,
                cell_opt=cell_opt, ion_symbol=ion_symbol,
                dashboard=dashboard, cache=cache,
            )

        # ================================================================
        # STEP 4 — OPEN-CIRCUIT VOLTAGE  (instant)
        # ================================================================
        if "ocv" in active_params:
            results.ocv = calc_ocv(
                e_ads_ev=e_ads, n_ions=n_ions,
                ion_symbol=ion_symbol,
                dashboard=dashboard, cache=cache,
            )

        # ================================================================
        # STEP 5 — DIFFUSION BARRIER  (NEB file reader OR CP2K CI-NEB)
        # ================================================================
        if "diffusion" in active_params or barrier_file or neb_images:
            if neb_images:
                # Full DFT CI-NEB run from image XYZ files
                results.diffusion_barrier = calc_neb_barrier(
                    image_dir         = neb_images,
                    out_dir           = out_dir,
                    atoms             = atoms,
                    num_cores         = num_cores,
                    mpi_ranks         = mpi_ranks,
                    use_high_accuracy = use_high_accuracy,
                    user_mult         = user_mult,
                    nproc_per_replica = nproc_rep,
                    dashboard         = dashboard,
                    cache             = cache,
                )
            else:
                # Fast path: read pre-computed barrier from plain text / CP2K output
                results.diffusion_barrier = calc_diffusion_barrier(
                    barrier_file=barrier_file,
                    dashboard=dashboard, cache=cache,
                )

        # ================================================================
        # STEP 6 — IONIC CONDUCTIVITY  (Nernst-Einstein)
        # ================================================================
        if "ionic_conductivity" in active_params:
            results.ionic_conductivity = calc_ionic_conductivity(
                e_ads_ev       = e_ads,
                n_ions         = n_ions,
                cell_volume_A3 = atoms.get_volume(),
                barrier        = results.diffusion_barrier,
                temperature_k  = temperature_k,
                ion_symbol     = ion_symbol,
                dashboard      = dashboard,
                cache          = cache,
            )

        # ================================================================
        # STEP 7 — DENSITY OF STATES  (parse .pdos files)
        # ================================================================
        if "dos" in active_params:
            results.dos = calc_dos(
                out_dir=out_dir,
                dashboard=dashboard, cache=cache,
            )

        # ================================================================
        # STEP 8 — CONVEX HULL  (thermodynamic phase stability)
        # ================================================================
        if "convex_hull" in active_params:
            results.convex_hull = calc_convex_hull(
                atoms               = atoms,
                e_form_per_atom_ev  = results.formation.e_form_per_atom_ev,
                competing_phases    = competing_phases or {},
                dashboard           = dashboard,
                cache               = cache,
            )

        # ================================================================
        # STEP 9 — BADER CHARGE ANALYSIS
        # ================================================================
        if "bader" in active_params:
            results.bader = calc_bader_charges(
                out_dir    = out_dir,
                ion_symbol = ion_symbol,
                n_ions     = n_ions,
                dashboard  = dashboard,
                cache      = cache,
            )

    finally:
        dashboard.stop()

    # ---- Save full JSON summary ----
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(asdict(results), indent=2, default=str))
    log.info(f"\n  All results saved to: {summary_path}")

    # ---- Final compact table ----
    _print_final_summary(results, formula, n_atoms, ion_symbol, n_ions,
                         out_dir, time.time() - t_total)

    # ---- Dashboard summary table ----
    dashboard.print_summary_table()

    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_e_mof(
    atoms: Atoms, out_dir: Path, cache: StateCache,
    num_cores: int, mpi_ranks: int,
    use_high_accuracy: bool, fast_mode: bool, screening_mode: bool,
    cell_opt: bool = False,
) -> Optional[float]:
    """
    Obtain the bare MOF energy (eV).

    CRITICAL: The run_type used here MUST match the level used for E(MOF+K).
    - cell_opt=False  → single-point ENERGY  (fast, for rigid MOFs like UiO-66)
    - cell_opt=True   → CELL_OPT relaxation  (required for flexible MOFs like ZIF-8)

    Without this consistency, the framework relaxation energy (can be 50+ eV!)
    is incorrectly absorbed into E_ads, giving completely wrong results.
    """
    data = cache.load("mof_energy") or {}
    if "e_mof" in data:
        # Check if cached result was computed at the correct level
        cached_cell_opt = data.get("cell_opt", False)
        if cached_cell_opt == cell_opt:
            return data["e_mof"]
        else:
            log.warning(
                f"  Cached E(MOF) was computed with cell_opt={cached_cell_opt} "
                f"but current run needs cell_opt={cell_opt}. Re-computing..."
            )

    from mof_screen.cp2k_input import generate_cp2k_input
    from mof_screen.cp2k_runner import run_cp2k
    from mof_screen.structure import detect_transition_metals, estimate_multiplicity
    from mof_screen.parsers import extract_total_energy

    inp_file = str(out_dir / "mof.inp")
    out_file = str(out_dir / "mof.out")

    run_type = "CELL_OPT" if cell_opt else "ENERGY"
    label    = "Bare-MOF-CellOpt" if cell_opt else "Bare-MOF"

    # If it's already run and completed at the right level
    if not cell_opt:
        e_mof = extract_total_energy(out_file)
        if e_mof is not None:
            cache.save("mof_energy", {"e_mof": e_mof, "cell_opt": cell_opt})
            log.info(f"  ✓ Loaded bare MOF energy: {e_mof:.6f} eV")
            return e_mof
    else:
        # For CELL_OPT, check if the cell-optimized output already exists
        cell_out_file = str(out_dir / "mof_cellopt.out")
        inp_file      = str(out_dir / "mof_cellopt.inp")
        out_file      = cell_out_file
        e_mof = extract_total_energy(cell_out_file)
        if e_mof is not None:
            cache.save("mof_energy", {"e_mof": e_mof, "cell_opt": cell_opt})
            log.info(f"  ✓ Loaded bare MOF cell-opt energy: {e_mof:.6f} eV")
            return e_mof

    log.info(f"  Computing bare MOF energy (E_MOF) — run_type={run_type}...")
    if cell_opt:
        log.info("  NOTE: cell_opt=True → relaxing bare MOF cell to match E(MOF+K) level.")
    metals = detect_transition_metals(atoms)
    mult = estimate_multiplicity(atoms)

    inp = generate_cp2k_input(
        atoms, metals, mult, "bare_mof",
        run_type=run_type,
        cell_opt=cell_opt,
        use_high_accuracy=use_high_accuracy,
        fast_mode=fast_mode,
        screening_mode=screening_mode,
    )
    Path(inp_file).write_text(inp)

    ok = run_cp2k(inp_file, out_file, num_cores, mpi_ranks, label=label)
    if not ok:
        log.error(f"  Bare MOF {run_type} calculation failed.")
        return None

    e_mof = extract_total_energy(out_file)
    if e_mof is not None:
        cache.save("mof_energy", {"e_mof": e_mof, "cell_opt": cell_opt})
        log.info(f"  ✓ Computed bare MOF energy: {e_mof:.6f} eV")
    else:
        log.error("  Failed to extract bare MOF energy from output.")
    return e_mof


def _load_adsorption_from_cache(
    results: AllResults, cache: StateCache
) -> None:
    """Load AdsorptionResult from cache into results if available."""
    if not cache.is_done("adsorption"):
        return
    data = cache.load("adsorption") or {}
    results.adsorption.e_ads_ev     = data.get("e_ads_ev")
    results.adsorption.e_mof_ion_ev = data.get("e_mof_ion_ev")
    results.adsorption.relaxed      = data.get("relaxed", False)
    results.adsorption.elapsed_min  = data.get("elapsed_min", 0.0)


def _print_dry_run(
    active_params: Set[str],
    cif_path: str,
    ion_symbol: str,
    n_ions: int,
    barrier_file: Optional[str],
) -> None:
    print("\n" + "═" * 65)
    print("  DRY RUN — execution plan  (no DFT will run)")
    print("═" * 65)
    print(f"  CIF        : {cif_path}")
    print(f"  Ion        : {ion_symbol} × {n_ions}")
    if barrier_file:
        print(f"  Barrier    : {barrier_file}")
    print(f"\n  Parameters to run ({len(active_params)}):")
    for p in ALL_PARAMS:
        status = "✓ WILL RUN" if p in active_params else "─ SKIP"
        print(f"    {status:<12s} {p}")
    print("═" * 65 + "\n")


def _print_final_summary(
    results: AllResults,
    formula: str,
    n_atoms: int,
    ion_symbol: str,
    n_ions: int,
    out_dir: Path,
    elapsed_s: float,
) -> None:
    print("\n" + "═" * 65)
    print("  FINAL SUMMARY")
    print("═" * 65)
    print(f"  MOF: {formula}  ({n_atoms} atoms)  |  Ion: {ion_symbol}×{n_ions}")
    print(f"  Total elapsed: {elapsed_s/60:.1f} min")
    print(f"  {'─'*61}")

    # Ionic conductivity
    ic = results.ionic_conductivity
    if ic.sigma_s_cm is not None:
        print(f"  Ionic σ        : {ic.sigma_s_cm:.3e} S/cm  [{ic.rank}]")
    elif ic.rank not in ("UNKNOWN", ""):
        print(f"  Ionic σ (rank) : {ic.rank}  (no barrier → estimated)")
    else:
        print("  Ionic σ        : NOT COMPUTED")

    # Adsorption
    ad = results.adsorption
    if ad.e_ads_ev is not None:
        print(f"  E_ads ({ion_symbol}×{n_ions})   : {ad.e_ads_ev:.4f} eV")
    else:
        print("  E_ads          : NOT COMPUTED")

    # Formation
    fm = results.formation
    if fm.e_form_per_atom_ev is not None:
        flag = "  ⚠ approx" if not fm.refs_complete else ""
        print(f"  E_form/atom    : {fm.e_form_per_atom_ev:.4f} eV/atom{flag}")
    else:
        print("  E_form/atom    : NOT COMPUTED")

    # Volume
    vl = results.volume
    if vl.expansion_pct is not None:
        print(f"  Volume exp.    : {vl.expansion_pct:.3f} %")
    else:
        print("  Volume exp.    : NOT COMPUTED")

    # OCV
    ov = results.ocv
    if ov.ocv_v is not None:
        print(f"  OCV            : {ov.ocv_v:.4f} V  (vs {ion_symbol}⁺/{ion_symbol})")
    else:
        print("  OCV            : NOT COMPUTED")

    # Diffusion
    db = results.diffusion_barrier
    if db.available and db.barrier_ev is not None:
        print(f"  Diff. barrier  : {db.barrier_ev:.4f} eV")
    else:
        print("  Diff. barrier  : NOT PROVIDED (use --barrier-file)")

    # DOS
    ds = results.dos
    if ds.parsed:
        fermi_str = f"{ds.fermi_ev:.4f} eV" if ds.fermi_ev is not None else "N/A"
        print(f"  DOS (Fermi)    : {fermi_str}  [{ds.n_pdos_files} PDOS files]")
    else:
        print("  DOS            : NOT PARSED (use --dos)")

    # Convex hull
    ch = results.convex_hull
    if ch.e_hull_ev_per_atom is not None:
        approx = "  ⚠ approx" if not ch.refs_used else ""
        print(f"  eHull          : {ch.e_hull_ev_per_atom:.4f} eV/atom  [{ch.stability_label}]{approx}")
    else:
        print("  eHull          : NOT COMPUTED (needs formation energy)")

    # Bader
    bd = results.bader
    if bd.parsed:
        print(
            f"  Bader Δq ({ion_symbol})  : {bd.ion_charge_transfer:.4f} e  "
            f"[{bd.method}]"
        )
    else:
        print("  Bader Δq       : NOT PARSED (use --bader)")

    print(f"  {'─'*61}")
    print(f"  Output dir     : {out_dir}")
    print("═" * 65 + "\n")

