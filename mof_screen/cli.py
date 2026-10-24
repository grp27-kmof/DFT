"""
mof_screen/cli.py
==================
Command-line interface for the MOF K⁺-anode DFT screening pipeline.

Entry points
------------
    python -m mof_screen                        # uses __main__.py
    python cli.py mof.cif --ion K --cores 16

Key flags
---------
    --only <param> [param ...]   run only specific parameters (standalone)
    --force-rerun <param> [...]  clear cache and re-run specific params
    --dry-run                    print plan, do not run DFT
    --rank                       batch mode: score and rank multiple CIFs
    --temperature <K>            temperature for Nernst-Einstein (default 300)
    --barrier-file <path>        pre-computed NEB barrier (eV) for diffusion/conductivity
    --compute-refs               run CP2K to compute elemental references
    --dos                        parse PDOS files for density of states
    --cell-opt                   relax cell during adsorption GEO_OPT
    --high-accuracy              use TZV2P basis (slower, publication quality)
    --fast                       use 400 Ry cutoff for quick screening
    --bader                      parse Bader/Hirshfeld/Mulliken charges from output
    --competing-phases <path>    JSON file {formula: E_form/atom} for convex hull
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt= "%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("mof_dft")

from mof_screen.constants import SUPPORTED_IONS
from mof_screen.workflow import ALL_PARAMS, run_screening


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="mof_screen",
        description=(
            "Production DFT screening of MOF structures as K⁺ anode materials.\n"
            "Input: one or more CIF files.  Output: per-CIF results/ directories."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline on one MOF (K⁺, 16 cores)
  python -m mof_screen MOF1.cif --ion K --cores 16

  # Run ONLY adsorption energy
  python -m mof_screen MOF1.cif --only adsorption --cores 16

  # Run ONLY OCV  (reads cached adsorption result)
  python -m mof_screen MOF1.cif --only ocv

  # Run ONLY ionic conductivity with a NEB barrier file
  python -m mof_screen MOF1.cif --only ionic_conductivity --barrier-file neb.txt

  # Re-run adsorption (clear its cache)
  python -m mof_screen MOF1.cif --force-rerun adsorption --cores 16

  # Dry-run: show what would execute without running DFT
  python -m mof_screen MOF1.cif --dry-run --ion K

  # Batch: screen and rank multiple MOFs
  python -m mof_screen MOF1.cif MOF2.cif MOF3.cif --ion K --rank --cores 16

  # Fast screening pass, then high-accuracy for the best candidate
  python -m mof_screen MOF1.cif --fast --ion K --cores 16
  python -m mof_screen MOF1.cif --high-accuracy --ion K --cores 32
        """,
    )

    # ---- Positional ----
    parser.add_argument(
        "cif_files", nargs="+",
        help="One or more CIF files to screen",
    )

    # ---- Execution control ----
    parser.add_argument(
        "--only", nargs="+", metavar="PARAM",
        choices=ALL_PARAMS,
        help=f"Run only the listed parameter(s). Choices: {ALL_PARAMS}",
    )
    parser.add_argument(
        "--force-rerun", nargs="+", metavar="PARAM",
        choices=ALL_PARAMS, dest="force_rerun",
        help="Clear cache and re-run the listed parameter(s).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print execution plan without running any DFT.",
    )
    parser.add_argument(
        "--rank", action="store_true",
        help="After all CIFs are done, print a ranked comparison table.",
    )

    # ---- DFT settings ----
    parser.add_argument(
        "--cores", "-n", type=int, default=16,
        help="OMP_NUM_THREADS per process (default: 16)",
    )
    parser.add_argument(
        "--mpi-ranks", type=int, default=1, dest="mpi_ranks",
        help="Number of MPI ranks (default: 1)",
    )
    parser.add_argument(
        "--high-accuracy", action="store_true",
        help="Use TZV2P basis set (3-5× slower, publication quality)",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Fast mode: 400 Ry cutoff only (slightly faster per step)",
    )
    parser.add_argument(
        "--screening", action="store_true",
        help=(
            "Screening mode: ALL 5 speed-up settings applied\n"
            "  400 Ry cutoff  → ~30%% faster per SCF step\n"
            "  EPS_SCF 1E-6   → ~10-20 fewer SCF iterations per GEO step\n"
            "  MAX_FORCE 0.001 → ~30-50%% fewer GEO_OPT steps\n"
            "  MAX_GEO 150    → hard cap on geometry steps\n"
            "Expected total speedup: 40-60%% vs production.\n"
            "E_ads accuracy: \u00b10.05-0.10 eV. Good for ranking, NOT for publication."
        ),
    )

    # ---- Ion settings ----
    parser.add_argument(
        "--ion", type=str, default="K",
        choices=SUPPORTED_IONS,
        help="Ion to insert (default: K)",
    )
    parser.add_argument(
        "--n-ions", type=int, default=1, dest="n_ions",
        help="Number of ions to insert per unit cell (default: 1)",
    )
    parser.add_argument(
        "--cell-opt", action="store_true",
        help="Relax cell vectors during adsorption GEO_OPT",
    )
    parser.add_argument(
        "--multiplicity", type=int, default=None,
        help="Spin multiplicity override (default: auto-detect)",
    )
    parser.add_argument(
        "--temperature", type=float, default=300.0,
        help="Temperature (K) for Nernst-Einstein ionic conductivity (default: 300)",
    )

    # ---- References ----
    parser.add_argument(
        "--ref-energies", type=str, default=None, dest="ref_energies",
        help="JSON file with pre-computed elemental reference energies",
    )
    parser.add_argument(
        "--compute-refs", action="store_true",
        help="Run CP2K to compute self-consistent elemental references",
    )
    parser.add_argument(
        "--ion-ref-ev", type=float, default=None, dest="ion_ref_ev",
        help="Bulk reference energy eV/atom for the chosen ion (overrides auto)",
    )

    # ---- Optional inputs ----
    parser.add_argument(
        "--barrier-file", type=str, default=None, dest="barrier_file",
        help="Path to pre-computed NEB barrier file (plain eV or CP2K NEB output)",
    )
    parser.add_argument(
        "--neb-images", type=str, default=None, dest="neb_images",
        help=(
            "Path to directory containing NEB image XYZ files (image00.xyz...imageN.xyz). "
            "Generates a CP2K CI-NEB (BAND) input and runs DFT to compute the diffusion "
            "barrier. Supersedes --barrier-file when both are provided."
        ),
    )
    parser.add_argument(
        "--nproc-rep", type=int, default=2, dest="nproc_rep",
        help="MPI ranks per NEB replica (default 2). Total MPI = nproc-rep x n_images.",
    )
    parser.add_argument(
        "--dos", action="store_true",
        help="Parse CP2K PDOS files for density of states",
    )
    parser.add_argument(
        "--bader", action="store_true",
        help=(
            "Parse Bader/Hirshfeld/Mulliken charges from adsorption output.\n"
            "Requires &PRINT → &HIRSHFELD T &END in CP2K input (or CUBE files + bader exe)."
        ),
    )
    parser.add_argument(
        "--competing-phases", type=str, default=None, dest="competing_phases",
        help=(
            "Path to a JSON file mapping reduced formula → E_form/atom (eV/atom) "
            "for competing phases used in the convex hull calculation.\n"
            "Example: {\"Fe2O3\": -0.72, \"FeO\": -0.51}.\n"
            "If omitted, the single-compound approximation is used (eHull ≈ E_form/atom)."
        ),
    )

    # ---- Misc ----
    parser.add_argument(
        "--cp2k-data-dir", type=str, default=None, dest="cp2k_data_dir",
        help="Override CP2K_DATA_DIR environment variable",
    )

    args = parser.parse_args(argv)

    # ---- Validate CIF paths ----
    for cif in args.cif_files:
        if not os.path.exists(cif):
            log.error(f"CIF not found: {cif}")
            sys.exit(1)

    all_results = []

    # ---- Run each CIF ----
    for cif_path in args.cif_files:
        log.info("")
        log.info(f"{'═'*65}")
        log.info(f"  Processing: {cif_path}")
        log.info(f"{'═'*65}")

        try:
            # ---- Load competing phases JSON if provided ----
            competing_phases: dict = {}
            if args.competing_phases:
                cp_path = args.competing_phases
                if not os.path.exists(cp_path):
                    log.error(f"Competing phases file not found: {cp_path}")
                    sys.exit(1)
                with open(cp_path) as fh:
                    competing_phases = json.load(fh)
                log.info(f"  Loaded {len(competing_phases)} competing phases from {cp_path}")

            result = run_screening(
                cif_path          = cif_path,
                only              = args.only,
                force_rerun       = args.force_rerun,
                dry_run           = args.dry_run,
                num_cores         = args.cores,
                mpi_ranks         = args.mpi_ranks,
                use_high_accuracy = args.high_accuracy,
                fast_mode         = args.fast,
                screening_mode    = args.screening,
                ion_symbol        = args.ion,
                n_ions            = args.n_ions,
                cell_opt          = args.cell_opt,
                user_mult         = args.multiplicity,
                temperature_k     = args.temperature,
                ref_energies_file = args.ref_energies,
                compute_refs      = args.compute_refs,
                ion_ref_ev        = args.ion_ref_ev,
                barrier_file      = args.barrier_file,
                neb_images        = args.neb_images,
                nproc_rep         = args.nproc_rep,
                compute_dos       = args.dos,
                cp2k_data_dir     = args.cp2k_data_dir,
                competing_phases  = competing_phases,
                compute_bader     = args.bader,
            )
            all_results.append((cif_path, result))
        except Exception as exc:
            import traceback as _tb
            log.error(f"  Screening failed for {cif_path}: {exc}")
            log.error("  Full traceback:\n" + _tb.format_exc())
            all_results.append((cif_path, None))

    # ---- Batch ranking ----
    if args.rank and len(all_results) > 1:
        _print_batch_ranking(all_results, args.ion)


# ---------------------------------------------------------------------------
# Batch ranking table
# ---------------------------------------------------------------------------
def _print_batch_ranking(results, ion_symbol: str) -> None:
    """
    Score and rank all screened MOFs by a weighted composite score.

    Scoring (lower is better for anodes):
        E_ads      : weight 0.30  (more negative = better, target −0.5 to −2 eV)
        E_form     : weight 0.20  (negative = stable)
        OCV        : weight 0.20  (0–1 V preferred)
        Vol exp.   : weight 0.15  (< 5 % preferred)
        σ rank     : weight 0.15  (EXCELLENT=4, GOOD=3, MODERATE=2, POOR=1)
    """
    _RANK_SCORE = {
        "EXCELLENT": 4, "GOOD": 3, "MODERATE": 2, "POOR": 1, "UNKNOWN": 0,
    }

    scored = []
    for cif_path, res in results:
        if res is None:
            continue
        name = Path(cif_path).stem

        e_ads = res.adsorption.e_ads_ev   or 0.0
        e_form = res.formation.e_form_per_atom_ev or 0.0
        ocv   = res.ocv.ocv_v             or 0.0
        vol   = res.volume.expansion_pct  or 100.0
        rank_key = res.ionic_conductivity.rank.split()[0] if res.ionic_conductivity.rank else "UNKNOWN"
        sigma_score = _RANK_SCORE.get(rank_key, 0)

        # Normalise each metric to a 0–1 score (higher = better anode)
        s_ads   = max(0.0, min(1.0, (-e_ads - 0.3) / 2.0))          # target −0.3 to −2.3
        s_form  = max(0.0, min(1.0, (-e_form) / 2.0))                # negative = 1
        s_ocv   = max(0.0, min(1.0, 1.0 - ocv / 2.0))               # 0 V = 1, 2 V = 0
        s_vol   = max(0.0, min(1.0, (20.0 - vol) / 20.0))           # 0 % = 1, 20 % = 0
        s_sigma = sigma_score / 4.0

        composite = (
            0.30 * s_ads + 0.20 * s_form +
            0.20 * s_ocv + 0.15 * s_vol + 0.15 * s_sigma
        )
        scored.append((composite, name, e_ads, e_form, ocv, vol,
                       res.ionic_conductivity.rank))

    scored.sort(reverse=True)

    print("\n" + "═" * 90)
    print(f"  BATCH RANKING  ({len(scored)} MOFs)  |  Ion: {ion_symbol}⁺")
    print("═" * 90)
    print(
        f"  {'Rank':>4s}  {'MOF':<20s}  {'Score':>6s}  "
        f"{'E_ads':>8s}  {'E_form/at':>10s}  "
        f"{'OCV':>6s}  {'ΔV%':>6s}  Conductivity"
    )
    print("  " + "─" * 86)
    for i, (score, name, e_ads, e_form, ocv, vol, rank) in enumerate(scored, 1):
        print(
            f"  {i:>4d}  {name:<20s}  {score:>6.3f}  "
            f"{e_ads:>+8.3f} eV  {e_form:>+8.3f} eV/at  "
            f"{ocv:>6.3f} V  {vol:>5.1f}%  {rank}"
        )
    print("═" * 90)
    best = scored[0][1] if scored else "N/A"
    print(f"\n  ★ BEST K⁺ ANODE CANDIDATE: {best}")
    print()


if __name__ == "__main__":
    main()
