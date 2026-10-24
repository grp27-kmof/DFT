"""
mof_screen/cp2k_runner.py
=========================
CP2K process launcher with live progress dashboard integration.

Key design points:
  - Streams CP2K output directly to file (no RAM buffering)
  - Feeds real-time SCF / GEO_OPT step counts to the ProgressDashboard
  - Supports WFN and coordinate restarts
  - MPI + OpenMP support
  - Never blocks; progress monitoring runs in a daemon thread
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from mof_screen.constants import CP2K_BINARY_NAMES, MAX_GEO_STEPS, MAX_SCF_STEPS
from mof_screen.parsers import parse_cp2k_progress, parse_neb_progress
from mof_screen.progress import ProgressDashboard

log = logging.getLogger("mof_dft.runner")


# ---------------------------------------------------------------------------
# Binary discovery
# ---------------------------------------------------------------------------
def find_cp2k() -> Optional[str]:
    """Search PATH for a usable CP2K binary. Returns the path or None."""
    for name in CP2K_BINARY_NAMES:
        path = shutil.which(name)
        if path:
            log.debug(f"  CP2K binary: {path}")
            return path
    return None


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------
def run_cp2k(
    inp_file: str,
    out_file: str,
    num_cores: int,
    mpi_ranks: int                   = 1,
    restart_wfn:    Optional[str]    = None,
    restart_coord:  Optional[str]    = None,
    label: str                       = "",
    param_name: str                  = "",
    dashboard: Optional[ProgressDashboard] = None,
    max_scf: int                     = MAX_SCF_STEPS,
    max_geo: int                     = MAX_GEO_STEPS,
    cwd: Optional[str]               = None,
) -> bool:
    """
    Launch CP2K and stream progress to the dashboard.

    Parameters
    ----------
    inp_file    : CP2K input file path
    out_file    : CP2K output file path (will be written / appended)
    num_cores   : OMP_NUM_THREADS per process
    mpi_ranks   : number of MPI ranks (1 = pure OpenMP)
    restart_wfn : path to a prior WFN restart file
    restart_coord: path to a prior coordinate restart file (.restart)
    label       : human-readable label for log messages
    param_name  : parameter key for the dashboard (e.g. "adsorption")
    dashboard   : live ProgressDashboard to update during the run
    max_scf     : expected maximum SCF steps (for % estimation)
    max_geo     : expected maximum GEO_OPT steps (for % estimation)

    Returns True on success (exit code 0), False otherwise.
    """
    cp2k = find_cp2k()
    if cp2k is None:
        log.error("CP2K binary not found.  Run: conda activate dft_env")
        return False

    # ---- Patch WFN restart into input ----
    if restart_wfn and os.path.exists(restart_wfn):
        log.info(f"    Restarting WFN from: {restart_wfn}")
        txt = Path(inp_file).read_text()
        import re as _re
        
        # 1. Change SCF_GUESS ATOMIC to RESTART
        new_txt = _re.sub(
            r"^(\s+)SCF_GUESS\s+ATOMIC",
            r"\g<1>SCF_GUESS  RESTART",
            txt,
            flags=_re.MULTILINE
        )
        
        # 2. Inject WFN_RESTART_FILE_NAME into the &DFT section
        if new_txt != txt:
            new_txt = _re.sub(
                r"^(\s+)&DFT\s*$",
                f"\\g<1>&DFT\n\\g<1>  WFN_RESTART_FILE_NAME {restart_wfn}",
                new_txt,
                flags=_re.MULTILINE
            )
        else:
            log.warning("    WFN patch: SCF_GUESS ATOMIC not found — restart won't apply.")
            
        Path(inp_file).write_text(new_txt)


    # ---- Patch coordinate restart into &GLOBAL ----
    if restart_coord and os.path.exists(restart_coord):
        log.info(f"    Restarting coords from: {restart_coord}")
        txt = Path(inp_file).read_text()
        if "&EXT_RESTART" not in txt:
            ext = (
                f"\n&EXT_RESTART\n"
                f"  RESTART_FILE_NAME {restart_coord}\n"
                f"  RESTART_COUNTERS .FALSE.\n"
                f"  RESTART_POS .TRUE.\n"
                f"  RESTART_CELL .FALSE.\n"
                f"&END EXT_RESTART\n"
            )
            txt = txt.replace("&END GLOBAL", "&END GLOBAL\n" + ext)
        Path(inp_file).write_text(txt)

    # ---- Build launch command ----
    # cp2k.ssmp / cp2k.sopt are shared-memory-only (OpenMP) builds.
    # They cannot be launched with mpirun — doing so causes "No such file or directory".
    # For these variants we run directly and multiply the thread count by mpi_ranks
    # to still utilise the full core budget.
    is_smp = cp2k.endswith(".ssmp") or cp2k.endswith(".sopt")
    if is_smp:
        mpi_cmd = []
        effective_threads = num_cores * mpi_ranks   # pack all parallelism into OMP
        log.info(f"    Note: {Path(cp2k).name} is an OpenMP-only build — skipping mpirun")
    else:
        mpi_cmd = ["mpirun", "-np", str(mpi_ranks)] if mpi_ranks > 1 else []
        effective_threads = num_cores
    cmd = mpi_cmd + [cp2k, "-i", inp_file, "-o", out_file]
    log.info(f"    CMD: {' '.join(cmd)}")


    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(effective_threads)
    env["OMP_PROC_BIND"]   = "close"
    env["OMP_PLACES"]      = "cores"
    # Prevent nested multithreading in BLAS libraries when CP2K handles OpenMP
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"]      = "1"

    # ---- Start dashboard progress updater thread ----
    stop_event = threading.Event()
    # Detect CI-NEB / BAND jobs by label so the updater uses the richer NEB parser
    _is_neb = any(kw in label.upper() for kw in ("NEB", "BAND"))
    if dashboard and param_name:
        updater = threading.Thread(
            target=_dashboard_updater,
            args=(out_file, param_name, dashboard, stop_event, max_scf, max_geo, _is_neb),
            daemon=True,
        )
        updater.start()
    else:
        updater = None

    # CP2K must be launched from a valid, readable directory.
    # CP2K 2024.x calls getcwd() at startup and raises SIGABRT (exit -6)
    # if the process working directory is unresolvable or inaccessible.
    # Prefer an explicitly supplied cwd; fall back to the input file's parent.
    out_dir_cwd = cwd if cwd is not None else str(Path(inp_file).parent)

    t0 = time.time()
    returncode = None
    # When restarting a NEB run, append to the existing output so the parser
    # can see ENERGIES blocks from ALL prior steps (not just the current restart).
    # For fresh runs, overwrite so we get a clean log.
    out_mode = "a" if (restart_coord and os.path.exists(out_file)) else "w"
    try:
        with open(out_file, out_mode) as fout, \
             open(out_file + ".stderr", out_mode) as ferr:
            proc = subprocess.Popen(
                cmd, stdout=fout, stderr=ferr, env=env,
                cwd=out_dir_cwd,
            )
        try:
            proc.wait()
        except KeyboardInterrupt:
            log.warning(f"    KeyboardInterrupt received — terminating CP2K (PID {proc.pid})...")
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            log.warning(f"    CP2K terminated. Restart files have been saved.")
            stop_event.set()
            if updater:
                updater.join(timeout=3)
            return False
        returncode = proc.returncode
    except Exception as exc:
        log.error(f"    Failed to launch CP2K: {exc}")
        if updater:
            stop_event.set()
            updater.join(timeout=3)
        return False

    # ---- Stop progress updater ----
    stop_event.set()
    if updater:
        updater.join(timeout=5)

    elapsed = time.time() - t0
    log.info(f"    {label} finished in {elapsed / 60:.1f} min (exit {returncode})")

    if returncode != 0:
        log.error(f"    Non-zero exit code {returncode}. See {out_file}.stderr")
        return False
    return True


# ---------------------------------------------------------------------------
# Dashboard updater thread target
# ---------------------------------------------------------------------------
def _dashboard_updater(
    out_file: str,
    param_name: str,
    dashboard: ProgressDashboard,
    stop_event: threading.Event,
    max_scf: int,
    max_geo: int,
    is_neb: bool = False,
) -> None:
    """
    Read the CP2K output file once per second and push updated progress
    counters to the dashboard.  For CI-NEB (BAND) jobs a richer set of
    metrics (BAND step, barrier estimate, max force) is shown.
    """
    while not stop_event.wait(timeout=1.0):
        if is_neb:
            neb = parse_neb_progress(out_file)
            step    = neb["band_step"]
            scf     = neb["scf_steps"]
            barrier = neb["barrier_ev"]
            force   = neb["max_force"]
            n_rep   = neb["n_replicas"]

            phase_parts = [f"BAND step {step}"]
            if scf > 0:
                phase_parts.append(f"SCF {scf}")
            if barrier is not None:
                phase_parts.append(f"barrier≈{barrier:.3f}eV")
            if force is not None:
                # `force` field is repurposed to hold dist_spread (see parse_neb_progress)
                phase_parts.append(f"dist_σ={force:.3f}Å")
            if n_rep > 0:
                phase_parts.append(f"{n_rep} replicas")
            phase = " | ".join(phase_parts)

            # Progress %: use step count as a rough proxy (typical NEB ~50-200 steps)
            # Clamp to 99% since we don't know the target step count.
            pct: Optional[float] = None
            if step > 0:
                # Assume ~100 steps typical; clamp to 99%
                pct = min(99.0, step / 1.0)   # 1 step = 1%, saturates gracefully

            dashboard.update(
                param_name,
                phase    = phase,
                pct      = pct,
                scf_step = scf,
                geo_step = step,
                max_scf  = max_scf,
                max_geo  = max_geo,
            )
        else:
            data = parse_cp2k_progress(out_file)

            geo  = data["geo_steps"]
            scf  = data["scf_steps"]

            if geo > 0:
                phase = f"GEO step {geo}/{max_geo} | SCF {scf}/{max_scf}"
            else:
                phase = f"SCF step {scf}/{max_scf}"
                if data["outer_scf"] > 0:
                    phase += f" (outer {data['outer_scf']})"

            if data["energy"] is not None:
                phase += f" | E={data['energy']:.4f} eV"

            dashboard.update(
                param_name,
                phase    = phase,
                scf_step = scf,
                geo_step = geo,
                max_scf  = max_scf,
                max_geo  = max_geo,
            )
