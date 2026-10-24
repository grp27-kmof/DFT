"""
mof_screen/calc_dos.py
=======================
PARAMETER: Density of States  (parse CP2K PDOS files)

Calculation type : Instant — parses .pdos files from prior ENERGY run
Standalone       : YES — only needs *.pdos files in the results dir
Cache file       : <results_dir>/dos.json

CP2K writes PDOS files named  <label>-PDOS-k1-1.pdos  during any
ENERGY run.  Each file has a header line:
    # Fermi energy [eV]:  X.XXX

Usage (standalone)
------------------
    from mof_screen.calc_dos import calc_dos
    result = calc_dos(
        out_dir=out_dir,
        fermi_fallback_ev=None,
        cache=cache, dashboard=dashboard,
    )
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Optional

from mof_screen.dataclasses_ import DOSResult
from mof_screen.printer import print_dos
from mof_screen.progress import ProgressDashboard
from mof_screen.state import StateCache

log = logging.getLogger("mof_dft.dos")

_PARAM = "dos"

_FERMI_RE = re.compile(
    r"Fermi energy[\s\[\w]*\]?:?\s*(-?[\d.]+(?:[Ee][+-]?\d+)?)",
    re.IGNORECASE,
)


def calc_dos(
    out_dir: Path,
    fermi_fallback_ev: Optional[float]     = None,
    dashboard: Optional[ProgressDashboard] = None,
    cache: Optional[StateCache]            = None,
) -> DOSResult:
    """
    Parse CP2K PDOS output files from the results directory.

    Extracts:
      - List of PDOS file paths
      - Fermi energy (from PDOS header or fallback from adsorption result)
    """
    log.info("")
    log.info("━" * 64)
    log.info("  DENSITY OF STATES  (instant — parses CP2K PDOS files)")
    log.info("━" * 64)

    result = DOSResult()

    # ---- Cache hit ----
    if cache and cache.is_done(_PARAM):
        log.info("  ✓ Loading DOS from cache.")
        data = cache.load(_PARAM) or {}
        result.pdos_files   = data.get("pdos_files", [])
        result.fermi_ev     = data.get("fermi_ev")
        result.n_pdos_files = data.get("n_pdos_files", 0)
        result.parsed       = data.get("parsed", False)
        if dashboard:
            dashboard.mark_cached(_PARAM)
        print_dos(result)
        return result

    if dashboard:
        dashboard.mark_running(_PARAM)

    t0 = time.time()

    pdos_files = sorted(out_dir.glob("*.pdos"))
    if not pdos_files:
        log.info("  No .pdos files found.")
        log.info("  PDOS files are written during ENERGY runs automatically.")
        if dashboard:
            dashboard.mark_skipped(_PARAM)
        print_dos(result)
        return result

    result.pdos_files   = [str(p) for p in pdos_files]
    result.n_pdos_files = len(pdos_files)

    # Extract Fermi energy from any PDOS file header
    for pf in pdos_files:
        try:
            with open(pf) as fh:
                for line in fh:
                    if not line.startswith("#"):
                        break
                    mm = _FERMI_RE.search(line)
                    if mm:
                        result.fermi_ev = float(mm.group(1))
                        break
        except OSError:
            pass
        if result.fermi_ev is not None:
            break

    # Fallback: use HOMO from adsorption/bandgap as Fermi energy
    if result.fermi_ev is None and fermi_fallback_ev is not None:
        result.fermi_ev = fermi_fallback_ev
        log.info("  Fermi energy: using fallback from prior calc.")

    result.parsed = True
    log.info(f"  Found {result.n_pdos_files} PDOS file(s).")
    if result.fermi_ev is not None:
        log.info(f"  Fermi energy : {result.fermi_ev:.4f} eV")
    for pf in pdos_files[:5]:
        log.info(f"    {pf.name}")
    if len(pdos_files) > 5:
        log.info(f"    … and {len(pdos_files) - 5} more")

    elapsed = (time.time() - t0) / 60.0
    log.info(f"  Elapsed      = {elapsed * 60:.2f} s")

    if cache:
        cache.save(_PARAM, result)
    if dashboard:
        dashboard.mark_done(_PARAM)

    print_dos(result)
    return result
