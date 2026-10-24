"""
mof_screen/calc_ocv.py
=======================
PARAMETER: Open-Circuit Voltage  (V)

Calculation type : Instant — derived from E_ads
Standalone       : YES — needs adsorption.json in the results dir
Cache file       : <results_dir>/ocv.json

Formula
-------
    OCV = −E_ads / (n_ions × z × e)      [eV → V]

    z = ion charge (1 for K⁺)
    Low OCV (0–1 V) is desirable for anodes to avoid electrolyte decomposition.

Usage (standalone)
------------------
    from mof_screen.calc_ocv import calc_ocv
    result = calc_ocv(
        e_ads_ev=-0.85, n_ions=1, ion_symbol="K",
        cache=cache, dashboard=dashboard,
    )
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from mof_screen.constants import ION_CHARGE
from mof_screen.dataclasses_ import OCVResult
from mof_screen.printer import print_ocv
from mof_screen.progress import ProgressDashboard
from mof_screen.state import StateCache

log = logging.getLogger("mof_dft.ocv")

_PARAM = "ocv"


def calc_ocv(
    e_ads_ev: Optional[float],
    n_ions: int,
    ion_symbol: str                        = "K",
    dashboard: Optional[ProgressDashboard] = None,
    cache: Optional[StateCache]            = None,
) -> OCVResult:
    """
    Derive the open-circuit voltage from the K⁺ adsorption energy.
    Instant — no DFT call needed.
    """
    log.info("")
    log.info("━" * 64)
    log.info("  OPEN-CIRCUIT VOLTAGE  (instant — derived from E_ads)")
    log.info("━" * 64)

    z      = ION_CHARGE.get(ion_symbol, 1)
    result = OCVResult(z_charge=z, n_ions=n_ions, ion_symbol=ion_symbol)

    # ---- Cache hit ----
    if cache and cache.is_done(_PARAM):
        log.info("  ✓ Loading OCV from cache.")
        data = cache.load(_PARAM) or {}
        result.ocv_v      = data.get("ocv_v")
        result.z_charge   = data.get("z_charge", z)
        result.n_ions     = data.get("n_ions", n_ions)
        result.ion_symbol = data.get("ion_symbol", ion_symbol)
        if dashboard:
            dashboard.mark_cached(_PARAM)
        print_ocv(result)
        return result

    if dashboard:
        dashboard.mark_running(_PARAM)

    t0 = time.time()

    if e_ads_ev is None:
        log.warning("  E_ads not available — OCV cannot be computed.")
        if dashboard:
            dashboard.mark_failed(_PARAM)
        print_ocv(result)
        return result

    result.ocv_v = -e_ads_ev / (n_ions * z)

    log.info(f"  E_ads      = {e_ads_ev:.6f} eV")
    log.info(f"  z (charge) = {z}  ({ion_symbol}⁺)")
    log.info(f"  n_ions     = {n_ions}")
    log.info(f"  OCV        = {result.ocv_v:.6f} V")

    elapsed = (time.time() - t0) / 60.0
    log.info(f"  Elapsed    = {elapsed * 60:.3f} s  (instant)")

    if cache:
        cache.save(_PARAM, result)
    if dashboard:
        dashboard.mark_done(_PARAM)

    print_ocv(result)
    return result
