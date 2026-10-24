"""
mof_screen/calc_ionic_conductivity.py
=======================================
PARAMETER: Ionic Conductivity  σ (S/cm)

Replaces the bandgap calculation entirely.

Calculation type : Derived (no DFT) — uses E_ads and diffusion barrier
Standalone       : YES — needs adsorption.json (or --e-ads-ev override)
                         and optionally a barrier file
Cache file       : <results_dir>/ionic_conductivity.json

Physics
-------
When a diffusion barrier Ea is available (from a pre-computed NEB run):

    Full Nernst-Einstein:
        D   = D0 × exp(−Ea / kB T)          [Arrhenius diffusion]
        n   = N_ions / V_cell               [carrier density, Å⁻³ → cm⁻³]
        σ   = n × z² × e² × D / (kB × T)  [S/cm]

When no barrier is available (estimate from E_ads alone):

    The adsorption energy is a proxy for the ion-framework interaction
    strength.  Very negative E_ads → ion is trapped → low conductivity.
    Moderate E_ads → fast hopping → high conductivity.

    Rank thresholds (tuned for K⁺ in MOF anodes):
        E_ads < −2.0 eV   →  POOR       (ion immobile)
        −2.0 to −1.0 eV   →  MODERATE
        −1.0 to −0.3 eV   →  GOOD
        −0.3 to  0.0 eV   →  EXCELLENT  (weak binding = fast hopping)
        > 0.0 eV           →  UNFAVORABLE

Usage (standalone)
------------------
    from mof_screen.calc_ionic_conductivity import calc_ionic_conductivity
    result = calc_ionic_conductivity(
        e_ads_ev=-0.85,
        n_ions=1,
        cell_volume_A3=12500.0,
        barrier_ev=0.35,        # optional
        temperature_k=300.0,
        ion_symbol="K",
        cache=cache,
        dashboard=dashboard,
    )
"""

from __future__ import annotations

import logging
import math
import time
from typing import Optional

from mof_screen.constants import (
    D0_CM2S,
    DEFAULT_TEMPERATURE,
    ELEM_CHARGE,
    ION_CHARGE,
    KB_EV,
)
from mof_screen.dataclasses_ import DiffusionBarrierResult, IonicConductivityResult
from mof_screen.printer import print_ionic_conductivity
from mof_screen.progress import ProgressDashboard
from mof_screen.state import StateCache

log = logging.getLogger("mof_dft.ionic_conductivity")

_PARAM = "ionic_conductivity"

# 1 Å³ = 1e-24 cm³  →  n (Å⁻³) × 1e24 = n (cm⁻³)
_ANG3_TO_CM3: float = 1.0e-24


def calc_ionic_conductivity(
    e_ads_ev: Optional[float],
    n_ions: int,
    cell_volume_A3: Optional[float],
    barrier: Optional[DiffusionBarrierResult] = None,
    temperature_k: float                     = DEFAULT_TEMPERATURE,
    ion_symbol: str                          = "K",
    dashboard: Optional[ProgressDashboard]   = None,
    cache: Optional[StateCache]              = None,
) -> IonicConductivityResult:
    """
    Compute (or estimate) the ionic conductivity σ (S/cm).

    Parameters
    ----------
    e_ads_ev        : K⁺ adsorption energy (eV); required for rank estimation
    n_ions          : number of K⁺ ions inserted in the MOF supercell
    cell_volume_A3  : MOF unit-cell volume (Å³); required for full σ
    barrier         : DiffusionBarrierResult (optional); enables full Nernst-Einstein
    temperature_k   : simulation temperature (K); default 300 K
    ion_symbol      : ion species (default "K")
    dashboard       : ProgressDashboard for live status updates
    cache           : StateCache for checkpointing
    """
    log.info("")
    log.info("━" * 64)
    log.info("  IONIC CONDUCTIVITY  (Nernst-Einstein)  —  derived")
    log.info("━" * 64)

    z = ION_CHARGE.get(ion_symbol, 1)
    result = IonicConductivityResult(
        temperature_k = temperature_k,
        ion_symbol    = ion_symbol,
        z_charge      = z,
    )

    # ---- Cache hit ----
    if cache and cache.is_done(_PARAM):
        log.info("  ✓ Loading ionic_conductivity from cache.")
        data = cache.load(_PARAM) or {}
        result.sigma_s_cm      = data.get("sigma_s_cm")
        result.diffusion_coeff = data.get("diffusion_coeff")
        result.barrier_ev      = data.get("barrier_ev")
        result.carrier_density = data.get("carrier_density")
        result.rank            = data.get("rank", "UNKNOWN")
        result.method          = data.get("method", "UNKNOWN")
        result.elapsed_min     = data.get("elapsed_min", 0.0)
        if dashboard:
            dashboard.mark_cached(_PARAM)
        print_ionic_conductivity(result)
        return result

    if dashboard:
        dashboard.mark_running(_PARAM)

    t0 = time.time()

    # ================================================================
    # PATH A: Full Nernst-Einstein when barrier is available
    # ================================================================
    barrier_ev = barrier.barrier_ev if (barrier and barrier.available) else None

    if barrier_ev is not None and cell_volume_A3 and cell_volume_A3 > 0:
        # Diffusion coefficient D (cm²/s)
        D = D0_CM2S * math.exp(-barrier_ev / (KB_EV * temperature_k))

        # Carrier density n (cm⁻³)
        # n = N_ions / V_cell   (ions per unit cell / cell volume in cm³)
        n_cm3 = n_ions / (cell_volume_A3 * _ANG3_TO_CM3)

        # Nernst-Einstein: σ = n z² e² D / (kB T)
        # Units: (cm⁻³)(C²)(cm²/s) / (J) = S/cm  ✓
        kBT_joules = KB_EV * temperature_k * ELEM_CHARGE   # J
        sigma      = (n_cm3 * z**2 * ELEM_CHARGE**2 * D) / kBT_joules

        result.sigma_s_cm      = sigma
        result.diffusion_coeff = D
        result.barrier_ev      = barrier_ev
        result.carrier_density = n_cm3
        result.method          = "nernst_einstein"

        # Rank by conductivity order-of-magnitude
        if sigma >= 1e-2:
            result.rank = "EXCELLENT  (≥ 10⁻² S/cm)"
        elif sigma >= 1e-4:
            result.rank = "GOOD       (10⁻⁴ – 10⁻² S/cm)"
        elif sigma >= 1e-6:
            result.rank = "MODERATE   (10⁻⁶ – 10⁻⁴ S/cm)"
        else:
            result.rank = "POOR       (< 10⁻⁶ S/cm)"

        log.info(f"  D           = {D:.3e} cm²/s  (Ea={barrier_ev:.4f} eV)")
        log.info(f"  n           = {n_cm3:.3e} cm⁻³")
        log.info(f"  σ           = {sigma:.3e} S/cm")
        log.info(f"  Rank        : {result.rank}")

    # ================================================================
    # PATH B: Qualitative rank from E_ads alone
    # ================================================================
    elif e_ads_ev is not None:
        result.method    = "eads_estimate"
        result.barrier_ev = None

        if e_ads_ev < -2.0:
            result.rank = "POOR       (E_ads < −2.0 eV: ion likely trapped)"
        elif e_ads_ev < -1.0:
            result.rank = "MODERATE   (E_ads −2.0 to −1.0 eV)"
        elif e_ads_ev < -0.3:
            result.rank = "GOOD       (E_ads −1.0 to −0.3 eV)"
        elif e_ads_ev < 0.0:
            result.rank = "EXCELLENT  (E_ads −0.3 to 0.0 eV: fast hopping)"
        else:
            result.rank = "UNFAVORABLE (E_ads > 0.0 eV)"

        log.info(f"  E_ads       = {e_ads_ev:.4f} eV")
        log.info(f"  Rank (estimate): {result.rank}")
        log.info(  "  Note: provide --barrier-file for quantitative σ (S/cm)")

    else:
        log.warning("  Neither E_ads nor barrier available — cannot rank conductivity.")
        result.rank   = "UNKNOWN"
        result.method = "none"
        if dashboard:
            dashboard.mark_failed(_PARAM)
        print_ionic_conductivity(result)
        return result

    result.elapsed_min = (time.time() - t0) / 60.0

    if cache:
        cache.save(_PARAM, result)
    if dashboard:
        dashboard.mark_done(_PARAM)

    print_ionic_conductivity(result)
    return result
