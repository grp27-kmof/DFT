"""
mof_screen/calc_volume.py
==========================
PARAMETER: Volume Expansion  (%)

Calculation type : Instant — reads adsorption.out cell volume
Standalone       : YES — needs adsorption.json in the results dir
Cache file       : <results_dir>/volume.json

Formula
-------
    ΔV % = (V_after − V_before) / V_before × 100

    < 5 % expansion is considered excellent for anode materials.

Usage (standalone)
------------------
    from mof_screen.calc_volume import calc_volume_expansion
    result = calc_volume_expansion(
        atoms=atoms, out_dir=out_dir,
        cell_opt=False, ion_symbol="K",
        cache=cache, dashboard=dashboard,
    )
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

from ase import Atoms
from ase.io import read

from mof_screen.dataclasses_ import VolumeResult
from mof_screen.parsers import extract_final_volume
from mof_screen.printer import print_volume
from mof_screen.progress import ProgressDashboard
from mof_screen.state import StateCache

log = logging.getLogger("mof_dft.volume")

_PARAM = "volume"


def calc_volume_expansion(
    atoms: Atoms,
    out_dir: Path,
    cell_opt: bool                         = False,
    ion_symbol: str                        = "K",
    dashboard: Optional[ProgressDashboard] = None,
    cache: Optional[StateCache]            = None,
) -> VolumeResult:
    """
    Compute volume expansion (%) from the adsorption GEO_OPT output.

    The volume is extracted from:
      1. 'CELL| Volume' lines in adsorption.out  (most reliable)
      2. The last frame of the trajectory XYZ file  (fallback)
      3. If fixed-cell GEO_OPT: expansion = 0 % by definition.

    Instant — no DFT call.
    """
    log.info("")
    log.info("━" * 64)
    log.info("  VOLUME EXPANSION  (instant — reads adsorption.out)")
    log.info("━" * 64)

    result = VolumeResult(cell_relaxed=cell_opt)

    # ---- Cache hit ----
    if cache and cache.is_done(_PARAM):
        log.info("  ✓ Loading volume from cache.")
        data = cache.load(_PARAM) or {}
        result.v_before_A3   = data.get("v_before_A3")
        result.v_after_A3    = data.get("v_after_A3")
        result.expansion_pct = data.get("expansion_pct")
        result.cell_relaxed  = data.get("cell_relaxed", cell_opt)
        if dashboard:
            dashboard.mark_cached(_PARAM)
        print_volume(result, ion_symbol)
        return result

    if dashboard:
        dashboard.mark_running(_PARAM)

    t0 = time.time()

    result.v_before_A3 = atoms.get_volume()
    out_file           = str(out_dir / "adsorption.out")

    # Try parsing from CP2K output
    v_after = extract_final_volume(out_file)

    # Fallback: read last frame of trajectory
    if v_after is None:
        for traj_name in [
            f"mof_{ion_symbol.lower()}-pos-1.xyz",
            "cp2k-pos-1.xyz",
        ]:
            traj_path = out_dir / traj_name
            if traj_path.exists():
                try:
                    v_after = read(str(traj_path), index=-1).get_volume()
                    log.info(f"  Volume from trajectory: {traj_path.name}")
                    break
                except Exception:
                    pass

    # For fixed-cell runs: expansion is zero by definition
    if v_after is None and not cell_opt:
        log.info("  Fixed-cell GEO_OPT → volume expansion = 0.000 %")
        log.info("  Use --cell-opt for true volume relaxation.")
        v_after = result.v_before_A3

    if v_after is None:
        log.error("  Could not determine relaxed volume.")
        if dashboard:
            dashboard.mark_failed(_PARAM)
        print_volume(result, ion_symbol)
        return result

    result.v_after_A3    = v_after
    result.expansion_pct = (
        (v_after - result.v_before_A3) / result.v_before_A3 * 100.0
    )

    log.info(f"  V(bare)       = {result.v_before_A3:.4f} Å³")
    log.info(f"  V(with {ion_symbol})   = {v_after:.4f} Å³")
    log.info(f"  Expansion     = {result.expansion_pct:.3f} %")

    elapsed = (time.time() - t0) / 60.0
    log.info(f"  Elapsed       = {elapsed * 60:.2f} s  (instant)")

    if cache:
        cache.save(_PARAM, result)
    if dashboard:
        dashboard.mark_done(_PARAM)

    print_volume(result, ion_symbol)
    return result
