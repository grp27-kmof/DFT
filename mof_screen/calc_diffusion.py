"""
mof_screen/calc_diffusion.py
=============================
PARAMETER: Diffusion Barrier  (eV)

Calculation type : Instant — reads a user-supplied NEB/CI-NEB barrier file
Standalone       : YES — needs only --barrier-file <path>
Cache file       : <results_dir>/diffusion.json

The file can be:
  1. A plain text file with a single float (eV) on the first non-blank line.
  2. A CP2K NEB output containing 'Activation energy: X.XXX eV'.

Usage (standalone)
------------------
    from mof_screen.calc_diffusion import calc_diffusion_barrier
    result = calc_diffusion_barrier(
        barrier_file="neb_result.txt",
        cache=cache, dashboard=dashboard,
    )
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Optional

from mof_screen.dataclasses_ import DiffusionBarrierResult
from mof_screen.printer import print_diffusion_barrier
from mof_screen.progress import ProgressDashboard
from mof_screen.state import StateCache

log = logging.getLogger("mof_dft.diffusion")

_PARAM = "diffusion"

_CP2K_NEB_RE = re.compile(
    r"Activation energy:\s*(-?[\d.]+(?:[Ee][+-]?\d+)?)\s*eV",
    re.IGNORECASE,
)


def calc_diffusion_barrier(
    barrier_file: Optional[str]            = None,
    dashboard: Optional[ProgressDashboard] = None,
    cache: Optional[StateCache]            = None,
) -> DiffusionBarrierResult:
    """
    Load the K⁺ diffusion barrier from a pre-computed NEB barrier file.

    Fully standalone — does not depend on any other calc module.
    """
    log.info("")
    log.info("━" * 64)
    log.info("  DIFFUSION BARRIER  (instant — reads NEB barrier file)")
    log.info("━" * 64)

    result = DiffusionBarrierResult()

    # ---- Cache hit ----
    if cache and cache.is_done(_PARAM):
        log.info("  ✓ Loading diffusion barrier from cache.")
        data = cache.load(_PARAM) or {}
        result.barrier_ev  = data.get("barrier_ev")
        result.source_file = data.get("source_file", "")
        result.available   = data.get("available", False)
        if dashboard:
            dashboard.mark_cached(_PARAM)
        print_diffusion_barrier(result)
        return result

    if dashboard:
        dashboard.mark_running(_PARAM)

    t0 = time.time()

    # ---- No file provided ----
    if not barrier_file:
        log.info("  No --barrier-file provided.  Skipping diffusion barrier.")
        if dashboard:
            dashboard.mark_skipped(_PARAM)
        print_diffusion_barrier(result)
        return result

    bf = Path(barrier_file)
    if not bf.exists():
        log.error(f"  Barrier file not found: {barrier_file}")
        if dashboard:
            dashboard.mark_failed(_PARAM)
        print_diffusion_barrier(result)
        return result

    result.source_file = str(bf)
    text               = bf.read_text()

    # 1. Try CP2K NEB format first ("Activation energy: X.XXX eV")
    m = _CP2K_NEB_RE.search(text)
    if m:
        result.barrier_ev = float(m.group(1))
        result.available  = True
        log.info(f"  Barrier (CP2K NEB): {result.barrier_ev:.4f} eV")
    else:
        # 2. Try extended XYZ (extxyz) multi-image format ("energy=... ")
        energy_matches = re.findall(r"energy\s*=\s*(-?[\d.]+)", text, re.IGNORECASE)
        if energy_matches:
            energies = [float(e) for e in energy_matches]
            # Convert to eV if energies differ across images
            max_e = max(energies)
            first_e = energies[0]
            barrier = max_e - first_e
            if abs(max_e - first_e) > 1e-6:
                result.barrier_ev = barrier
                result.available  = True
                log.info(f"  Barrier (parsed from extxyz {len(energies)} images): {result.barrier_ev:.4f} eV")
            else:
                log.warning(
                    f"  extxyz file found with {len(energies)} images, but ALL images have "
                    f"identical energy ({first_e:.6f} eV). "
                    "This is an uncalculated/interpolated initial NEB path (barrier = 0.0 eV)."
                )

        # 3. Fallback: plain float on the first non-header, non-comment line
        if not result.available:
            for line in text.splitlines():
                line = line.strip()
                # Ignore atom counts, comments, and extended xyz headers
                if not line or line.startswith("#") or line.isdigit() or "Lattice=" in line or "Properties=" in line:
                    continue
                try:
                    val = float(line)
                    result.barrier_ev = val
                    result.available  = True
                    log.info(f"  Barrier (plain text): {result.barrier_ev:.4f} eV")
                    break
                except ValueError:
                    pass

    if not result.available:
        log.error("  Could not parse a diffusion barrier value from the file.")
        if dashboard:
            dashboard.mark_failed(_PARAM)
        print_diffusion_barrier(result)
        return result

    elapsed = (time.time() - t0) / 60.0
    log.info(f"  Elapsed    = {elapsed * 60:.3f} s  (instant)")

    if cache:
        cache.save(_PARAM, result)
    if dashboard:
        dashboard.mark_done(_PARAM)

    print_diffusion_barrier(result)
    return result
