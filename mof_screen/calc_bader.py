"""
mof_screen/calc_bader.py
=========================
PARAMETER: Bader Charge Analysis  (ionic charge transfer)

Calculation type : Post-processing — parses CP2K CUBE files with the
                   Henkelman group's Bader code (bader), OR falls back to
                   a Hirshfeld-charge estimate from the CP2K output file
                   if no Bader executable is found.
Standalone       : YES — needs the MOF+ion output directory and cube files
Cache file       : <results_dir>/bader.json

Physical meaning
----------------
Bader charge analysis partitions the electron density into atomic basins
(Bader, 1990) and assigns a net charge to each atom.  For anode screening:

    Δq(ion)   — charge transferred FROM the ion to the framework.
                 Ideal range: 0.6–1.0 e for monovalent ions (K⁺, Na⁺, Li⁺).
    Δq(frame) — compensating charge on the nearest framework metal node.

    If Δq(ion) ≈ 1  → full electron transfer (ionic bonding, good storage)
    If Δq(ion) ≈ 0  → metallic / weak binding (poor anode)

Fallback (Hirshfeld)
---------------------
CP2K can write Hirshfeld charges to the output file with
``&PRINT → &MULLIKEN / &HIRSHFELD`` without requiring extra CUBE files.
When cube files are absent (--no-cube or HPC scratch limit), the module
parses Hirshfeld charges from the .out file and flags the result accordingly.

Usage (standalone)
------------------
    from mof_screen.calc_bader import calc_bader_charges
    result = calc_bader_charges(
        out_dir=out_dir,
        ion_symbol=\"K\",
        n_ions=1,
        dashboard=dashboard,
        cache=cache,
    )
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

from mof_screen.dataclasses_ import BaderResult
from mof_screen.printer import print_bader
from mof_screen.progress import ProgressDashboard
from mof_screen.state import StateCache

log = logging.getLogger("mof_dft.bader")

_PARAM = "bader"

# ── Regex patterns ───────────────────────────────────────────────────────────
_RE_HIRSHFELD = re.compile(
    r"Hirshfeld Charges\s*\n"
    r"\s*#\s+Element\s+Ref Charge\s+Pop\s+Net charge\s*\n"
    r"((?:\s*\d+\s+\w+\s+[\d.+-]+\s+[\d.+-]+\s+[+-]?[\d.]+\s*\n)+)",
    re.MULTILINE,
)
_RE_HIRSH_ROW = re.compile(
    r"^\s*(\d+)\s+(\w+)\s+([\d.+-]+)\s+([\d.+-]+)\s+([+-]?[\d.]+)\s*$",
    re.MULTILINE,
)
_RE_MULLIKEN = re.compile(
    r"Mulliken Population Analysis\s*\n"
    r"\s*#\s+Atom\s+El\s+Charge\s*\n"
    r"((?:\s*\d+\s+\d+\s+\w+\s+[+-]?[\d.]+\s*\n)+)",
    re.MULTILINE,
)
_RE_MULL_ROW  = re.compile(
    r"^\s*(\d+)\s+(\d+)\s+(\w+)\s+([+-]?[\d.]+)\s*$", re.MULTILINE
)
_RE_ACF_LINE = re.compile(
    r"^\s*(\d+)\s+([\d.+-]+)\s+([\d.+-]+)\s+([\d.+-]+)\s+([\d.]+)\s+([\d.+-]+)\s+([\d.]+)\s*$"
)


def _find_bader_exe() -> Optional[str]:
    """Return path to Henkelman bader executable, or None if not found."""
    for name in ("bader", "bader.x", "bader_charge"):
        path = subprocess.run(
            ["which", name], capture_output=True, text=True
        ).stdout.strip()
        if path:
            return path
    return None


def _run_bader(cube_chg: Path, cube_ref: Path, work_dir: Path) -> Optional[Path]:
    """
    Run the Henkelman bader code on CP2K CUBE files.

    Returns the path to ACF.dat (atomic charges file) if successful, else None.
    """
    bader_exe = _find_bader_exe()
    if bader_exe is None:
        log.warning("  bader executable not found — falling back to Hirshfeld charges.")
        return None

    acf = work_dir / "ACF.dat"
    cmd = [bader_exe, str(cube_chg), "-ref", str(cube_ref)]
    log.info(f"  Running: {' '.join(cmd)}")
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, cwd=str(work_dir), timeout=300
        )
        if proc.returncode != 0:
            log.warning(f"  bader returned non-zero exit: {proc.returncode}")
            log.warning(f"  stderr: {proc.stderr[:500]}")
            return None
        if acf.exists():
            return acf
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        log.warning(f"  bader failed: {exc}")
    return None


def _parse_acf(acf_path: Path) -> List[Dict]:
    """Parse Henkelman ACF.dat → list of {idx, x, y, z, charge, min_dist, vol}."""
    rows = []
    with open(acf_path) as fh:
        for line in fh:
            m = _RE_ACF_LINE.match(line)
            if m:
                rows.append({
                    "idx":      int(m.group(1)),
                    "x":        float(m.group(2)),
                    "y":        float(m.group(3)),
                    "z":        float(m.group(4)),
                    "charge":   float(m.group(5)),
                    "min_dist": float(m.group(6)),
                    "vol":      float(m.group(7)),
                })
    return rows


def _parse_hirshfeld(out_text: str) -> Optional[List[Dict]]:
    """Extract Hirshfeld net charges from a CP2K output file."""
    m = _RE_HIRSHFELD.search(out_text)
    if not m:
        return None
    rows = []
    for rm in _RE_HIRSH_ROW.finditer(m.group(1)):
        rows.append({
            "idx":    int(rm.group(1)),
            "elem":   rm.group(2),
            "charge": float(rm.group(5)),   # net charge
        })
    return rows if rows else None


def _parse_mulliken(out_text: str) -> Optional[List[Dict]]:
    """Extract Mulliken charges from a CP2K output file (fallback of fallback)."""
    m = _RE_MULLIKEN.search(out_text)
    if not m:
        return None
    rows = []
    for rm in _RE_MULL_ROW.finditer(m.group(1)):
        rows.append({
            "idx":    int(rm.group(1)),
            "elem":   rm.group(3),
            "charge": float(rm.group(4)),
        })
    return rows if rows else None


def calc_bader_charges(
    out_dir:    Path,
    ion_symbol: str                        = "K",
    n_ions:     int                        = 1,
    dashboard:  Optional[ProgressDashboard] = None,
    cache:      Optional[StateCache]        = None,
) -> BaderResult:
    """
    Compute or parse Bader/Hirshfeld/Mulliken charges for the MOF+ion system.

    Priority order:
        1. Henkelman bader code  (most accurate — requires CUBE files)
        2. CP2K Hirshfeld charges (from adsorption.out)
        3. CP2K Mulliken charges  (from adsorption.out — least accurate)
        4. NOT COMPUTED           (if no charge data found)

    Parameters
    ----------
    out_dir    : results/<MOF_stem>/ directory containing adsorption.out
    ion_symbol : ion element symbol, e.g. 'K'
    n_ions     : number of inserted ions
    dashboard  : live progress dashboard (optional)
    cache      : per-CIF state cache (optional)
    """
    log.info("")
    log.info("━" * 64)
    log.info("  BADER CHARGE ANALYSIS  (charge partitioning)")
    log.info("━" * 64)

    result = BaderResult(ion_symbol=ion_symbol, n_ions=n_ions)

    # ---- Cache hit ----
    if cache and cache.is_done(_PARAM):
        log.info("  ✓ Loading bader from cache.")
        data = cache.load(_PARAM) or {}
        result.ion_symbol          = data.get("ion_symbol", ion_symbol)
        result.n_ions              = data.get("n_ions", n_ions)
        result.ion_charge_e        = data.get("ion_charge_e")
        result.ion_charge_transfer = data.get("ion_charge_transfer")
        result.method              = data.get("method", "UNKNOWN")
        result.parsed              = data.get("parsed", False)
        result.all_charges         = data.get("all_charges", [])
        result.framework_avg_charge= data.get("framework_avg_charge")
        if dashboard:
            dashboard.mark_cached(_PARAM)
        print_bader(result)
        return result

    if dashboard:
        dashboard.mark_running(_PARAM)

    t0 = time.time()
    out_file = out_dir / "adsorption.out"

    # ── 1. Try Henkelman bader with CUBE files ───────────────────────────────
    cube_chg = out_dir / "MOF_ION-ELECTRON_DENSITY-1_0.cube"
    cube_ref = out_dir / "MOF_ION-ELECTRON_DENSITY-1_0.cube"  # reference (bare density)

    # Look for any CUBE files that match CP2K naming convention
    cube_files = sorted(out_dir.glob("*.cube"))
    if len(cube_files) >= 2:
        cube_chg = cube_files[-1]   # latest = MOF+ion
        cube_ref = cube_files[-2]   # previous = bare MOF (if present)
        acf_path = _run_bader(cube_chg, cube_ref, out_dir)
        if acf_path:
            charges = _parse_acf(acf_path)
            _fill_result_from_charges(result, charges, ion_symbol, n_ions, "bader")
    elif len(cube_files) == 1:
        log.info("  Only one CUBE file found — need two (bare + ion) for Bader. Trying Hirshfeld.")

    # ── 2. Fallback: Hirshfeld charges from CP2K output ─────────────────────
    if not result.parsed and out_file.exists():
        out_text = out_file.read_text(errors="replace")
        rows = _parse_hirshfeld(out_text)
        if rows:
            _fill_result_from_charges(result, rows, ion_symbol, n_ions, "hirshfeld")
            log.info("  ✓ Hirshfeld charges found in adsorption.out")
        else:
            # ── 3. Fallback: Mulliken ────────────────────────────────────────
            rows = _parse_mulliken(out_text)
            if rows:
                _fill_result_from_charges(result, rows, ion_symbol, n_ions, "mulliken")
                log.info("  ✓ Mulliken charges found in adsorption.out (less accurate)")
            else:
                log.warning("  No Bader/Hirshfeld/Mulliken charges found.")
                log.warning("  Add &PRINT → &HIRSHFELD T &END to CP2K input to enable.")
    elif not result.parsed:
        log.warning(f"  adsorption.out not found: {out_file}")

    # ── Summary logs ─────────────────────────────────────────────────────────
    if result.parsed:
        log.info(f"  Method                : {result.method}")
        log.info(f"  Ion ({ion_symbol}) charge     : {result.ion_charge_e:.4f} e")
        log.info(f"  Charge transfer Δq    : {result.ion_charge_transfer:.4f} e")
        if result.framework_avg_charge is not None:
            log.info(f"  Framework avg charge  : {result.framework_avg_charge:.4f} e")
    else:
        result.method = "NOT COMPUTED"

    elapsed = (time.time() - t0) / 60.0
    log.info(f"  Elapsed               : {elapsed * 60:.2f} s")
    result.elapsed_min = elapsed

    if cache:
        cache.save(_PARAM, result)
    if dashboard:
        dashboard.mark_done(_PARAM) if result.parsed else dashboard.mark_failed(_PARAM)

    print_bader(result)
    return result


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------
def _fill_result_from_charges(
    result: BaderResult,
    charges: List[Dict],
    ion_symbol: str,
    n_ions: int,
    method: str,
) -> None:
    """Fill BaderResult fields from a list of {elem, charge} dicts."""
    if not charges:
        return

    result.method     = method
    result.all_charges = [
        {"elem": r.get("elem", ion_symbol), "charge": r["charge"]}
        for r in charges
    ]

    # Identify ion atoms (by element symbol) — take the last n_ions occurrences
    ion_atoms = [r for r in charges if r.get("elem", "").upper() == ion_symbol.upper()]
    if not ion_atoms:
        # For ACF.dat (no element labels), use the last n_ions rows as ion
        ion_atoms = charges[-n_ions:]

    ion_charge_total = sum(r["charge"] for r in ion_atoms[-n_ions:])
    result.ion_charge_e = ion_charge_total / n_ions   # per-ion average

    # Charge transfer = nominal charge (e.g. +1 for K) − actual Bader charge
    # Bader charge convention: atomic charge = Z_nuc − n_bader_electrons
    # For a fully ionic K+: Bader charge ≈ +1.  Transfer ≈ 1.0 e.
    result.ion_charge_transfer = result.ion_charge_e  # same in Bader convention

    # Framework atoms: everything that is NOT the ion
    frame_atoms = [r for r in charges if r.get("elem", "").upper() != ion_symbol.upper()]
    if frame_atoms:
        result.framework_avg_charge = (
            sum(r["charge"] for r in frame_atoms) / len(frame_atoms)
        )

    result.parsed = True
