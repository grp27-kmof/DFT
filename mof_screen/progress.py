"""
mof_screen/progress.py
======================
Live per-parameter progress dashboard for the MOF DFT screening pipeline.

Shows — for every parameter — in real time:
  • Status:   PENDING / RUNNING / DONE / FAILED / SKIPPED / CACHED
  • Progress: animated bar + percentage (0–100%)
  • Elapsed:  wall-clock time since the parameter started
  • ETA:      estimated time remaining based on DFT step counters

The dashboard redraws itself in-place every second using ANSI escape codes
so the terminal is not flooded with scrolling output.

Usage (from workflow.py)
------------------------
    dash = ProgressDashboard(params=["adsorption","formation",...])
    dash.start()

    dash.mark_running("adsorption")
    dash.update("adsorption", pct=45, phase="GEO step 14/300 | SCF 87/300")
    dash.mark_done("adsorption")

    dash.stop()   # freezes the dashboard and prints final summary
"""

from __future__ import annotations

import sys
import threading
import time
from typing import Dict, List, Optional

from mof_screen.constants import EXPECTED_TIMES_MIN
from mof_screen.dataclasses_ import CalcStatus, ParameterProgress


# ANSI codes
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_CYAN   = "\033[96m"
_DIM    = "\033[2m"
_BLUE   = "\033[94m"

_BAR_WIDTH  = 16   # characters for the progress bar
_PANEL_COLS = 78   # total panel width


def _bar(pct: float, width: int = _BAR_WIDTH) -> str:
    """Render a Unicode progress bar for the given percentage (0–100)."""
    filled = int(width * pct / 100)
    return "█" * filled + "░" * (width - filled)


def _fmt_time(seconds: Optional[float]) -> str:
    """Format seconds as 'Xm Ys' or '--:--' if unknown."""
    if seconds is None or seconds < 0:
        return "  --:--"
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:3d}m{s:02d}s"


def _status_color(status: CalcStatus) -> str:
    return {
        CalcStatus.PENDING:  _DIM,
        CalcStatus.RUNNING:  _CYAN + _BOLD,
        CalcStatus.DONE:     _GREEN,
        CalcStatus.CACHED:   _GREEN + _DIM,
        CalcStatus.FAILED:   _RED + _BOLD,
        CalcStatus.SKIPPED:  _DIM,
    }.get(status, _RESET)


class ProgressDashboard:
    """
    Thread-safe live dashboard that redraws itself every second.

    The dashboard is drawn below any other log output using ANSI cursor
    movement.  It gracefully degrades in non-TTY environments (HPC batch
    jobs) by falling back to periodic log lines.
    """

    _REDRAW_INTERVAL    = 1.0     # seconds — TTY refresh rate
    _LOG_INTERVAL       = 300.0   # seconds — non-TTY log rate (every 5 min)

    def __init__(
        self,
        params: List[str],
        mof_name: str = "",
        ion_symbol: str = "K",
    ) -> None:
        self._params: List[str]  = params
        self._mof_name: str      = mof_name
        self._ion_symbol: str    = ion_symbol

        # Per-parameter progress objects
        self._prog: Dict[str, ParameterProgress] = {
            p: ParameterProgress(
                name      = p,
                status    = CalcStatus.PENDING,
                expected_s= EXPECTED_TIMES_MIN.get(p, 1.0) * 60,
            )
            for p in params
        }

        self._lock      = threading.Lock()
        self._stop      = threading.Event()
        self._thread    = threading.Thread(target=self._loop, daemon=True)
        self._t0        = time.time()
        self._is_tty    = sys.stdout.isatty()
        self._n_lines   = 0      # lines drawn in last frame (for clearing)
        self._last_log  = 0.0   # last time a non-TTY log line was emitted

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._t0 = time.time()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
        self._clear_panel()
        self._draw(final=True)

    # ------------------------------------------------------------------
    # Parameter state mutators (thread-safe)
    # ------------------------------------------------------------------

    def mark_running(self, param: str) -> None:
        with self._lock:
            p = self._prog.get(param)
            if p:
                p.status    = CalcStatus.RUNNING
                p.elapsed_s = 0.0
                p._t_start  = time.time()   # type: ignore[attr-defined]

    def mark_done(self, param: str) -> None:
        with self._lock:
            p = self._prog.get(param)
            if p:
                p.status   = CalcStatus.DONE
                p.pct_done = 100.0
                if hasattr(p, "_t_start"):
                    p.elapsed_s = time.time() - p._t_start   # type: ignore[attr-defined]

    def mark_failed(self, param: str) -> None:
        with self._lock:
            p = self._prog.get(param)
            if p:
                p.status = CalcStatus.FAILED

    def mark_skipped(self, param: str) -> None:
        with self._lock:
            p = self._prog.get(param)
            if p:
                p.status = CalcStatus.SKIPPED

    def mark_cached(self, param: str) -> None:
        with self._lock:
            p = self._prog.get(param)
            if p:
                p.status   = CalcStatus.CACHED
                p.pct_done = 100.0

    def update(
        self,
        param: str,
        *,
        pct: Optional[float]    = None,
        phase: str              = "",
        scf_step: int           = 0,
        geo_step: int           = 0,
        max_scf: int            = 300,
        max_geo: int            = 300,
    ) -> None:
        """
        Update progress metrics for a running parameter.

        pct     — override percentage (0–100). If None, auto-compute
                  from geo_step / max_geo (or scf_step / max_scf).
        phase   — human-readable description, e.g. "GEO step 14/300"
        """
        with self._lock:
            p = self._prog.get(param)
            if p is None or p.status != CalcStatus.RUNNING:
                return

            p.scf_step = scf_step
            p.geo_step = geo_step
            p.phase    = phase

            if pct is not None:
                p.pct_done = max(0.0, min(100.0, pct))
            elif geo_step > 0 and max_geo > 0:
                # Nested: each GEO step contains SCF cycles
                geo_pct = geo_step / max_geo * 100.0
                scf_pct = (scf_step / max(max_scf, 1)) / max_geo * 100.0
                p.pct_done = min(geo_pct + scf_pct, 99.0)
            elif scf_step > 0 and max_scf > 0:
                p.pct_done = min(scf_step / max_scf * 100.0, 99.0)

            # Elapsed and ETA
            if hasattr(p, "_t_start"):
                p.elapsed_s = time.time() - p._t_start  # type: ignore[attr-defined]
                if p.pct_done > 2.0:
                    total_est  = p.elapsed_s / (p.pct_done / 100.0)
                    p.eta_s    = max(0.0, total_est - p.elapsed_s)
                elif p.expected_s:
                    p.eta_s = p.expected_s

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._refresh_elapsed()
            self._redraw()
            time.sleep(self._REDRAW_INTERVAL)

    def _refresh_elapsed(self) -> None:
        """Update elapsed time for all RUNNING parameters."""
        with self._lock:
            for p in self._prog.values():
                if p.status == CalcStatus.RUNNING and hasattr(p, "_t_start"):
                    p.elapsed_s = time.time() - p._t_start   # type: ignore[attr-defined]

    def _redraw(self) -> None:
        if self._is_tty:
            self._clear_panel()
            self._draw(final=False)
        else:
            # Non-TTY (log file): emit one compact status line every 5 minutes
            now = time.time()
            if now - self._last_log < self._LOG_INTERVAL:
                return
            self._last_log = now
            elapsed_total = _fmt_time(now - self._t0).strip()
            with self._lock:
                running = [p for p in self._prog.values() if p.status == CalcStatus.RUNNING]
                if running:
                    for p in running:
                        phase = p.phase or "running"
                        sys.stdout.write(
                            f"  [STATUS +{elapsed_total}] {p.name}: "
                            f"{phase}  elapsed={_fmt_time(p.elapsed_s).strip()}\n"
                        )
                else:
                    sys.stdout.write(
                        f"  [STATUS +{elapsed_total}] No active calculations.\n"
                    )
            sys.stdout.flush()

    def _clear_panel(self) -> None:
        """Move cursor up n_lines to overwrite previous frame."""
        if self._is_tty and self._n_lines > 0:
            sys.stdout.write(f"\033[{self._n_lines}A\033[J")
            sys.stdout.flush()
        self._n_lines = 0

    def _draw(self, *, final: bool = False) -> None:
        """Render the full dashboard panel."""
        overall_elapsed = time.time() - self._t0
        lines: List[str] = []

        W = _PANEL_COLS
        sep = "─" * W

        # ── Header ──────────────────────────────────────────────────
        header = (
            f" MOF K⁺-ANODE DFT PROGRESS │ "
            f"{self._mof_name}  │  Ion: {self._ion_symbol}  │  "
            f"Elapsed: {_fmt_time(overall_elapsed).strip()}"
        )
        lines.append("╔" + "═" * W + "╗")
        lines.append(f"║  {_BOLD}{header[:W - 2]:<{W - 2}}{_RESET}║")
        lines.append("╠" + "═" * W + "╣")

        # ── Column headers ───────────────────────────────────────────
        col_hdr = (
            f"  {'Parameter':<22s}"
            f"{'Status':<10s}"
            f"{'Progress':^20s}"
            f"{'Elapsed':>8s}"
            f"{'ETA':>8s}"
        )
        lines.append(f"║{col_hdr:<{W}}║")
        lines.append("╠" + "─" * W + "╣")

        # ── Per-parameter rows ───────────────────────────────────────
        with self._lock:
            prog_snapshot = list(self._prog.values())

        total_weight   = 0.0
        weighted_done  = 0.0

        for p in prog_snapshot:
            sc = _status_color(p.status)

            if p.status in (CalcStatus.DONE, CalcStatus.CACHED):
                bar_str = _GREEN + "█" * _BAR_WIDTH + _RESET
                pct_str = "100%"
            elif p.status == CalcStatus.RUNNING:
                bar_str = _CYAN + _bar(p.pct_done) + _RESET
                pct_str = f"{p.pct_done:4.1f}%"
            elif p.status == CalcStatus.FAILED:
                bar_str = _RED + "✗" * _BAR_WIDTH + _RESET
                pct_str = "ERR "
            elif p.status == CalcStatus.SKIPPED:
                bar_str = _DIM + "─" * _BAR_WIDTH + _RESET
                pct_str = " -- "
            else:
                bar_str = _DIM + "░" * _BAR_WIDTH + _RESET
                pct_str = "  0%"

            elapsed_str = _fmt_time(p.elapsed_s) if p.elapsed_s > 0 else "    --"
            eta_str     = _fmt_time(p.eta_s)     if p.eta_s is not None else "    --"
            status_str  = f"{sc}{p.status.value:<9s}{_RESET}"

            row = (
                f"  {p.name:<22s}"
                f"{status_str} "
                f"{bar_str} {pct_str:<5s}"
                f"{elapsed_str}"
                f"{eta_str}"
            )
            lines.append(f"║{row}║")

            # Weight for overall %
            w = EXPECTED_TIMES_MIN.get(p.name, 1.0)
            total_weight  += w
            if p.status in (CalcStatus.DONE, CalcStatus.CACHED):
                weighted_done += w
            elif p.status == CalcStatus.RUNNING:
                weighted_done += w * p.pct_done / 100.0

        # ── Overall progress ─────────────────────────────────────────
        overall_pct = (weighted_done / total_weight * 100.0) if total_weight > 0 else 0.0
        lines.append("╠" + "─" * W + "╣")

        total_eta: Optional[float] = None
        if overall_pct > 1.0:
            total_secs = overall_elapsed / (overall_pct / 100.0)
            total_eta  = max(0.0, total_secs - overall_elapsed)

        overall_bar  = _bar(overall_pct, width=24)
        overall_line = (
            f"  {'Overall':<22s}"
            f"{'':10s}"
            f"{_BOLD}{_BLUE}{overall_bar}{_RESET} "
            f"{_BOLD}{overall_pct:4.1f}%{_RESET}"
            f"{_fmt_time(overall_elapsed)}"
            f"{_fmt_time(total_eta)}"
        )
        lines.append(f"║{overall_line}║")
        lines.append("╚" + "═" * W + "╝")

        output = "\n".join(lines) + "\n"
        sys.stdout.write(output)
        sys.stdout.flush()
        self._n_lines = len(lines) + 1

    # ------------------------------------------------------------------
    # Final summary table (printed after stop())
    # ------------------------------------------------------------------

    def print_summary_table(self) -> None:
        """Print a compact summary of all parameter outcomes."""
        print("\n" + "═" * _PANEL_COLS)
        print(f"  PARAMETER PROGRESS SUMMARY  │  MOF: {self._mof_name}")
        print("═" * _PANEL_COLS)
        print(
            f"  {'Parameter':<22s}"
            f"{'Status':<12s}"
            f"{'%':>6s}"
            f"{'Elapsed':>10s}"
            f"{'ETA':>10s}"
        )
        print("  " + "─" * (_PANEL_COLS - 2))
        with self._lock:
            for p in self._prog.values():
                pct = f"{p.pct_done:.0f}%" if p.pct_done > 0 else "--"
                print(
                    f"  {p.name:<22s}"
                    f"{p.status.value:<12s}"
                    f"{pct:>6s}"
                    f"{_fmt_time(p.elapsed_s):>10s}"
                    f"{_fmt_time(p.eta_s):>10s}"
                )
        print("═" * _PANEL_COLS)
