#!/usr/bin/env python3
"""
watch_neb.py  —  Live CI-NEB terminal dashboard
================================================
Usage:
  python watch_neb.py results/UiO-66/neb
  python watch_neb.py results/UiO-66/neb/neb_band.out
"""
import re, sys, time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mof_screen.parsers import parse_neb_progress

# ── ANSI codes ──────────────────────────────────────────────────────────────
RS = "\033[0m";  BO = "\033[1m";  CY = "\033[96m"
GR = "\033[92m"; YL = "\033[93m"; DM = "\033[2m"

_ANSI_RE = re.compile(r"\033\[[0-9;]*m")

def _vlen(s: str) -> int:
    """Visual length of a string — strips ANSI codes before measuring."""
    return len(_ANSI_RE.sub("", s))

def _pad(s: str, width: int, fill: str = " ") -> str:
    """Right-pad string s to visual width `width`, accounting for ANSI codes."""
    return s + fill * max(0, width - _vlen(s))

def _bar(pct: float, width: int = 40) -> str:
    filled = int(width * pct / 100)
    return GR + "█" * filled + DM + "░" * (width - filled) + RS

def _fmt_t(secs: float) -> str:
    h, rem = divmod(int(secs), 3600)
    m, s   = divmod(rem, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"

def _clear(n: int) -> None:
    if n > 0:
        sys.stdout.write(f"\033[{n}A\033[J")

def _row(content: str, W: int) -> str:
    """Wrap content in box sides padded to visual width W interior."""
    return "║ " + _pad(content, W) + " ║"

def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <neb_dir_or_out_file>"); sys.exit(1)

    t = Path(sys.argv[1])
    out_file = (t / "neb_band.out") if t.is_dir() else t
    if not out_file.exists():
        print(f"Error: {out_file} not found"); sys.exit(1)

    W = 66          # visual width of box interior (between ║ borders)
    t0 = time.time()
    prev = 0

    try:
        while True:
            d        = parse_neb_progress(str(out_file))
            step     = d["band_step"]
            scf      = d["scf_steps"]
            barrier  = d["barrier_ev"]
            spread   = d["max_force"]
            n_rep    = d["n_replicas"]
            conv     = d["converged"]
            elapsed  = time.time() - t0
            ts       = datetime.now().strftime("%H:%M:%S")

            # ── build rows ────────────────────────────────────────────
            rows = []
            rows.append("╔" + "═" * (W + 2) + "╗")
            rows.append(_row(f"{BO}CI-NEB LIVE MONITOR{RS}  │  {out_file.parent.name}/neb_band.out", W))
            rows.append("╠" + "─" * (W + 2) + "╣")

            # step / scf / timestamp
            rows.append(_row(
                f"{CY}BAND steps :{RS} {BO}{step:>4d}{RS}"
                f"   {DM}│{RS}  SCF last: {scf:>3d}"
                f"   {DM}│{RS}  {DM}{ts}{RS}", W))

            # barrier
            b_str = f"{BO}{YL}{barrier:7.4f} eV{RS}" if barrier is not None else f"{DM}not yet available{RS}"
            rows.append(_row(f"{CY}Barrier     :{RS}  {b_str}", W))

            # geometry spread
            sp_str = f"{spread:6.3f} Å  {DM}↓ decreases as NEB converges{RS}" if spread is not None else f"{DM}not available{RS}"
            rows.append(_row(f"{CY}Geom spread :{RS}  {sp_str}", W))

            # replicas + elapsed
            rows.append(_row(f"{CY}Replicas    :{RS}  {n_rep or '?'}   {DM}│{RS}  Wall time: {_fmt_t(elapsed)}", W))

            rows.append("╠" + "─" * (W + 2) + "╣")

            # progress bar
            pct = 100.0 if conv else min(99.0, float(step))
            bar_row = f"{_bar(pct, W - 8)}  {pct:4.0f}%"
            rows.append(_row(bar_row, W))

            # status
            status = f"{GR}{BO}✓  CONVERGED{RS}" if conv else f"{CY}⟳  RUNNING …{RS}"
            rows.append(_row(status, W))
            rows.append("╚" + "═" * (W + 2) + "╝")

            # ── redraw ────────────────────────────────────────────────
            _clear(prev)
            out = "\n".join(rows) + "\n"
            sys.stdout.write(out)
            sys.stdout.flush()
            prev = len(rows) + 1

            if conv:
                print(f"\n{GR}{BO}NEB converged!{RS}  Final barrier: {barrier:.4f} eV")
                break
            time.sleep(2.0)

    except KeyboardInterrupt:
        print(f"\n{DM}Stopped.{RS}")

if __name__ == "__main__":
    main()
