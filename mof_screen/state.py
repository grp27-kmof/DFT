"""
mof_screen/state.py
===================
Production-grade checkpoint / resume system.

Every parameter writes its result to  <results_dir>/<param>.json
immediately when it finishes.  Subsequent runs (or --only invocations)
read from this cache instead of re-running expensive DFT calculations.

Usage
-----
    cache = StateCache(results_dir)
    cache.save("adsorption", ads_result)     # writes adsorption.json
    data  = cache.load("adsorption")         # dict | None
    cache.is_done("adsorption")              # True / False
    cache.clear("adsorption")               # force re-run
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("mof_dft.state")


class StateCache:
    """
    Per-CIF JSON cache.  One JSON file per parameter in results_dir/.
    Thread-safe for single-process use (no concurrent writers).
    """

    # Recognised parameter names
    ALL_PARAMS: List[str] = [
        "adsorption",
        "formation",
        "volume",
        "ocv",
        "diffusion",
        "dos",
        "ionic_conductivity",
    ]

    def __init__(self, results_dir: Path) -> None:
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(self, param: str, result: Any) -> None:
        """
        Persist a result dataclass (or plain dict) as <param>.json.
        Adds a '_saved_at' timestamp for auditability.
        """
        if hasattr(result, "__dataclass_fields__"):
            data: Dict[str, Any] = asdict(result)   # type: ignore[arg-type]
        elif isinstance(result, dict):
            data = result
        else:
            data = {"value": result}

        data["_saved_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        path = self._path(param)
        path.write_text(json.dumps(data, indent=2, default=str))
        log.debug(f"  [cache] Saved {param} → {path.name}")

    def load(self, param: str) -> Optional[Dict[str, Any]]:
        """
        Load a cached result dict.  Returns None if the file does not
        exist or is corrupted.
        """
        path = self._path(param)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            log.debug(f"  [cache] Loaded {param} from {path.name}")
            return data
        except (json.JSONDecodeError, OSError) as exc:
            log.warning(f"  [cache] Corrupted cache for {param}: {exc}")
            return None

    def is_done(self, param: str) -> bool:
        """Return True if a valid cache file exists for this param."""
        path = self._path(param)
        if not path.exists():
            return False
        try:
            json.loads(path.read_text())
            return True
        except (json.JSONDecodeError, OSError):
            return False

    def clear(self, param: str) -> None:
        """Delete the cache file for a single param (force re-run)."""
        path = self._path(param)
        if path.exists():
            path.unlink()
            log.info(f"  [cache] Cleared {param}.json — will re-run.")

    def clear_all(self) -> None:
        """Delete ALL parameter cache files in the results directory."""
        for param in self.ALL_PARAMS:
            self.clear(param)

    def status_summary(self) -> Dict[str, bool]:
        """Return {param: is_done} for all parameters."""
        return {p: self.is_done(p) for p in self.ALL_PARAMS}

    def print_status(self) -> None:
        """Print a compact table of cache status to stdout."""
        print("\n  Cache status:")
        for param, done in self.status_summary().items():
            icon = "✓" if done else "○"
            state = "CACHED" if done else "pending"
            print(f"    {icon}  {param:<22s} {state}")

    # ------------------------------------------------------------------
    # Dependency helpers
    # ------------------------------------------------------------------

    # Which params need which prior param's output
    _DEPENDENCIES: Dict[str, List[str]] = {
        "adsorption":           [],
        "formation":            ["adsorption"],   # needs E(MOF) from adsorption.out
        "volume":               ["adsorption"],   # reads adsorption.out cell info
        "ocv":                  ["adsorption"],   # needs E_ads
        "ionic_conductivity":   ["adsorption"],   # needs E_ads; barrier is optional
        "diffusion":            [],               # standalone — reads NEB file
        "dos":                  [],               # standalone — reads .pdos files
    }

    def missing_dependencies(self, param: str) -> List[str]:
        """
        Return list of dependency param names that are NOT yet cached.
        Empty list means all dependencies satisfied.
        """
        deps = self._DEPENDENCIES.get(param, [])
        return [d for d in deps if not self.is_done(d)]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _path(self, param: str) -> Path:
        return self.results_dir / f"{param}.json"
