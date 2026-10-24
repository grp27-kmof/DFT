"""
mof_screen/__init__.py
========================
Public API for the mof_screen package.

Typical use
-----------
    from mof_screen import run_screening
    results = run_screening("MOF1.cif", ion_symbol="K", num_cores=16)

    # Standalone parameter
    from mof_screen.calc_ocv import calc_ocv
    ocv = calc_ocv(e_ads_ev=-0.75, n_ions=1)
"""

from __future__ import annotations

__version__ = "1.0.0"
__author__  = "MOF-DFT Team"

from mof_screen.workflow import run_screening
from mof_screen.dataclasses_ import (
    AllResults,
    AdsorptionResult,
    FormationResult,
    VolumeResult,
    OCVResult,
    DiffusionBarrierResult,
    DOSResult,
    IonicConductivityResult,
)

__all__ = [
    "run_screening",
    "AllResults",
    "AdsorptionResult",
    "FormationResult",
    "VolumeResult",
    "OCVResult",
    "DiffusionBarrierResult",
    "DOSResult",
    "IonicConductivityResult",
]
