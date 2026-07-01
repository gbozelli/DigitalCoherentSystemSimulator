"""Digital coherent system simulator package."""

from __future__ import annotations

import pathlib
import sys

from .channel import EDFA, OpticalFiber
from .graphics import spectrum_analysis
from .receiver import DPQAMReceiver
from .transmitter import DPQAMTransmitter

SRC_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

__all__ = [
    "DPQAMTransmitter",
    "DPQAMReceiver",
    "EDFA",
    "OpticalFiber",
    "spectrum_analysis",
]
