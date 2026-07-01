import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.digital_coherent_sim import DPQAMReceiver, DPQAMTransmitter, EDFA, OpticalFiber, spectrum_analysis


def test_public_api_exposes_expected_symbols():
    assert DPQAMTransmitter is not None
    assert DPQAMReceiver is not None
    assert EDFA is not None
    assert OpticalFiber is not None
    assert spectrum_analysis is not None


def test_transmitter_receiver_wrappers_preserve_call_shape():
    tx = DPQAMTransmitter({"P_laser_TX": 0.01})
    receiver = DPQAMReceiver()

    assert hasattr(tx, "run")
    assert hasattr(receiver, "run")
    assert callable(tx.run)
    assert callable(receiver.run)


def test_channel_wrapper_accepts_same_inputs_as_before():
    t = np.linspace(0, 1e-9, 64)
    s_in = np.zeros((2, len(t)), dtype=np.complex128)
    s_in[0, :] = 1 + 1j
    s_in[1, :] = 0.5 - 0.25j

    out = EDFA(s_in, t, G_edfa_dB=6.0, NF_dB=0.0, lambda0=1550e-9)

    assert out.shape == s_in.shape
