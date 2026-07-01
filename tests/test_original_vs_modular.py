import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.EDFA import EDFA as legacy_edfa
from src.digital_coherent_sim.channels.channel import EDFA as modular_edfa


def test_edfa_original_and_modular_are_equivalent_for_same_inputs():
    np.random.seed(123)
    t = np.linspace(0, 1e-9, 64)
    s_in = np.zeros((2, len(t)), dtype=np.complex128)
    s_in[0, :] = 1 + 1j
    s_in[1, :] = 0.5 - 0.25j

    np.random.seed(123)
    legacy_output = legacy_edfa(s_in, t, 6.0, 0.0, 1550e-9)
    np.random.seed(123)
    modular_output = modular_edfa(s_in, t, G_edfa_dB=6.0, NF_dB=0.0, lambda0=1550e-9)

    assert legacy_output.shape == modular_output.shape
    assert legacy_output.dtype == modular_output.dtype
    assert np.allclose(legacy_output[0, :], modular_output[0, :], rtol=1e-12, atol=1e-12)
    assert np.allclose(legacy_output[1, :], modular_output[1, :], rtol=1e-12, atol=1e-12)
