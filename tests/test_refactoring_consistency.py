import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.digital_coherent_sim.channel import EDFA as top_level_edfa
from src.digital_coherent_sim.channels.channel import EDFA as subpackage_edfa
from src.digital_coherent_sim.utils import ensure_2d_complex, load_legacy_module


def test_top_level_channel_wrapper_matches_subpackage_behavior():
    t = np.linspace(0, 1e-9, 64)
    s_in = np.zeros((2, len(t)), dtype=np.complex128)
    s_in[0, :] = 1 + 1j
    s_in[1, :] = 0.5 - 0.25j

    np.random.seed(123)
    top_level_output = top_level_edfa(s_in, t, G_edfa_dB=6.0, NF_dB=0.0, lambda0=1550e-9)
    np.random.seed(123)
    subpackage_output = subpackage_edfa(s_in, t, G_edfa_dB=6.0, NF_dB=0.0, lambda0=1550e-9)

    assert top_level_output.shape == subpackage_output.shape
    assert np.allclose(top_level_output[0, :], subpackage_output[0, :])
    assert np.allclose(top_level_output[1, :], subpackage_output[1, :])


def test_ensure_2d_complex_expands_vector_to_two_polarizations():
    vector = np.array([1 + 0j, 2 + 1j])

    shaped = ensure_2d_complex(vector)

    assert shaped.shape == (2, 2)
    assert np.allclose(shaped[0, :], vector)


def test_load_legacy_module_uses_caching():
    first = load_legacy_module("EDFA")
    second = load_legacy_module("EDFA")

    assert first is second
