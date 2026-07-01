import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.digital_coherent_sim.channel as channel_module
import src.digital_coherent_sim.graphics as graphics_module

EDFA = channel_module.EDFA
OpticalFiber = channel_module.OpticalFiber
spectrum_analysis = graphics_module.spectrum_analysis


def test_edfa_scales_signal_power():
    t = np.linspace(0, 1e-9, 64)
    s_in = np.zeros((2, len(t)), dtype=np.complex128)
    s_in[0, :] = 1 + 1j
    s_in[1, :] = 0.5 - 0.25j

    s_out = EDFA(s_in, t, G_edfa_dB=6.0, NF_dB=0.0, lambda0=1550e-9)

    assert s_out.shape == s_in.shape
    assert np.mean(np.abs(s_out[0, :]) ** 2) > np.mean(np.abs(s_in[0, :]) ** 2)


def test_fiber_returns_two_polarizations():
    t = np.linspace(0, 1e-9, 64)
    A = np.zeros((2, len(t)), dtype=np.complex128)
    A[0, :] = 1 + 0j
    A[1, :] = 0.5 + 0.25j

    out = OpticalFiber().propagate(A, t)

    assert out.shape == A.shape
    assert out[0].shape == (len(t),)


def test_spectrum_analysis_runs_without_error():
    t = np.linspace(0, 1e-9, 256)
    s = np.exp(1j * 2 * np.pi * 50e6 * t)

    spectrum_analysis(s, t)
