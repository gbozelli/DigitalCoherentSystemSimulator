from __future__ import annotations

import numpy as np


class EDFAAmplifier:
    def __init__(self, config: dict | None = None) -> None:
        self.config = config or {}

    def apply(self, s_in, t, *, G_edfa_dB=20.0, NF_dB=5.0, lambda0=1550e-9):
        h = 6.626e-34
        c = 3e8
        f0 = c / lambda0
        ts = t[1] - t[0]
        fs = 1 / ts
        BW_sim = fs

        G_edfa = 10 ** (G_edfa_dB / 10)
        sx = s_in[0]
        sy = s_in[1]
        N = len(sx)

        sx = np.sqrt(G_edfa) * sx
        sy = np.sqrt(G_edfa) * sy

        F_n = 10 ** (NF_dB / 10)
        n_sp = F_n / 2
        S_ASE = n_sp * h * f0 * (G_edfa - 1)
        P_ASE = S_ASE * BW_sim

        ase_scale = np.sqrt(P_ASE) / 2
        s_ASEx = ase_scale * np.random.rand(N) + 1j * ase_scale * np.random.rand(N)
        s_ASEy = ase_scale * np.random.rand(N) + 1j * ase_scale * np.random.rand(N)
        s_outx = sx + s_ASEx
        s_outy = sy + s_ASEy

        s = np.zeros((2, len(t)), dtype=np.complex128)
        s[0, :] = s_outx
        s[1, :] = s_outy
        return s


def EDFA(s_in, t, *, G_edfa_dB=20.0, NF_dB=5.0, lambda0=1550e-9):
    amplifier = EDFAAmplifier()
    return amplifier.apply(s_in, t, G_edfa_dB=G_edfa_dB, NF_dB=NF_dB, lambda0=lambda0)
