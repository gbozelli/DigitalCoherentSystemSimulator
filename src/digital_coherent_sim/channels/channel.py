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
        sx = np.sqrt(G_edfa) * s_in[0]
        sy = np.sqrt(G_edfa) * s_in[1]
        N = len(sx)

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


class OpticalFiber:
    def __init__(self, config: dict | None = None) -> None:
        self.config = config or {}

    def propagate(self, A, t, *, L=100e3, DeltaL=1e3, D=16e-6, lambda0=1550e-9,
                  alpha_dB=0.2e-3, n2=2.6e-20, Aeff=80e-12, SoP_rotation=None):
        c = 3e8
        pi = np.pi

        Nt = len(t)
        Deltat = t[1] - t[0]
        fs = 1 / Deltat
        f = np.arange(Nt) / (Nt / fs) - fs / 2
        Nz = int(L / DeltaL)
        z = np.linspace(0, L - DeltaL, Nz)
        alpha_Np = alpha_dB / 4.343
        k = 2 * pi / lambda0
        gamma = n2 * k / Aeff

        Ax = A[0, :]
        Ay = A[1, :]

        for _ in range(len(z)):
            Ax_f = np.fft.fftshift(np.fft.fft(Ax))
            Ay_f = np.fft.fftshift(np.fft.fft(Ay))
            Ax_f = Ax_f * np.exp(-alpha_Np / 2 * DeltaL)
            Ay_f = Ay_f * np.exp(-alpha_Np / 2 * DeltaL)
            dispersion_phase = (
                -1j * (D * lambda0**2 / (4 * np.pi * c) * DeltaL * ((2 * np.pi) * f) ** 2)
            )
            Ax_f *= np.exp(dispersion_phase)
            Ay_f *= np.exp(dispersion_phase)
            Ax = np.fft.ifft(np.fft.ifftshift(Ax_f))
            Ay = np.fft.ifft(np.fft.ifftshift(Ay_f))
            Ixy = (abs(Ax))**2 + (abs(Ay))**2
            Ax *= np.exp(1j * gamma * Ixy * DeltaL)
            Ay *= np.exp(1j * gamma * Ixy * DeltaL)
            if SoP_rotation:
                theta_rand = np.random.rand(1) * np.pi
                phi_rand = np.random.rand(1) * 2 * np.pi
                Ax_aux = Ax * np.exp(1j * phi_rand / 2)
                Ay_aux = Ay * np.exp(-1j * phi_rand / 2)
                Ax = np.cos(theta_rand) * Ax_aux + np.sin(theta_rand) * Ay_aux
                Ay = -np.sin(theta_rand) * Ax_aux + np.cos(theta_rand) * Ay_aux

        out = np.zeros((2, len(t)), dtype=np.complex128)
        out[0, :] = Ax
        out[1, :] = Ay
        return out


def EDFA(s_in, t, *, G_edfa_dB=20.0, NF_dB=5.0, lambda0=1550e-9):
    amplifier = EDFAAmplifier()
    return amplifier.apply(s_in, t, G_edfa_dB=G_edfa_dB, NF_dB=NF_dB, lambda0=lambda0)


def OpticalFiberWrapper(A, t, **kwargs):
    return OpticalFiber().propagate(A, t, **kwargs)
