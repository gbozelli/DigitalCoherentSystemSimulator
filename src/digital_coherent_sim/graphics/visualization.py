from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt


def spectrum_analysis(s, t):
    ts = t[1] - t[0]
    fs = 1 / ts
    Deltaf = 1 / (ts * len(t))
    f = np.arange(len(t)) * Deltaf - fs / 2
    s = np.ravel(s)
    plt.figure()
    plt.plot(f / 1e9, 30 + 10 * np.log10(np.fft.fftshift(np.abs(np.fft.fft(s)))))
    plt.xlabel("Frequency [GHz]")
    plt.ylabel("Power [dBm]")
    plt.tight_layout()
    return plt.gcf()
