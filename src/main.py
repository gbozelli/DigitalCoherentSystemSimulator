from __future__ import annotations

import argparse

import numpy as np

from digital_coherent_sim import DPQAMReceiver, DPQAMTransmitter, EDFA, OpticalFiber, spectrum_analysis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a digital coherent system simulation")
    parser.add_argument("--snr", type=float, default=20.0)
    parser.add_argument("--modulation", default="16qam")
    parser.add_argument("--num-samples", type=int, default=4096)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    tx = DPQAMTransmitter({"P_laser_TX": 0.01})
    t, E_TX, E_carrier, s_tx, const_tx, s_b = tx.run(N_inf=args.num_samples)
    E_amp = EDFA(E_TX, t, G_edfa_dB=20.0, NF_dB=5.0, lambda0=1550e-9)
    E_RX = OpticalFiber().propagate(E_amp, t)
    receiver = DPQAMReceiver()
    result = receiver.run(E_RX, t, M=16, SpS=16, N_inf=args.num_samples, RollOff=0.2, ts=t[1] - t[0], s_b=s_b)
    print(f"Simulation completed for modulation={args.modulation} snr={args.snr}")
    print(result)
    spectrum_analysis(E_TX[0, :], t)


if __name__ == "__main__":
    main()
