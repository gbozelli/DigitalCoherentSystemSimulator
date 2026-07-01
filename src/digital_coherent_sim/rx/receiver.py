from __future__ import annotations


class DPQAMReceiver:
    def __init__(self, config: dict | None = None) -> None:
        self.config = config or {}

    def run(self, E_RX, t, M, SpS, N_inf, RollOff, ts, s_b, *, channel_id=None, power_dBm=None,
            P_laser=0.01, Delta_nu=0, N_sync=256, sync_seed_X=0, sync_seed_Y=123,
            Freq_offset=0, plot_flag=False, BW=10e9, shot_noise=True,
            thermal_noise=True, L=0, D=0, R=0.9):
        from ...DPQAM_receiver import DPQAM_receiver as legacy_receiver

        return legacy_receiver(
            E_RX,
            t,
            M,
            SpS,
            N_inf,
            RollOff,
            ts,
            s_b,
            channel_id=channel_id,
            power_dBm=power_dBm,
            P_laser=P_laser,
            Delta_nu=Delta_nu,
            N_sync=N_sync,
            sync_seed_X=sync_seed_X,
            sync_seed_Y=sync_seed_Y,
            Freq_offset=Freq_offset,
            plot_flag=plot_flag,
            BW=BW,
            shot_noise=shot_noise,
            thermal_noise=thermal_noise,
            L=L,
            D=D,
            R=R,
        )
