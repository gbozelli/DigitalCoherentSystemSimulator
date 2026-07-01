from __future__ import annotations


class DPQAMTransmitter:
    def __init__(self, config: dict | None = None) -> None:
        self.config = config or {}

    def run(self, *, M=16, SpS=16, RollOff=0.2, ts=1e-9, sync_seed_X=0, sync_seed_Y=123,
            inf_seed_X=23, inf_seed_Y=37, N_MIMO=128, N_sync=128, N_inf=4096,
            N_zeros_init=10, N_zeros_final=10, Delta_nu=0, Freq_offset=0, ind_mod=0.1,
            Splitter_IL=1.0, Upper_IL=2.0, Lower_IL=2.0, Combiner_IL=1.1,
            Vpi=2.5, Vbias=2.5, plot_flag=False):
        from ...DPQAM_transmitter import DPQAM_transmitter as legacy_transmitter

        return legacy_transmitter(
            P_laser_TX=self.config.get("P_laser_TX", 0.01),
            M=M,
            SpS=SpS,
            RollOff=RollOff,
            ts=ts,
            sync_seed_X=sync_seed_X,
            sync_seed_Y=sync_seed_Y,
            inf_seed_X=inf_seed_X,
            inf_seed_Y=inf_seed_Y,
            N_MIMO=N_MIMO,
            N_sync=N_sync,
            N_inf=N_inf,
            N_zeros_init=N_zeros_init,
            N_zeros_final=N_zeros_final,
            Delta_nu=Delta_nu,
            Freq_offset=Freq_offset,
            ind_mod=ind_mod,
            Splitter_IL=Splitter_IL,
            Upper_IL=Upper_IL,
            Lower_IL=Lower_IL,
            Combiner_IL=Combiner_IL,
            Vpi=Vpi,
            Vbias=Vbias,
            plot_flag=plot_flag,
        )
