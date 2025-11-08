import numpy as np
import numpy.matlib as ml
import matplotlib.pyplot as plt
from scipy import signal
from QAM_transmitter import QAM_transmitter
from QAM_receiver import QAM_receiver
from QAM_receiver_DP import QAM_receiver_DP
from DPQAM_receiver import DPQAM_receiver
from single_frequency_laser import single_frequency_laser
from MZM import MZM
from HybridNetwork import HybridNetwork
from photodiode import photodiode
from fiber import fiber
from EDFA import EDFA
from constellation_analysis import constellation_analysis
from spectrum_analysis import spectrum_analysis
from DPQAM_transmitter import DPQAM_transmitter
from optical_filter import optical_filter

# General Parameters
M = 16                 # Number of points in the constellation
SpS = 16               # Number of samples per symbol
RollOff = 0.2          # Roll-off factor
BaudRate = 14e9        # Baudrate in Sps
Tsym = 1/BaudRate      # Symbol peiod in s
ts = Tsym/SpS          # Sampling peiod
N_zeros_init = 100     # Number of initial zeros
N_sync = 256           # Number of synchronization symbols
N_inf = 4*4096           # Number of information symbols (payload)
N_zeros_final = 100    # Number of final zeros
sync_seed_X=0          # Seed for the X polarization
sync_seed_Y=123        # Seed for the Y polarization

# Configuration of the optical blocks
P_laser_TX = 0.01      # Transmitter laser power in W
Delta_nu_TX = 100e3    # Transmitter laser linewidth in Hz
P_laser_RX = 0.01      # Transmitter laser power in W
Delta_nu_RX = 100e3    # Transmitter laser linewidth in Hz
Freq_offset_TX = 0     # Offset frequency of the transmitter laser in Hz
Freq_offset_RX = 55e6  # Offset frequency of the transmitter laser in Hz
ind_mod = 0.1          # Modulation index
alpha_dB = 0.2e-3     # Fiber attenuation in dB/m
D = 16e-6             # Dispersion in [s/m^2]
n2= 2.6e-20           # Nonlinear coeffient in m^2/W
DeltaL = 1e3          # Fiber segment in m
lambda0 = 1550e-9     # Nominal wavelength in m
Aeff = 80e-12         # Effective modal area in m^2
SoP_rotation = True   # Activate SoP rotation
#G_edfa_dB = 24
NF_dB = 5
P_LOP = 0.002          # Launch optical power in W

N_span = 5
L_span = 50e3
L = N_span*L_span

t, E_TX, E_carrier, s_tx, s_b = DPQAM_transmitter(P_laser_TX,M = M, SpS = SpS, RollOff = RollOff, \
                ts = ts, sync_seed_X=0,sync_seed_Y=123,N_sync = 256,N_inf = N_inf,N_zeros_init=N_zeros_init,\
                N_zeros_final=N_zeros_final,ind_mod = ind_mod,Delta_nu=Delta_nu_TX,Freq_offset=Freq_offset_TX,\
                plot_flag = True)
# Visualization of the spectrum
#spectrum_analysis(E_TX[0,:],t)
P_aux = np.mean((np.abs(E_TX[0,:]))**2)+np.mean((np.abs(E_TX[0,:]))**2)
G_edfa_dB = 10*np.log10(P_LOP/P_aux)
print('The required gain is:',G_edfa_dB,'dB')
E_TX = EDFA(E_TX,t,G_edfa_dB,NF_dB,lambda0=1550e-9)

# Multispan channel
E_inline = E_TX
G_edfa_dB = alpha_dB*L_span

print(G_edfa_dB)

for aux in range(N_span):
    print('Signal propagation in span number',aux)
    E_inline = fiber(E_inline,t,L_span,DeltaL,D,lambda0,alpha_dB,n2,Aeff,SoP_rotation) # Output field
    E_inline = EDFA(E_inline,t,G_edfa_dB,NF_dB,lambda0=1550e-9)
E_RX = E_inline

# Optical receiver
BER = DPQAM_receiver(E_RX,E_RX,t,M,SpS,N_inf,RollOff,ts,s_b, P_laser = P_laser_RX, Delta_nu = Delta_nu_RX, N_sync = N_sync, sync_seed_X=0,sync_seed_Y=123,Freq_offset = Freq_offset_RX, \
                     plot_flag = True, shot_noise = True, thermal_noise = True,L=L,D=D)
print('The BER of the X component is:',BER[0])
print('The BER of the Y component is:',BER[1])

plt.close('all')
