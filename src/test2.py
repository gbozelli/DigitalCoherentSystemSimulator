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

plt.close('all')

# General Parameters
M = 16                 # Number of points in the constellation
SpS = 16               # Number of samples per symbol
RollOff = 0.2          # Roll-off factor
BaudRate = 4*14e9        # Baudrate in Sps
Tsym = 1/BaudRate      # Symbol period in s
ts = Tsym/SpS          # Sampling period
N_zeros_init = 100     # Number of initial zeros
N_sync = 256           # Number of synchronization symbols
N_inf = 4096           # Number of information symbols (payload)
N_zeros_final = 100    # Number of final zeros
sync_seed_X=0          # Seed for the X polarization
sync_seed_Y=123        # Seed for the Y polarization

# Configuration of the optical blocks
P_laser_TX = 0.01      # Transmitter laser power in W
Delta_nu_TX = 0*100e3    # Transmitter laser linewidth in Hz
P_laser_RX = 0.01      # Transmitter laser power in W
Delta_nu_RX = 0*100e3    # Transmitter laser linewidth in Hz
ind_mod = 0.1          # Modulation index

L = 90e3             # Fiber length in [m]
alpha_dB = 0.2e-3     # Fiber attenuation in dB/m
D = 16e-6             # Dispersion in [s/m^2]
n2= 2.6e-20           # Nonlinear coeffient in m^2/W
DeltaL = 1e3          # Fiber segment in m
lambda0 = 1550e-9     # Nominal wavelength in m
Aeff = 80e-12         # Effective modal area in m^2
SoP_rotation = False   # Activate SoP rotation
#G_edfa_dB = 24
NF_dB = 3


P_LOP_dB_array = np.linspace(-5,10,31)
P_LOP_array = 10**(P_LOP_dB_array/10)
print(P_LOP_dB_array)
print(P_LOP_array)
BER_1_array = np.zeros(len(P_LOP_array))
BER_2_array = np.zeros(len(P_LOP_array))
BER_3_array = np.zeros(len(P_LOP_array))
BER_4_array = np.zeros(len(P_LOP_array))
for counter, aux_LOP in enumerate(P_LOP_array):
    plt.close('all')
    print('\n')
    print(counter,aux_LOP)
    plt.close('all')
    P_LOP = aux_LOP/1e3

    # Channel 1
    print('Generating channel 1...')
    Delta_f_1 = -3*75e9/2                 # Center frequency channel 1
    Freq_offset_TX_1 = Delta_f_1       # Offset frequency of the transmitter laser in Hz
    Freq_offset_RX_1 = Delta_f_1  # Offset frequency of the transmitter laser in Hz
    t, E_TX, E_carrier, s_tx, const_tx_1, s_b_1 = DPQAM_transmitter(P_laser_TX, M = M, SpS = SpS, RollOff = RollOff, \
                    ts = ts, sync_seed_X=0,sync_seed_Y=123,inf_seed_X=2,inf_seed_Y=3,
                    N_sync = 256,N_inf = N_inf,N_zeros_init=N_zeros_init,\
                    N_zeros_final=N_zeros_final,ind_mod = ind_mod,Delta_nu=Delta_nu_TX,Freq_offset=Freq_offset_TX_1,\
                    plot_flag = False)
    P_aux = np.mean((np.abs(E_TX[0,:]))**2)+np.mean((np.abs(E_TX[1,:]))**2)
    G_edfa_dB = 10*np.log10(P_LOP/P_aux)
    #print('The required gain is:',G_edfa_dB,'dB')
    E_LOP_1 = EDFA(E_TX,t,G_edfa_dB,NF_dB,lambda0=1550e-9)
    E_LOP_1 = E_LOP_1-np.mean(E_LOP_1)
    #print(s_b_1[0,:20])
    #print(s_b_1[1,:20])
    
    # Channel 2
    print('Generating channel 2...')
    Delta_f_2 = -75e9/2                 # Center frequency channel 2
    Freq_offset_TX_2 = Delta_f_2       # Offset frequency of the transmitter laser in Hz
    Freq_offset_RX_2 = Delta_f_2  # Offset frequency of the transmitter laser in Hz
    t, E_TX, E_carrier, s_tx, const_tx_2, s_b_2 = DPQAM_transmitter(P_laser_TX, M = M, SpS = SpS, RollOff = RollOff, \
                    ts = ts, sync_seed_X=0,sync_seed_Y=123,inf_seed_X=4,inf_seed_Y=5,
                    N_sync = 256,N_inf = N_inf,N_zeros_init=N_zeros_init,\
                    N_zeros_final=N_zeros_final,ind_mod = ind_mod,Delta_nu=Delta_nu_TX,Freq_offset=Freq_offset_TX_2,\
                    plot_flag = False)
    P_aux = np.mean((np.abs(E_TX[0,:]))**2)+np.mean((np.abs(E_TX[1,:]))**2)
    G_edfa_dB = 10*np.log10(P_LOP/P_aux)
    #print('The required gain is:',G_edfa_dB,'dB')
    E_LOP_2 = EDFA(E_TX,t,G_edfa_dB,NF_dB,lambda0=1550e-9)
    E_LOP_2 = E_LOP_2-np.mean(E_LOP_2)
    #print(s_b_2[0,:20])
    #print(s_b_2[1,:20])
    
    # Channel 3
    print('Generating channel 3...')
    Delta_f_3 = 75e9/2                 # Center frequency channel 1
    Freq_offset_TX_3 = Delta_f_3       # Offset frequency of the transmitter laser in Hz
    Freq_offset_RX_3 = Delta_f_3  # Offset frequency of the transmitter laser in Hz
    t, E_TX, E_carrier, s_tx, const_tx_3, s_b_3 = DPQAM_transmitter(P_laser_TX, M = M, SpS = SpS, RollOff = RollOff, \
                    ts = ts, sync_seed_X=0,sync_seed_Y=123,inf_seed_X=6,inf_seed_Y=7,
                    N_sync = 256,N_inf = N_inf,N_zeros_init=N_zeros_init,\
                    N_zeros_final=N_zeros_final,ind_mod = ind_mod,Delta_nu=Delta_nu_TX,Freq_offset=Freq_offset_TX_3,\
                    plot_flag = False)
    P_aux = np.mean((np.abs(E_TX[0,:]))**2)+np.mean((np.abs(E_TX[1,:]))**2)
    G_edfa_dB = 10*np.log10(P_LOP/P_aux)
    #print('The required gain is:',G_edfa_dB,'dB')
    E_LOP_3 = EDFA(E_TX,t,G_edfa_dB,NF_dB,lambda0=1550e-9)
    E_LOP_3 = E_LOP_3-np.mean(E_LOP_3)
    #print(s_b_3[0,:20])
    #print(s_b_3[1,:20])
    
    # Channel 4
    print('Generating channel 4...')
    Delta_f_4 = 3*75e9/2                 # Center frequency channel 2
    Freq_offset_TX_4 = Delta_f_4       # Offset frequency of the transmitter laser in Hz
    Freq_offset_RX_4 = Delta_f_4  # Offset frequency of the transmitter laser in Hz
    t, E_TX, E_carrier, s_tx, const_tx_4, s_b_4 = DPQAM_transmitter(P_laser_TX, M = M, SpS = SpS, RollOff = RollOff, \
                    ts = ts, sync_seed_X=0,sync_seed_Y=123,inf_seed_X=8,inf_seed_Y=9,
                    N_sync = 256,N_inf = N_inf,N_zeros_init=N_zeros_init,\
                    N_zeros_final=N_zeros_final,ind_mod = ind_mod,Delta_nu=Delta_nu_TX,Freq_offset=Freq_offset_TX_4,\
                    plot_flag = False)
    P_aux = np.mean((np.abs(E_TX[0,:]))**2)+np.mean((np.abs(E_TX[1,:]))**2)
    G_edfa_dB = 10*np.log10(P_LOP/P_aux)
    #print('The required gain is:',G_edfa_dB,'dB')
    E_LOP_4 = EDFA(E_TX,t,G_edfa_dB,NF_dB,lambda0=1550e-9)
    E_LOP_4 = E_LOP_4-np.mean(E_LOP_4)
    #print(s_b_4[0,:20])
    #print(s_b_4[1,:20])
    
    
    print('Combining the signals...')
    E_LOP = E_LOP_1+E_LOP_2+E_LOP_3+E_LOP_4
    spectrum_analysis(E_LOP[0,:],t)
    
    # Channel
    print('Transmitting through the fiber...')
    E_RX = fiber(E_LOP,t,L,DeltaL,D,lambda0,alpha_dB,n2,Aeff,SoP_rotation) # Output field
    
    # Optical receiver
    
    # Received spectrum
    spectrum_analysis(E_RX[0,:],t)
    
    # Channel 1
    print('Receiving channel 1...')
    E_RXf = optical_filter(E_RX,t,f0 = Freq_offset_TX_1, BW = 75e9)
    BER, const_rx = DPQAM_receiver(E_RX,t,M,SpS,N_inf,RollOff,ts, s_b_1, P_laser = P_laser_RX,channel_id=1, Delta_nu = Delta_nu_RX, N_sync = N_sync, sync_seed_X=0,sync_seed_Y=123,
                         Freq_offset = Freq_offset_RX_1, \
                         plot_flag = True, BW=BaudRate*0.75, shot_noise = True, thermal_noise = True,L=L,D=D)
    #BER = DPQAM_receiver(E_RXf,s_b, P_laser = P_laser_RX, Delta_nu = Delta_nu_RX, N_sync = N_sync, sync_seed_X=0,sync_seed_Y=123,Freq_offset = Freq_offset_RX_1, \
    #                     plot_flag = False, BW=BaudRate*0.75, shot_noise = True, thermal_noise = True,L=L,D=D)
    print('Channel 1 - The BER of the X component is:',BER[0])
    print('Channel 1 - The BER of the Y component is:',BER[1])
    BER_1_array[counter] = (BER[0]+BER[1])/2
    
    # Channel 2
    print('Receiving channel 2...')
    E_RXf = optical_filter(E_RX,t,f0 = Freq_offset_TX_2, BW = 75e9)
    BER, const_rx = DPQAM_receiver(E_RX,t,M,SpS,N_inf,RollOff,ts, s_b_2, P_laser = P_laser_RX,channel_id=2, Delta_nu = Delta_nu_RX, N_sync = N_sync, sync_seed_X=0,sync_seed_Y=123,
                         Freq_offset = Freq_offset_RX_2, \
                         plot_flag = True, BW=BaudRate*0.75, shot_noise = True, thermal_noise = True,L=L,D=D)
    #BER = DPQAM_receiver(E_RXf,s_b, P_laser = P_laser_RX, Delta_nu = Delta_nu_RX, N_sync = N_sync, sync_seed_X=0,sync_seed_Y=123,Freq_offset = Freq_offset_RX_2, \
    #                     plot_flag = False, BW=BaudRate*0.75, shot_noise = True, thermal_noise = True,L=L,D=D)
    print('Channel 2 - The BER of the X component is:',BER[0])
    print('Channel 2 - The BER of the Y component is:',BER[1])
    BER_2_array[counter] = (BER[0]+BER[1])/2
    
    # Channel 3
    print('Receiving channel 3...')
    E_RXf = optical_filter(E_RX,t,f0 = Freq_offset_TX_3, BW = 75e9)
    BER, const_rx = DPQAM_receiver(E_RX,t,M,SpS,N_inf,RollOff,ts, s_b_3, P_laser = P_laser_RX,channel_id=3, Delta_nu = Delta_nu_RX, N_sync = N_sync, sync_seed_X=0,sync_seed_Y=123,
                         Freq_offset = Freq_offset_RX_3, \
                         plot_flag = True, BW=BaudRate*0.75, shot_noise = True, thermal_noise = True,L=L,D=D)
    #BER = DPQAM_receiver(E_RXf,s_b, P_laser = P_laser_RX, Delta_nu = Delta_nu_RX, N_sync = N_sync, sync_seed_X=0,sync_seed_Y=123,Freq_offset = Freq_offset_RX_3, \
    #                     plot_flag = False, BW=BaudRate*0.75, shot_noise = True, thermal_noise = True,L=L,D=D)
    print('Channel 3 - The BER of the X component is:',BER[0])
    print('Channel 3 - The BER of the Y component is:',BER[1])
    BER_3_array[counter] = (BER[0]+BER[1])/2
    
    # Channel 4
    print('Receiving channel 3...')
    E_RXf = optical_filter(E_RX,t,f0 = Freq_offset_TX_4, BW = 75e9)
    BER, const_rx = DPQAM_receiver(E_RX,t,M,SpS,N_inf,RollOff,ts, s_b_4, P_laser = P_laser_RX,channel_id=4, Delta_nu = Delta_nu_RX, N_sync = N_sync, sync_seed_X=0,sync_seed_Y=123,
                         Freq_offset = Freq_offset_RX_4, \
                         plot_flag = True, BW=BaudRate*0.75, shot_noise = True, thermal_noise = True,L=L,D=D)
    #BER = DPQAM_receiver(E_RXf,s_b, P_laser = P_laser_RX, Delta_nu = Delta_nu_RX, N_sync = N_sync, sync_seed_X=0,sync_seed_Y=123,Freq_offset = Freq_offset_RX_4, \
    #                     plot_flag = False, BW=BaudRate*0.75, shot_noise = True, thermal_noise = True,L=L,D=D)
    print('Channel 4 - The BER of the X component is:',BER[0])
    print('Channel 4 - The BER of the Y component is:',BER[1])
    BER_4_array[counter] = (BER[0]+BER[1])/2

plt.figure()
plt.plot(P_LOP_dB_array,BER_1_array)
plt.plot(P_LOP_dB_array,BER_2_array)
plt.plot(P_LOP_dB_array,BER_3_array)
plt.plot(P_LOP_dB_array,BER_4_array)
plt.yscale('log')