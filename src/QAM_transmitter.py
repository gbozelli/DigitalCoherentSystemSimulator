import numpy as np
import matplotlib.pyplot as plt
from QAM_mod import QAM_mod
from DAC_Nyquist import DAC_Nyquist
from single_frequency_laser import single_frequency_laser
from MZM import MZM



# QAM transmitter
def QAM_transmitter(N_sync=128,sync_seed=0,\
                    N_MIMO=32,MIMO_i=0,\
                    N_inf=1024,M=16,SpS=16,RollOff=0.2,ts=1e-12,
                    N_zeros_init=20,N_zeros_final=20,plot_flag=True):

    ## Sequence generation

    # Initial zeros
    s_zeros_init = np.zeros(N_zeros_init)
    # Syncronization sequence
    np.random.seed(sync_seed)
    s_sync_i = np.random.randint(0,2,N_sync,dtype=int)*2-1
    # MIMO sequence
    #print(MIMO_i)
    if MIMO_i == 1:
        s_MIMO_i = np.append(np.zeros(int(N_MIMO/2)),np.ones(int(N_MIMO/2)))
    else:
        s_MIMO_i = np.append(np.ones(int(N_MIMO/2)),np.zeros(int(N_MIMO/2)))
    s_MIMO_i = s_MIMO_i*np.mod(np.arange(N_MIMO)+MIMO_i,2)
    #print(s_MIMO_i)

    # Information sequence
    N_bits = int(np.log2(M))
    N_bits_total = int(N_bits*N_inf)
    s_b = np.random.randint(0,2,N_bits_total,dtype=int)
    s_inf = QAM_mod(s_b,M)
    s_inf = s_inf/np.max(np.real(s_inf))

    # Final zeros
    s_zeros_final = np.zeros(N_zeros_final)

    # Concatenate
    s = np.concatenate([s_zeros_init,s_sync_i,s_inf,s_zeros_final])

    ## Conversion to the continuous time
    s_cont = DAC_Nyquist(s=s,SpS=SpS,RollOff=RollOff)

    ind = np.arange(len(s))*SpS

    t = ts*(np.arange(len(s_cont)))

    if plot_flag:
        plt.figure()
        plt.plot(t*1e9,np.real(s_cont))
        plt.plot(ind*ts*1e9,np.real(s),'.')
        plt.xlabel('Time [ns]')
        plt.ylabel('In-phase component [a.u.]')
        plt.xlim([0,np.max(t)*1e9])

        plt.figure()
        plt.plot(np.real(s_cont),np.imag(s_cont),alpha=0.2)
        plt.plot(np.real(s),np.imag(s),'.')
        plt.xlabel('In-phase')
        plt.ylabel('Quadrature')
        plt.axis('square')

    return [s_cont,t,s_b]

def DPQAM_transmitter(M = 16, SpS = 16, RollOff = 0.2, ts = 1e-9, sync_seed_X=0,sync_seed_Y=123,\
                           N_MIMO = 128,N_sync = 128,N_inf = 4096,N_zeros_init=10,N_zeros_final=10,Delta_nu=0,Freq_offset=0,\
                           ind_mod = 0.1, Splitter_IL=1.0,Upper_IL=2.0,Lower_IL=2.0,Combiner_IL=1.1,\
                           Vpi = 2.5, Vbias=2.5, plot_flag = False):

    #------------------------------------- Electrical signal --------------------------------------------%
    # Complex signal for X polarization
    s_tx_X, t, s_b_X = QAM_transmitter(M = M, SpS = SpS, RollOff = RollOff,ts = ts,sync_seed=sync_seed_X,\
                           N_sync = N_sync,N_MIMO=N_MIMO,MIMO_i=0,N_inf = N_inf,N_zeros_init=N_zeros_init,N_zeros_final=N_zeros_final,\
                           plot_flag = False)

    # Complex signal for Y polarization
    s_tx_Y, t, s_b_Y = QAM_transmitter(M = M, SpS = SpS, RollOff = RollOff,ts = ts,sync_seed=sync_seed_Y,\
                           N_sync = N_sync,N_MIMO=N_MIMO,MIMO_i=1,N_inf = N_inf,N_zeros_init=N_zeros_init,N_zeros_final=N_zeros_final,\
                           plot_flag = False)

    # Combination of the signals for both polarizations
    # Binary sequences
    s_b = np.zeros((2,len(s_b_X)),dtype=int)
    s_b[0,:] = s_b_X
    s_b[1,:] = s_b_Y
    # Electrical signals
    s_tx = np.zeros((2,len(t)),dtype=complex)
    s_tx[0,:] = s_tx_X
    s_tx[1,:] = s_tx_Y

    #--------------------------------------- Optical signal ---------------------------------------------%
    E_carrier = single_frequency_laser(t = t, P = P_laser_TX, Delta_nu = Delta_nu,\
                                       Freq_offset = Freq_offset,X_fraction = 0.5, phi_pol = 0)

    # Polarization splitting
    E_carrier_X = E_carrier[0,:]
    E_carrier_Y = E_carrier[1,:]

    #------------------------------------- Optical modulation --------------------------------------------%
    # Modulation depth
    Kmod = ind_mod

    # Modulation of the X component
    # Modulation of the in-phase component of the X polarization
    E_carrier_X_I = E_carrier_X/np.sqrt(2)
    s_tx_X_I = np.real(s_tx_X)
    Vrf_upper = Kmod*s_tx_X_I
    Vrf_lower = -Kmod*s_tx_X_I
    E_mod_X_I = MZM(E_carrier_X_I,Vrf_upper,Vrf_lower,Splitter_IL=Splitter_IL,Upper_IL=Upper_IL,\
                    Lower_IL=Lower_IL,Combiner_IL=Combiner_IL,Vpi = Vpi,Vbias=Vbias)
    # Modulation of the in-phase component of the X polarization
    E_carrier_X_Q = E_carrier_X/np.sqrt(2)*np.exp(1j*np.pi/2)
    s_tx_X_Q = np.imag(s_tx_X)
    Vrf_upper = Kmod*s_tx_X_Q
    Vrf_lower = -Kmod*s_tx_X_Q
    E_mod_X_Q = MZM(E_carrier_X_Q,Vrf_upper,Vrf_lower,Splitter_IL=Splitter_IL,Upper_IL=Upper_IL,\
                    Lower_IL=Lower_IL,Combiner_IL=Combiner_IL,Vpi = Vpi,Vbias=Vbias)
    # Combine the components
    E_mod_X = 1/np.sqrt(2)*E_mod_X_I+1/np.sqrt(2)*E_mod_X_Q

    # Modulation of the Y component
    # Modulation of the in-phase component of the Y polarization
    E_carrier_Y_I = E_carrier_Y/np.sqrt(2)
    s_tx_Y_I = np.real(s_tx_Y)
    Vrf_upper = Kmod*s_tx_Y_I
    Vrf_lower = -Kmod*s_tx_Y_I
    E_mod_Y_I = MZM(E_carrier_Y_I,Vrf_upper,Vrf_lower,Splitter_IL=Splitter_IL,Upper_IL=Upper_IL,\
                    Lower_IL=Lower_IL,Combiner_IL=Combiner_IL,Vpi = Vpi,Vbias=Vbias)
    # Modulation of the in-phase component of the X polarization
    E_carrier_Y_Q = E_carrier_Y/np.sqrt(2)*np.exp(1j*np.pi/2)
    s_tx_Y_Q = np.imag(s_tx_Y)
    Vrf_upper = Kmod*s_tx_Y_Q
    Vrf_lower = -Kmod*s_tx_Y_Q
    E_mod_Y_Q = MZM(E_carrier_Y_Q,Vrf_upper,Vrf_lower,Splitter_IL=Splitter_IL,Upper_IL=Upper_IL,\
                    Lower_IL=Lower_IL,Combiner_IL=Combiner_IL,Vpi = Vpi,Vbias=Vbias)
    # Combine the components
    E_mod_Y = 1/np.sqrt(2)*E_mod_Y_I+1/np.sqrt(2)*E_mod_Y_Q

    if plot_flag:
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(np.real(E_mod_X))
        plt.plot(np.imag(E_mod_X))
        plt.subplot(2,1,2)
        plt.plot(np.real(E_mod_Y))
        plt.plot(np.imag(E_mod_Y))

    # Combine polarizations
    E_opt = np.zeros((2,len(t)),dtype=complex)
    #print(np.shape(A))
    E_opt[0,:] = E_mod_X
    E_opt[1,:] = E_mod_Y

    return [t, E_opt, E_carrier, s_tx, s_b]
