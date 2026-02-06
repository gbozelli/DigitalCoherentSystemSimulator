# Digital Coherent System Simulator — Core Source (`src`)

A modular **digital coherent communication system simulator** implemented in Python.  
This project provides an end-to-end simulation framework for baseband digital signal processing (DSP) chains used in modern coherent communication systems, with emphasis on performance evaluation and research-oriented experimentation.

The simulator is intended for use in **digital and optical communications research**, enabling analysis of modulation schemes, channel impairments, receiver DSP algorithms, and Bit Error Rate (BER) performance.

---

## Overview

Coherent communication systems transmit complex baseband signals that are affected by linear and stochastic impairments introduced by the channel. At the receiver, digital signal processing techniques are employed to mitigate these effects and recover the transmitted information with minimal error probability.

This simulator implements a structured and extensible DSP chain analogous to real-world coherent transceivers, allowing controlled experimentation and performance analysis.

Main features include:

- Transmitter-side digital modulation and pulse shaping
- Channel modeling with noise and carrier impairments
- Receiver-side DSP blocks for synchronization and equalization
- BER computation and constellation analysis
- Modular architecture for research and algorithm development

---

## Repository Structure

```
src/
├── transmitter/           # Transmitter DSP blocks and modulation
├── channel/               # Channel models and impairments
├── receiver/              # Receiver DSP algorithms
├── utils/                 # Utility and helper functions
├── config.py              # Global simulation parameters
└── main.py                # Simulation entry point
```

---

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/gbozelli/DigitalCoherentSystemSimulator.git
cd DigitalCoherentSystemSimulator
pip install -r requirements.txt
```

---

## System Architecture

The simulator follows a conventional coherent communication pipeline:

1. **Bit Generation and Mapping**  
   Random bit sequences are generated and mapped to complex symbols according to the selected modulation scheme.

2. **Pulse Shaping**  
   Transmit filtering is applied to limit signal bandwidth and reduce intersymbol interference.

3. **Channel Modeling**  
   The signal is impaired by additive noise and carrier distortions, emulating practical transmission conditions.

4. **Receiver DSP Chain**  
   - Matched filtering  
   - Timing recovery (optional)  
   - Adaptive equalization  
   - Carrier frequency and phase recovery  

5. **Demodulation and Performance Evaluation**  
   Detected symbols are demapped into bits and compared against the transmitted sequence to compute BER.

Each block is configurable and can be independently modified or replaced.

---

## Modules Description

### Transmitter

- **Bit Source**: Pseudo-random binary sequence generation  
- **Symbol Mapper**: Mapping to complex constellations (e.g., QAM)  
- **Pulse Shaping Filter**: Typically Root-Raised Cosine (RRC)

### Channel

- **AWGN Channel**: Additive white Gaussian noise  
- **Phase Noise**: Random carrier phase variations  
- **Frequency Offset**: Carrier frequency mismatch

### Receiver

- **Matched Filter**: Maximizes signal-to-noise ratio  
- **Equalizer**: Adaptive linear equalization (e.g., LMS, CMA)  
- **Carrier Recovery**: Estimation and correction of phase and frequency offsets  
- **Timing Recovery**: Optional symbol timing estimation

---

## Running Simulations

A typical simulation can be executed via:

```bash
python src/main.py --modulation 16qam --snr 20 --num-samples 1000000 --equalizer lms
```

Common parameters include:

| Parameter | Description |
|---------|-------------|
| `--modulation` | Modulation format (e.g., QPSK, 16QAM) |
| `--snr` | Signal-to-noise ratio in dB |
| `--num-samples` | Number of transmitted symbols |
| `--equalizer` | Equalization algorithm |

---

## Output Metrics

The simulator provides quantitative and qualitative performance indicators:

- Bit Error Ratio (BER)
- Constellation diagrams
- Equalizer convergence behavior

These outputs facilitate direct comparison between algorithms and system configurations.

---

## Research Applications

This simulator is suitable for:

- Performance evaluation of coherent DSP algorithms
- Study of modulation formats under channel impairments
- Testing adaptive equalization and carrier recovery techniques
- Integration with optimization or machine-learning-based DSP approaches

---

## License

This project is released under the MIT License.
