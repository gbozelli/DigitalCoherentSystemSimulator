# Digital Coherent System Simulator

This repository provides an educational and research-oriented Python simulator for digital coherent communication systems. The implementation targets the main functional blocks of a coherent transceiver:

- transmitter DSP and optical modulation
- fiber channel propagation and amplification
- coherent receiver DSP and symbol decision
- visualization utilities for spectra and constellations

## Package Layout

- src/digital_coherent_sim/transmitter.py: transmitter orchestration
- src/digital_coherent_sim/channel.py: EDFA and optical fiber models
- src/digital_coherent_sim/receiver.py: receiver orchestration
- src/digital_coherent_sim/graphics.py: plotting helpers
- src/main.py: command-line entry point

## Development Commands

```bash
python -m pip install -r requirements.txt
pytest
ruff check .
```

## Future Direction

The current structure is designed to be published as a Python library in the future. The package API is intentionally centered around small, composable objects and functions that can evolve independently.
