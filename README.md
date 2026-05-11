# Lumoxic Engine

Core computational engine for [Lumoxic AI](https://lumoxicai.me) photon computing research.

The engine converts light into binary data using the **Binary Bounce Engine** — photons bounce off reflective surfaces, and each bounce angle is converted to a binary bit (angle ≥ 45° → `1`, angle < 45° → `0`).

## Installation

```bash
pip install -e .
```

For development:
```bash
pip install -e ".[dev]"
```

## Quick Start

### Python API

```python
from lumoxic import Client

client = Client(threshold=45.0, seed=42)

# Process a single photon
result = client.process(wavelength=450, angle=37, bounces=8)
print(result.binary_stream.as_string)  # e.g. "01101010"

# Run batch simulation
result = client.simulate(photon_count=256, wavelength=450)
print(f"Generated {result.binary_stream.length} bits")
print(f"Entropy: {result.binary_stream.entropy:.4f}")

# Analyze binary output
analysis = client.analyze(result.binary_stream.bits)
print(f"Frequency: {analysis['frequency']}")
```

### CLI

```bash
# Process a single photon
lumoxic process --wavelength 450 --angle 37 --bounces 8

# Run simulation
lumoxic simulate --photons 256 --wavelength 450 --json

# Train LNBE model
lumoxic train --epochs 100 --samples 1000
```

## Architecture

```
lumoxic/
├── client.py          # Unified Client API
├── cli.py             # Command-line interface
├── core/              # Types, constants
├── photon/            # Photon emission and processing
├── bounce/            # Binary Bounce Engine (core simulation)
├── binary/            # Angle-to-bit encoding, stream assembly, pattern analysis
├── models/            # LNBE neural network, inference, training
├── optics/            # Wavelength analysis, refraction (Snell's law)
└── probability/       # Probability field mapping
```

## Core Concept

The Binary Bounce Engine works by:
1. **Emitting** a photon at a specific wavelength and angle
2. **Bouncing** it off reflective surfaces (glass, mirror, crystal)
3. **Encoding** each bounce angle to binary (≥45° = 1, <45° = 0)
4. **Assembling** the bits into a binary stream for AI training

## Tests

```bash
pytest tests/ -v
```

## License

MIT — Copyright 2026 Lumoxic AI
