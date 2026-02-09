# laserion

`laserion` is a Python toolkit for defining laser pulses, polarization states,
and computing momentum distribution functions (MDF) for electrons born by
tunnel ionization.

The code provides:
- modular laser pulse definitions (temporal, transverse, polarization),
- single-pulse and multi-pulse field evaluation,
- basic plotting helpers for fields and MDFs,
- an ADK-based ionization model for MDF construction.

This repository is under active development.

## Installation

Clone the repository and install in editable mode:

```bash
pip install -e .
```

# laserion C version

We have implemented the a C version with MPI support
