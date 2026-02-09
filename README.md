# laserion

`laserion` is a toolkit for defining laser pulses, polarization states,
and computing momentum distribution functions (MDFs) for electrons born via
tunnel ionization.

The code provides:
- modular laser pulse definitions (temporal, transverse, and polarization),
- single-pulse and multi-pulse field evaluation,
- basic plotting helpers for fields and MDFs,
- an ADK-based ionization model for MDF construction.

## Installation (Python)

Clone the repository and install in editable mode:

```bash
pip install -e .
```

## Installation (C)

The C implementation is recommended for production use, as it supports MPI parallelism and HDF5 output.

Requirements:
- A C compiler (preferably an MPI-enabled compiler such as `mpicc`)
- HDF5

From the `csrc/` directory, run:

```bash
make CC=/path/to/mpi/bin/mpicc H5_ROOT=/path/to/hdf5/
```
The compiled executable will be located at `csrc/bin/laserion`

Note: `H5_ROOT` must point to the HDF5 installation prefix containing `include/` and `lib/`. The MPI compiler used to build HDF5 must match the one provided via `CC`. At present, the code has been tested only with GNU compilers and OpenMPI.