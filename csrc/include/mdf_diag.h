#pragma once

#include <mpi.h>
#include "inputdeck.h"
#define MDF_DIAG_SKIPPED_NO_A 777001

#ifdef __cplusplus
extern "C"
{
#endif

    /* Compute all phase-space MDF diagnostics requested in sim->phase_space
     * from cached fields located in cache_dir (expects Ex/Ey/Ez, and optionally Ax/Ay/Az).
     *
     * Output: one HDF5 file per PhaseSpaceSpec, written to out_dir:
     *   out_dir/mdf_<kind>.h5
     *
     * Parallelism:
     *   - all ranks participate in computation and HDF5 reads
     *   - histogram reduced with MPI_Allreduce
     *   - root (rank 0) writes output
     */
    int mdf_diag_run_all_from_cache(const InputSimSpec *sim,
                                    const char *cache_dir,
                                    const char *out_dir,
                                    MPI_Comm comm);

    /* Compute one MDF histogram for a given PhaseSpaceSpec from cache. */
    int mdf_diag_run_one_from_cache(const InputSimSpec *sim,
                                    const PhaseSpaceSpec *ps,
                                    const char *cache_dir,
                                    const char *out_dir,
                                    MPI_Comm comm);

#ifdef __cplusplus
} /* extern "C" */
#endif
