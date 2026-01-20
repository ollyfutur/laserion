#ifndef FIELD_CACHE_H
#define FIELD_CACHE_H

#include <mpi.h>
#include "inputdeck.h"
#include "laser.h"

#ifdef __cplusplus
extern "C"
{
#endif

    typedef struct
    {
        int compute_A;     // 0: E only, 1: E and A
        int merge_on_root; // 0: keep per-rank files only, 1: root merges into a single file
        int root_rank;     // usually 0
    } FieldCacheOptions;

    /* Reasonable defaults: compute E and A, and merge on root. */
    FieldCacheOptions field_cache_default_options(void);

    /*
     * Compute and write field cache for the grid defined in `sim->grid`,
     * using the supplied active LaserPulse `pulse`.
     *
     * Output naming:
     *   - per-rank files:  <prefix>.rankNNNN.h5
     *   - merged file:     <prefix>.h5   (only if merge_on_root=1)
     *
     * Returns 0 on success.
     */
    int field_cache_run(const InputSimSpec *sim,
                        const LaserPulse *pulse,
                        const char *prefix,
                        const FieldCacheOptions *opt,
                        MPI_Comm comm);

#ifdef __cplusplus
}
#endif

#endif
