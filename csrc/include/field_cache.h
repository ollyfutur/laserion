#ifndef FIELD_CACHE_H
#define FIELD_CACHE_H

#include <stdint.h>
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
        uint64_t io_buffer_bytes;
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

    int field_cache_is_compatible(const char *cache_dir,
                                  const InputSimSpec *sim,
                                  int require_A,
                                  char *why, size_t why_sz);

#ifdef __cplusplus
}
#endif

#endif
