#ifndef FIELD_DIAG_H
#define FIELD_DIAG_H

#include <mpi.h>
#include "inputdeck.h"

#ifdef __cplusplus
extern "C"
{
#endif

    typedef struct
    {
        int root_rank;         /* rank that performs reads/writes (default 0) */
        int write_time_iter_0; /* if nonzero, write TIME=0 and ITER=0 in output */
    } FieldDiagOptions;

    FieldDiagOptions field_diag_default_options(void);

    /*
     * Run all field diagnostics specified by sim->field_diag by reading cache files
     * from sim->field_cache.out_dir and writing slices as HDF5 grid diags via diag_h5.
     *
     * Output naming:
     *   <out_prefix>_fd<idx>_<Comp>_<axes>.h5
     *
     * Returns 0 on success; nonzero on failure (root rank reports).
     */
    int field_diag_run_from_cache(const InputSimSpec *sim,
                                  const char *out_prefix,
                                  const FieldDiagOptions *opt,
                                  MPI_Comm comm);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* FIELD_DIAG_H */
