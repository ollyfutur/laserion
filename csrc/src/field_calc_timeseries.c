/* ============================================================================
 * File: field_calc_timeseries.c
 * ============================================================================
 */
#include "field_calc_timeseries.h"

#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* -------------------------- MPI safety helpers --------------------------- */

static int mpi_safe_init_if_needed(int *did_init)
{
    int inited = 0;
    *did_init = 0;

    if (MPI_Initialized(&inited) != MPI_SUCCESS)
        return 1;

    if (!inited)
    {
        if (MPI_Init(NULL, NULL) != MPI_SUCCESS)
            return 2;
        *did_init = 1;
    }
    return 0;
}

static void mpi_safe_finalize_if_needed(int did_init)
{
    if (did_init)
        MPI_Finalize();
}

/* -------------------------- distribution helpers -------------------------- */

static void compute_counts_displs_points(int size, int np, int *counts, int *displs)
{
    const int base = (size > 0) ? (np / size) : 0;
    const int rem  = (size > 0) ? (np % size) : 0;

    int disp = 0;
    for (int r = 0; r < size; ++r)
    {
        const int cnt = base + (r < rem ? 1 : 0);
        counts[r] = cnt;
        displs[r] = disp;
        disp += cnt;
    }
}

static int gatherv_floats(MPI_Comm comm,
                          const float *sendbuf, int sendcount,
                          float *recvbuf,
                          const int *recvcounts, const int *displs,
                          int root)
{
    return (MPI_Gatherv((void*)sendbuf, sendcount, MPI_FLOAT,
                        recvbuf, (int*)recvcounts, (int*)displs, MPI_FLOAT,
                        root, comm) == MPI_SUCCESS) ? 0 : 1;
}

/* -------------------------- dataset naming helpers ------------------------ */

static size_t count_requested_datasets(uint32_t qmask, uint32_t cmask)
{
    size_t n = 0;
    if (qmask & (uint32_t)FIELD_Q_E)
    {
        if (cmask & (uint32_t)FIELD_C_X) ++n;
        if (cmask & (uint32_t)FIELD_C_Y) ++n;
        if (cmask & (uint32_t)FIELD_C_Z) ++n;
    }
    if (qmask & (uint32_t)FIELD_Q_A)
    {
        if (cmask & (uint32_t)FIELD_C_X) ++n;
        if (cmask & (uint32_t)FIELD_C_Y) ++n;
        if (cmask & (uint32_t)FIELD_C_Z) ++n;
    }
    return n;
}

static void fill_dataset_descriptors(uint32_t qmask, uint32_t cmask,
                                     const char **names,
                                     const char **units,
                                     const char **long_names)
{
    size_t k = 0;

    /* E */
    if (qmask & (uint32_t)FIELD_Q_E)
    {
        if (cmask & (uint32_t)FIELD_C_X) { names[k]="fields/Ex"; units[k]="GV/m"; long_names[k]="E_x"; ++k; }
        if (cmask & (uint32_t)FIELD_C_Y) { names[k]="fields/Ey"; units[k]="GV/m"; long_names[k]="E_y"; ++k; }
        if (cmask & (uint32_t)FIELD_C_Z) { names[k]="fields/Ez"; units[k]="GV/m"; long_names[k]="E_z"; ++k; }
    }

    /* A */
    if (qmask & (uint32_t)FIELD_Q_A)
    {
        /* Unit for A depends on your normalization; keep "a.u." as placeholder string */
        if (cmask & (uint32_t)FIELD_C_X) { names[k]="fields/Ax"; units[k]="a.u."; long_names[k]="A_x"; ++k; }
        if (cmask & (uint32_t)FIELD_C_Y) { names[k]="fields/Ay"; units[k]="a.u."; long_names[k]="A_y"; ++k; }
        if (cmask & (uint32_t)FIELD_C_Z) { names[k]="fields/Az"; units[k]="a.u."; long_names[k]="A_z"; ++k; }
    }
}

/* ----------------------------- main routine ------------------------------ */

int field_calc_timeseries_points(const struct LaserPulse *pulse,
                                 const FieldRequest *req,
                                 FieldResult *out,
                                 MPI_Comm comm)
{
    if (!pulse || !req || !out) return 1;
    if (req->kind != FIELD_REQ_TIMESERIES_POINTS) return 2;

    /* Validate request consistency */
    if (field_request_validate(req) != 0) return 3;

    const FieldRequestTimeseriesPoints *R = &req->u.ts_points;
    const uint32_t qmask = req->opt.quantity_mask;
    const uint32_t cmask = req->opt.component_mask;

    const size_t np = R->np;

    /* Materialize time axis values on all ranks (needed for evaluation) */
    FieldAxis t_axis;
    if (field_axis_from_spec(&t_axis, &R->t) != 0) return 4;
    const size_t nt = t_axis.n;

    int did_init = 0;
    if (mpi_safe_init_if_needed(&did_init) != 0)
    {
        free(t_axis.values);
        return 5;
    }

    int rank = 0, size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    const int root = req->opt.root_rank;

    /* Distribution arrays (points per rank) */
    int *pcounts = (int*)calloc((size_t)size, sizeof(int));
    int *pdispls = (int*)calloc((size_t)size, sizeof(int));
    if (!pcounts || !pdispls)
    {
        free(pcounts); free(pdispls);
        free(t_axis.values);
        mpi_safe_finalize_if_needed(did_init);
        return 6;
    }
    compute_counts_displs_points(size, (int)np, pcounts, pdispls);

    const int local_np = pcounts[rank];
    const int local_p0 = pdispls[rank];

    /* Number of datasets requested */
    const size_t nds = count_requested_datasets(qmask, cmask);

    /* Local buffers per dataset: [local_np][nt] with t fastest */
    const size_t n_local = (size_t)local_np * nt;

    float **local_ds = NULL;
    if (nds > 0)
    {
        local_ds = (float**)calloc(nds, sizeof(float*));
        if (!local_ds)
        {
            free(pcounts); free(pdispls);
            free(t_axis.values);
            mpi_safe_finalize_if_needed(did_init);
            return 7;
        }
        for (size_t k = 0; k < nds; ++k)
        {
            local_ds[k] = (float*)malloc(sizeof(float) * n_local);
            if (!local_ds[k])
            {
                for (size_t j = 0; j < k; ++j) free(local_ds[j]);
                free(local_ds);
                free(pcounts); free(pdispls);
                free(t_axis.values);
                mpi_safe_finalize_if_needed(did_init);
                return 8;
            }
        }
    }

    /* Dataset name/units (shared, non-owning) */
    const char **names = NULL, **units = NULL, **long_names = NULL;
    if (nds > 0)
    {
        names      = (const char**)calloc(nds, sizeof(char*));
        units      = (const char**)calloc(nds, sizeof(char*));
        long_names = (const char**)calloc(nds, sizeof(char*));
        if (!names || !units || !long_names)
        {
            free(names); free(units); free(long_names);
            for (size_t k = 0; k < nds; ++k) free(local_ds[k]);
            free(local_ds);
            free(pcounts); free(pdispls);
            free(t_axis.values);
            mpi_safe_finalize_if_needed(did_init);
            return 9;
        }
        fill_dataset_descriptors(qmask, cmask, names, units, long_names);
    }

    /* Compute local: loop points, then time */
    for (int ip = 0; ip < local_np; ++ip)
    {
        const int ig = local_p0 + ip;
        const double rr[3] = {
            (np > 0) ? R->points_um[ig][0] : 0.0,
            (np > 0) ? R->points_um[ig][1] : 0.0,
            (np > 0) ? R->points_um[ig][2] : 0.0
        };

        for (size_t it = 0; it < nt; ++it)
        {
            const double tt = t_axis.values[it];

            double E[3] = {0,0,0};
            double A[3] = {0,0,0};

            if (qmask & (uint32_t)FIELD_Q_E)
                LaserPulse_E(pulse, tt, rr, E);
            if (qmask & (uint32_t)FIELD_Q_A)
                LaserPulse_A(pulse, tt, rr, A);

            /* Fill local dataset buffers in the same order as fill_dataset_descriptors */
            size_t k = 0;
            const size_t idx = (size_t)ip * nt + it;

            if (qmask & (uint32_t)FIELD_Q_E)
            {
                if (cmask & (uint32_t)FIELD_C_X) local_ds[k++][idx] = (float)E[0];
                if (cmask & (uint32_t)FIELD_C_Y) local_ds[k++][idx] = (float)E[1];
                if (cmask & (uint32_t)FIELD_C_Z) local_ds[k++][idx] = (float)E[2];
            }
            if (qmask & (uint32_t)FIELD_Q_A)
            {
                if (cmask & (uint32_t)FIELD_C_X) local_ds[k++][idx] = (float)A[0];
                if (cmask & (uint32_t)FIELD_C_Y) local_ds[k++][idx] = (float)A[1];
                if (cmask & (uint32_t)FIELD_C_Z) local_ds[k++][idx] = (float)A[2];
            }
        }
    }

    /* Assemble policy */
    if (req->opt.assemble != FIELD_ASSEMBLE_GATHER_TO_ROOT)
    {
        /* For now, only gather-to-root is implemented for timeseries. */
        free(names); free(units); free(long_names);
        for (size_t k = 0; k < nds; ++k) free(local_ds[k]);
        free(local_ds);
        free(pcounts); free(pdispls);
        free(t_axis.values);
        mpi_safe_finalize_if_needed(did_init);
        return 10;
    }

    /* Prepare gatherv counts/displs in float-elements */
    int *rcounts_f = (int*)calloc((size_t)size, sizeof(int));
    int *rdispls_f = (int*)calloc((size_t)size, sizeof(int));
    if (!rcounts_f || !rdispls_f)
    {
        free(rcounts_f); free(rdispls_f);
        free(names); free(units); free(long_names);
        for (size_t k = 0; k < nds; ++k) free(local_ds[k]);
        free(local_ds);
        free(pcounts); free(pdispls);
        free(t_axis.values);
        mpi_safe_finalize_if_needed(did_init);
        return 11;
    }
    for (int r = 0; r < size; ++r)
    {
        rcounts_f[r] = (int)((size_t)pcounts[r] * nt);
        rdispls_f[r] = (int)((size_t)pdispls[r] * nt);
    }

    /* Root allocates and fills FieldResult */
    field_result_init(out);

    if (rank == root)
    {
        out->layout = FIELD_RES_2D;
        out->naxes  = 2;
        out->label  = req->opt.label;

        /* axes[0] = time (fastest), axes[1] = point index */
        out->axes[0] = t_axis; /* take ownership of t_axis.values */
        if (field_axis_make_index(&out->axes[1], FIELD_AXIS_P, np, "p", "1") != 0)
        {
            field_result_free(out);
            /* t_axis.values owned by out now; field_result_free will free it */
            free(rcounts_f); free(rdispls_f);
            free(names); free(units); free(long_names);
            for (size_t k = 0; k < nds; ++k) free(local_ds[k]);
            free(local_ds);
            free(pcounts); free(pdispls);
            mpi_safe_finalize_if_needed(did_init);
            return 12;
        }

        /* Store point coordinates (optional but very useful) */
        out->has_points = (np > 0) ? 1 : 0;
        if (out->has_points)
        {
            out->points_x_um = (float*)malloc(sizeof(float) * np);
            out->points_y_um = (float*)malloc(sizeof(float) * np);
            out->points_z_um = (float*)malloc(sizeof(float) * np);
            if (!out->points_x_um || !out->points_y_um || !out->points_z_um)
            {
                field_result_free(out);
                free(rcounts_f); free(rdispls_f);
                free(names); free(units); free(long_names);
                for (size_t k = 0; k < nds; ++k) free(local_ds[k]);
                free(local_ds);
                free(pcounts); free(pdispls);
                mpi_safe_finalize_if_needed(did_init);
                return 13;
            }
            for (size_t i = 0; i < np; ++i)
            {
                out->points_x_um[i] = (float)R->points_um[i][0];
                out->points_y_um[i] = (float)R->points_um[i][1];
                out->points_z_um[i] = (float)R->points_um[i][2];
            }
        }

        /* Allocate datasets (shape: [nt][np] with axis1 fastest -> but our data is [np][nt].
           Important: FieldResult convention was: dims[0]=axis1.n, dims[1]=axis2.n.
           axis1 is time (nt) and axis2 is p (np).
           However, local buffers are stored as [p][t] (t fastest), which corresponds to
           a contiguous array where time varies fastest inside each point.
           That is equivalent to a 2D array with dims = (nt, np) if the storage is transposed
           relative to indexing. To avoid confusion, we define the dataset memory layout as:
             data[(ip*nt) + it]  (p-major, t-minor)
           and we keep it consistent for writers: they must interpret as [np][nt] with t fastest.
           Therefore: we set dims[0]=nt and dims[1]=np BUT we will store in out->datasets[k].data
           in the same [np][nt] order and document that axis1 is fastest (t).
           This is consistent if writer uses "axis1 fastest" and expects contiguous blocks per axis2.
        */
        /* We will allocate datasets but override dims to reflect [np][nt] storage with t fastest. */

        if (field_result_alloc_datasets(out, nds, names, units, long_names) != 0)
        {
            field_result_free(out);
            free(rcounts_f); free(rdispls_f);
            free(names); free(units); free(long_names);
            for (size_t k = 0; k < nds; ++k) free(local_ds[k]);
            free(local_ds);
            free(pcounts); free(pdispls);
            mpi_safe_finalize_if_needed(did_init);
            return 14;
        }

        /* Override dataset dims to match storage [np][nt] but describe axes order (t, p).
           We'll keep ndims=2, dims[0]=nt, dims[1]=np. */
        for (size_t k = 0; k < nds; ++k)
        {
            out->datasets[k].ndims = 2;
            out->datasets[k].dims[0] = nt;
            out->datasets[k].dims[1] = np;
        }
    }
    else
    {
        /* Non-root must free t_axis.values because it is not transferred into out */
        free(t_axis.values);
    }

    /* Root receives into out->datasets[k].data; others pass NULL */
    int gather_err = 0;
    for (size_t k = 0; k < nds; ++k)
    {
        float *recvbuf = (rank == root) ? out->datasets[k].data : NULL;
        /* sendcount in floats */
        const int sendcount = (int)n_local;
        gather_err |= gatherv_floats(comm, local_ds[k], sendcount, recvbuf, rcounts_f, rdispls_f, root);
    }

    /* Ensure all ranks see failure */
    int gather_err_all = 0;
    MPI_Allreduce(&gather_err, &gather_err_all, 1, MPI_INT, MPI_MAX, comm);

    /* Cleanup local */
    free(rcounts_f); free(rdispls_f);
    free(names); free(units); free(long_names);

    for (size_t k = 0; k < nds; ++k) free(local_ds[k]);
    free(local_ds);

    free(pcounts); free(pdispls);

    mpi_safe_finalize_if_needed(did_init);

    if (gather_err_all != 0)
    {
        if (rank == root)
            field_result_free(out);
        return 15;
    }

    return 0;
}
/* ============================================================================
 * File: field_calc_timeseries.c
 * ============================================================================
 */
#include "field_calc_timeseries.h"

#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* -------------------------- MPI safety helpers --------------------------- */

static int mpi_safe_init_if_needed(int *did_init)
{
    int inited = 0;
    *did_init = 0;

    if (MPI_Initialized(&inited) != MPI_SUCCESS)
        return 1;

    if (!inited)
    {
        if (MPI_Init(NULL, NULL) != MPI_SUCCESS)
            return 2;
        *did_init = 1;
    }
    return 0;
}

static void mpi_safe_finalize_if_needed(int did_init)
{
    if (did_init)
        MPI_Finalize();
}

/* -------------------------- distribution helpers -------------------------- */

static void compute_counts_displs_points(int size, int np, int *counts, int *displs)
{
    const int base = (size > 0) ? (np / size) : 0;
    const int rem  = (size > 0) ? (np % size) : 0;

    int disp = 0;
    for (int r = 0; r < size; ++r)
    {
        const int cnt = base + (r < rem ? 1 : 0);
        counts[r] = cnt;
        displs[r] = disp;
        disp += cnt;
    }
}

static int gatherv_floats(MPI_Comm comm,
                          const float *sendbuf, int sendcount,
                          float *recvbuf,
                          const int *recvcounts, const int *displs,
                          int root)
{
    return (MPI_Gatherv((void*)sendbuf, sendcount, MPI_FLOAT,
                        recvbuf, (int*)recvcounts, (int*)displs, MPI_FLOAT,
                        root, comm) == MPI_SUCCESS) ? 0 : 1;
}

/* -------------------------- dataset naming helpers ------------------------ */

static size_t count_requested_datasets(uint32_t qmask, uint32_t cmask)
{
    size_t n = 0;
    if (qmask & (uint32_t)FIELD_Q_E)
    {
        if (cmask & (uint32_t)FIELD_C_X) ++n;
        if (cmask & (uint32_t)FIELD_C_Y) ++n;
        if (cmask & (uint32_t)FIELD_C_Z) ++n;
    }
    if (qmask & (uint32_t)FIELD_Q_A)
    {
        if (cmask & (uint32_t)FIELD_C_X) ++n;
        if (cmask & (uint32_t)FIELD_C_Y) ++n;
        if (cmask & (uint32_t)FIELD_C_Z) ++n;
    }
    return n;
}

static void fill_dataset_descriptors(uint32_t qmask, uint32_t cmask,
                                     const char **names,
                                     const char **units,
                                     const char **long_names)
{
    size_t k = 0;

    /* E */
    if (qmask & (uint32_t)FIELD_Q_E)
    {
        if (cmask & (uint32_t)FIELD_C_X) { names[k]="fields/Ex"; units[k]="GV/m"; long_names[k]="E_x"; ++k; }
        if (cmask & (uint32_t)FIELD_C_Y) { names[k]="fields/Ey"; units[k]="GV/m"; long_names[k]="E_y"; ++k; }
        if (cmask & (uint32_t)FIELD_C_Z) { names[k]="fields/Ez"; units[k]="GV/m"; long_names[k]="E_z"; ++k; }
    }

    /* A */
    if (qmask & (uint32_t)FIELD_Q_A)
    {
        /* Unit for A depends on your normalization; keep "a.u." as placeholder string */
        if (cmask & (uint32_t)FIELD_C_X) { names[k]="fields/Ax"; units[k]="a.u."; long_names[k]="A_x"; ++k; }
        if (cmask & (uint32_t)FIELD_C_Y) { names[k]="fields/Ay"; units[k]="a.u."; long_names[k]="A_y"; ++k; }
        if (cmask & (uint32_t)FIELD_C_Z) { names[k]="fields/Az"; units[k]="a.u."; long_names[k]="A_z"; ++k; }
    }
}

/* ----------------------------- main routine ------------------------------ */

int field_calc_timeseries_points(const struct LaserPulse *pulse,
                                 const FieldRequest *req,
                                 FieldResult *out,
                                 MPI_Comm comm)
{
    if (!pulse || !req || !out) return 1;
    if (req->kind != FIELD_REQ_TIMESERIES_POINTS) return 2;

    /* Validate request consistency */
    if (field_request_validate(req) != 0) return 3;

    const FieldRequestTimeseriesPoints *R = &req->u.ts_points;
    const uint32_t qmask = req->opt.quantity_mask;
    const uint32_t cmask = req->opt.component_mask;

    const size_t np = R->np;

    /* Materialize time axis values on all ranks (needed for evaluation) */
    FieldAxis t_axis;
    if (field_axis_from_spec(&t_axis, &R->t) != 0) return 4;
    const size_t nt = t_axis.n;

    int did_init = 0;
    if (mpi_safe_init_if_needed(&did_init) != 0)
    {
        free(t_axis.values);
        return 5;
    }

    int rank = 0, size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    const int root = req->opt.root_rank;

    /* Distribution arrays (points per rank) */
    int *pcounts = (int*)calloc((size_t)size, sizeof(int));
    int *pdispls = (int*)calloc((size_t)size, sizeof(int));
    if (!pcounts || !pdispls)
    {
        free(pcounts); free(pdispls);
        free(t_axis.values);
        mpi_safe_finalize_if_needed(did_init);
        return 6;
    }
    compute_counts_displs_points(size, (int)np, pcounts, pdispls);

    const int local_np = pcounts[rank];
    const int local_p0 = pdispls[rank];

    /* Number of datasets requested */
    const size_t nds = count_requested_datasets(qmask, cmask);

    /* Local buffers per dataset: [local_np][nt] with t fastest */
    const size_t n_local = (size_t)local_np * nt;

    float **local_ds = NULL;
    if (nds > 0)
    {
        local_ds = (float**)calloc(nds, sizeof(float*));
        if (!local_ds)
        {
            free(pcounts); free(pdispls);
            free(t_axis.values);
            mpi_safe_finalize_if_needed(did_init);
            return 7;
        }
        for (size_t k = 0; k < nds; ++k)
        {
            local_ds[k] = (float*)malloc(sizeof(float) * n_local);
            if (!local_ds[k])
            {
                for (size_t j = 0; j < k; ++j) free(local_ds[j]);
                free(local_ds);
                free(pcounts); free(pdispls);
                free(t_axis.values);
                mpi_safe_finalize_if_needed(did_init);
                return 8;
            }
        }
    }

    /* Dataset name/units (shared, non-owning) */
    const char **names = NULL, **units = NULL, **long_names = NULL;
    if (nds > 0)
    {
        names      = (const char**)calloc(nds, sizeof(char*));
        units      = (const char**)calloc(nds, sizeof(char*));
        long_names = (const char**)calloc(nds, sizeof(char*));
        if (!names || !units || !long_names)
        {
            free(names); free(units); free(long_names);
            for (size_t k = 0; k < nds; ++k) free(local_ds[k]);
            free(local_ds);
            free(pcounts); free(pdispls);
            free(t_axis.values);
            mpi_safe_finalize_if_needed(did_init);
            return 9;
        }
        fill_dataset_descriptors(qmask, cmask, names, units, long_names);
    }

    /* Compute local: loop points, then time */
    for (int ip = 0; ip < local_np; ++ip)
    {
        const int ig = local_p0 + ip;
        const double rr[3] = {
            (np > 0) ? R->points_um[ig][0] : 0.0,
            (np > 0) ? R->points_um[ig][1] : 0.0,
            (np > 0) ? R->points_um[ig][2] : 0.0
        };

        for (size_t it = 0; it < nt; ++it)
        {
            const double tt = t_axis.values[it];

            double E[3] = {0,0,0};
            double A[3] = {0,0,0};

            if (qmask & (uint32_t)FIELD_Q_E)
                LaserPulse_E(pulse, tt, rr, E);
            if (qmask & (uint32_t)FIELD_Q_A)
                LaserPulse_A(pulse, tt, rr, A);

            /* Fill local dataset buffers in the same order as fill_dataset_descriptors */
            size_t k = 0;
            const size_t idx = (size_t)ip * nt + it;

            if (qmask & (uint32_t)FIELD_Q_E)
            {
                if (cmask & (uint32_t)FIELD_C_X) local_ds[k++][idx] = (float)E[0];
                if (cmask & (uint32_t)FIELD_C_Y) local_ds[k++][idx] = (float)E[1];
                if (cmask & (uint32_t)FIELD_C_Z) local_ds[k++][idx] = (float)E[2];
            }
            if (qmask & (uint32_t)FIELD_Q_A)
            {
                if (cmask & (uint32_t)FIELD_C_X) local_ds[k++][idx] = (float)A[0];
                if (cmask & (uint32_t)FIELD_C_Y) local_ds[k++][idx] = (float)A[1];
                if (cmask & (uint32_t)FIELD_C_Z) local_ds[k++][idx] = (float)A[2];
            }
        }
    }

    /* Assemble policy */
    if (req->opt.assemble != FIELD_ASSEMBLE_GATHER_TO_ROOT)
    {
        /* For now, only gather-to-root is implemented for timeseries. */
        free(names); free(units); free(long_names);
        for (size_t k = 0; k < nds; ++k) free(local_ds[k]);
        free(local_ds);
        free(pcounts); free(pdispls);
        free(t_axis.values);
        mpi_safe_finalize_if_needed(did_init);
        return 10;
    }

    /* Prepare gatherv counts/displs in float-elements */
    int *rcounts_f = (int*)calloc((size_t)size, sizeof(int));
    int *rdispls_f = (int*)calloc((size_t)size, sizeof(int));
    if (!rcounts_f || !rdispls_f)
    {
        free(rcounts_f); free(rdispls_f);
        free(names); free(units); free(long_names);
        for (size_t k = 0; k < nds; ++k) free(local_ds[k]);
        free(local_ds);
        free(pcounts); free(pdispls);
        free(t_axis.values);
        mpi_safe_finalize_if_needed(did_init);
        return 11;
    }
    for (int r = 0; r < size; ++r)
    {
        rcounts_f[r] = (int)((size_t)pcounts[r] * nt);
        rdispls_f[r] = (int)((size_t)pdispls[r] * nt);
    }

    /* Root allocates and fills FieldResult */
    field_result_init(out);

    if (rank == root)
    {
        out->layout = FIELD_RES_2D;
        out->naxes  = 2;
        out->label  = req->opt.label;

        /* axes[0] = time (fastest), axes[1] = point index */
        out->axes[0] = t_axis; /* take ownership of t_axis.values */
        if (field_axis_make_index(&out->axes[1], FIELD_AXIS_P, np, "p", "1") != 0)
        {
            field_result_free(out);
            /* t_axis.values owned by out now; field_result_free will free it */
            free(rcounts_f); free(rdispls_f);
            free(names); free(units); free(long_names);
            for (size_t k = 0; k < nds; ++k) free(local_ds[k]);
            free(local_ds);
            free(pcounts); free(pdispls);
            mpi_safe_finalize_if_needed(did_init);
            return 12;
        }

        /* Store point coordinates (optional but very useful) */
        out->has_points = (np > 0) ? 1 : 0;
        if (out->has_points)
        {
            out->points_x_um = (float*)malloc(sizeof(float) * np);
            out->points_y_um = (float*)malloc(sizeof(float) * np);
            out->points_z_um = (float*)malloc(sizeof(float) * np);
            if (!out->points_x_um || !out->points_y_um || !out->points_z_um)
            {
                field_result_free(out);
                free(rcounts_f); free(rdispls_f);
                free(names); free(units); free(long_names);
                for (size_t k = 0; k < nds; ++k) free(local_ds[k]);
                free(local_ds);
                free(pcounts); free(pdispls);
                mpi_safe_finalize_if_needed(did_init);
                return 13;
            }
            for (size_t i = 0; i < np; ++i)
            {
                out->points_x_um[i] = (float)R->points_um[i][0];
                out->points_y_um[i] = (float)R->points_um[i][1];
                out->points_z_um[i] = (float)R->points_um[i][2];
            }
        }

        /* Allocate datasets (shape: [nt][np] with axis1 fastest -> but our data is [np][nt].
           Important: FieldResult convention was: dims[0]=axis1.n, dims[1]=axis2.n.
           axis1 is time (nt) and axis2 is p (np).
           However, local buffers are stored as [p][t] (t fastest), which corresponds to
           a contiguous array where time varies fastest inside each point.
           That is equivalent to a 2D array with dims = (nt, np) if the storage is transposed
           relative to indexing. To avoid confusion, we define the dataset memory layout as:
             data[(ip*nt) + it]  (p-major, t-minor)
           and we keep it consistent for writers: they must interpret as [np][nt] with t fastest.
           Therefore: we set dims[0]=nt and dims[1]=np BUT we will store in out->datasets[k].data
           in the same [np][nt] order and document that axis1 is fastest (t).
           This is consistent if writer uses "axis1 fastest" and expects contiguous blocks per axis2.
        */
        /* We will allocate datasets but override dims to reflect [np][nt] storage with t fastest. */

        if (field_result_alloc_datasets(out, nds, names, units, long_names) != 0)
        {
            field_result_free(out);
            free(rcounts_f); free(rdispls_f);
            free(names); free(units); free(long_names);
            for (size_t k = 0; k < nds; ++k) free(local_ds[k]);
            free(local_ds);
            free(pcounts); free(pdispls);
            mpi_safe_finalize_if_needed(did_init);
            return 14;
        }

        /* Override dataset dims to match storage [np][nt] but describe axes order (t, p).
           We'll keep ndims=2, dims[0]=nt, dims[1]=np. */
        for (size_t k = 0; k < nds; ++k)
        {
            out->datasets[k].ndims = 2;
            out->datasets[k].dims[0] = nt;
            out->datasets[k].dims[1] = np;
        }
    }
    else
    {
        /* Non-root must free t_axis.values because it is not transferred into out */
        free(t_axis.values);
    }

    /* Root receives into out->datasets[k].data; others pass NULL */
    int gather_err = 0;
    for (size_t k = 0; k < nds; ++k)
    {
        float *recvbuf = (rank == root) ? out->datasets[k].data : NULL;
        /* sendcount in floats */
        const int sendcount = (int)n_local;
        gather_err |= gatherv_floats(comm, local_ds[k], sendcount, recvbuf, rcounts_f, rdispls_f, root);
    }

    /* Ensure all ranks see failure */
    int gather_err_all = 0;
    MPI_Allreduce(&gather_err, &gather_err_all, 1, MPI_INT, MPI_MAX, comm);

    /* Cleanup local */
    free(rcounts_f); free(rdispls_f);
    free(names); free(units); free(long_names);

    for (size_t k = 0; k < nds; ++k) free(local_ds[k]);
    free(local_ds);

    free(pcounts); free(pdispls);

    mpi_safe_finalize_if_needed(did_init);

    if (gather_err_all != 0)
    {
        if (rank == root)
            field_result_free(out);
        return 15;
    }

    return 0;
}

