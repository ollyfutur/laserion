/* diag_h5.h */
#pragma once
#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

    /* ============================================================================
     * HDF5 Diagnostics (generic)
     *
     * Grid diagnostics:
     *   - root attribute TYPE="grid"
     *   - dataset /<dataset_name> float32 [n] or [n2,n1]
     *   - /AXIS/AXIS1 and /AXIS/AXIS2 are float64 [2] = [min,max]
     *
     * Particle diagnostics:
     *   - root attribute TYPE="particles"
     *   - root attributes QUANTS/LABELS/UNITS are string arrays (length = nquants)
     *   - root attribute OFFSET_T is float64 array (length = nquants) (default 0)
     *   - datasets are float32 [N]: ene,p1,p2,p3,x1,x2,x3,q
     *   - /SIMULATION group is NOT created (not required)
     * ========================================================================== */

    typedef enum
    {
        DIAG_KIND_GRID_1D = 1,
        DIAG_KIND_GRID_2D = 2,
        DIAG_KIND_PARTICLE = 3
    } DiagKind;

    typedef enum
    {
        DIAG_AXIS_T = 0,
        DIAG_AXIS_X = 1,
        DIAG_AXIS_Y = 2,
        DIAG_AXIS_Z = 3
    } DiagAxisID;

    typedef struct
    {
        double t, x, y, z;
    } DiagFixedCoords;

    /* Axis metadata written into /AXIS/AXIS{1,2} (min/max only). */
    typedef struct
    {
        DiagAxisID id;         /* t/x/y/z */
        const char *long_name; /* optional */
        const char *units;     /* optional */
        double vmin;
        double vmax;
    } DiagAxis;

    /* Particle schema entries (stored in root attrs QUANTS/LABELS/UNITS/OFFSET_T). */
    typedef struct
    {
        const char *quant; /* e.g. "x1", "p2", "ene" */
        const char *label; /* e.g. "z", "p_x", "\\epsilon_k" */
        const char *units; /* e.g. "\\mu m", "m_e c", "m_e c^2" */
        double offset_t;   /* default 0.0 for everything */
    } DiagParticleQuant;

    /* Particle physical mapping.
     * File names are fixed: x1,x2,x3 and p1,p2,p3, but we want semantic mapping.
     */
    typedef enum
    {
        /* Default requested mapping:
         * x1<-z, x2<-x, x3<-y, p1<-pz, p2<-px, p3<-py
         */
        DIAG_PARTICLE_MAP_ZXY_PZPXPY = 0,

        /* Alternative common mapping (if ever needed later):
         * x1<-x, x2<-y, x3<-z, p1<-px, p2<-py, p3<-pz
         */
        DIAG_PARTICLE_MAP_XYZ_PXPYPZ = 1
    } DiagParticleMap;

    /* Particle input data: provide physical arrays. Writer maps them to datasets.
     *
     * If q is NULL, it will be written as -1.0 for all particles.
     * If ene is NULL, it will be computed as sqrt(p^2 + 1) - 1 (float32 output).
     */
    typedef struct
    {
        size_t n;

        /* Physical positions [\mu m] (or your internal units; attrs will state \mu m). */
        const float *x;
        const float *y;
        const float *z;

        /* Physical momenta [m_e c]. */
        const float *px;
        const float *py;
        const float *pz;

        const float *q;   /* optional */
        const float *ene; /* optional: kinetic energy */
    } DiagParticleData;

    /* Unified diagnostic request. */
    typedef struct
    {
        DiagKind kind;

        const char *dataset_name; /* for grid: dataset name; for particles: species name */
        const char *units;        /* grid only (root UNITS attr); particles use per-quant UNITS[] */
        const char *label;        /* grid only (root LABEL attr) */

        double time_value; /* ok to set 0.0 */
        int iter_value;    /* ok to set 0 */

        union
        {
            struct
            {
                DiagAxis axis1;
                size_t n1;
                DiagFixedCoords fixed;
            } grid1d;

            struct
            {
                DiagAxis axis1;
                DiagAxis axis2;
                size_t n1;
                size_t n2;
                DiagFixedCoords fixed;
            } grid2d;

            struct
            {
                const DiagParticleQuant *quants; /* ordered schema; if NULL uses defaults */
                size_t nquants;                  /* if quants==NULL uses default size */

                DiagParticleMap map; /* default DIAG_PARTICLE_MAP_ZXY_PZPXPY */
            } particle;
        } u;

    } DiagRequest;

    /* Dispatcher:
     * - GRID_1D: data -> float[n1]
     * - GRID_2D: data -> float[n2*n1] (C-order)
     * - PARTICLE: data -> DiagParticleData*
     */
    int diag_h5_write(const char *path, const DiagRequest *req, const void *data);

    /* Convenience wrappers (grid). */
    int diag_h5_write_grid_1d(const char *path,
                              const char *dataset_name,
                              const char *units,
                              const char *label,
                              double time_value,
                              int iter_value,
                              const float *data,
                              size_t n,
                              const DiagAxis *axis1,
                              const DiagFixedCoords *fixed);

    int diag_h5_write_grid_2d(const char *path,
                              const char *dataset_name,
                              const char *units,
                              const char *label,
                              double time_value,
                              int iter_value,
                              const float *data,
                              size_t n1,
                              size_t n2,
                              const DiagAxis *axis1,
                              const DiagAxis *axis2,
                              const DiagFixedCoords *fixed);

    /* Particle writer (real). */
    int diag_h5_write_particles(const char *path,
                                const DiagRequest *req,
                                const DiagParticleData *pdata);

#ifdef __cplusplus
}
#endif
