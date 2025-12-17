#ifndef LASERION_DIAGNOSTICS_H
#define LASERION_DIAGNOSTICS_H

/*
 * Minimal diagnostics framework (Steps 1–3):
 *  - Time series of E/A at a fixed point:   (t) -> {Ex,Ey,Ez,Ax,Ay,Az}
 *  - 1D lineout of a chosen component:      (s) -> value   with s in {x,y,z} at fixed t and other coords
 *  - 2D slice of a chosen component:        (u,v) -> value with u,v in {x,y,z} at fixed t and fixed third coord
 *
 * Output format for now: CSV files (one file per diagnostic).
 *
 * MPI/HDF5 are intentionally not used here; the code structure keeps the outer loops
 * explicit so MPI domain-decomposition can be added later with minimal churn.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

    /* Forward declaration (from core.h) */
    typedef struct LaserPulse LaserPulse;

    /* Which field to sample */
    typedef enum
    {
        DIAG_FIELD_E = 0,
        DIAG_FIELD_A = 1
    } DiagFieldKind;

    /* Which vector component */
    typedef enum
    {
        DIAG_COMP_X = 0,
        DIAG_COMP_Y = 1,
        DIAG_COMP_Z = 2
    } DiagComponent;

    /* Which spatial axis */
    typedef enum
    {
        DIAG_AXIS_X = 0,
        DIAG_AXIS_Y = 1,
        DIAG_AXIS_Z = 2,
        DIAG_AXIS_T = 3 /* for completeness; not used as "spatial axis" */
    } DiagAxis;

    /* Common run context */
    typedef struct
    {
        const LaserPulse *pulse; /* SinglePulse or MultiPulse via base class */

        /* For A(t,r): caller may set pulse integration bounds elsewhere.
           Diagnostics just call LaserPulse_A(...) */
        double Atmin_fs;
        double Atmax_fs;
        double Adt_fs;
    } DiagContext;

    /* ---------------- Time series diagnostic ---------------- */

    typedef struct
    {
        /* fixed position */
        double x_um, y_um, z_um;

        /* time grid */
        double tmin_fs, tmax_fs;
        size_t nt;

        /* output */
        const char *csv_path;
    } DiagTimeSeries;

    /* Writes CSV with columns:
     *   t_fs, Ex,Ey,Ez, Ax,Ay,Az
     */
    int diag_run_time_series(const DiagContext *ctx, const DiagTimeSeries *spec);

    /* ---------------- 1D lineout diagnostic ---------------- */

    typedef struct
    {
        DiagFieldKind field; /* E or A */
        DiagComponent comp;  /* x/y/z */

        /* vary along one axis */
        DiagAxis axis; /* x/y/z only */
        double smin_um, smax_um;
        size_t ns;

        /* fixed other coordinates + time */
        double t_fs;
        double x_um, y_um, z_um;

        /* output */
        const char *csv_path;
    } DiagLineout1D;

    /* Writes CSV with columns:
     *   s_um, value
     */
    int diag_run_lineout_1d(const DiagContext *ctx, const DiagLineout1D *spec);

    /* ---------------- 2D slice diagnostic ---------------- */

    typedef struct
    {
        DiagFieldKind field; /* E or A */
        DiagComponent comp;  /* x/y/z */

        /* 2D axes u,v are two of x,y,z (must be different) */
        DiagAxis axis_u; /* x/y/z */
        DiagAxis axis_v; /* x/y/z */

        double umin_um, umax_um;
        double vmin_um, vmax_um;
        size_t nu, nv;

        /* fixed third coordinate + time */
        DiagAxis axis_fixed; /* the remaining axis among x,y,z */
        double fixed_um;
        double t_fs;

        /* output */
        const char *csv_path;
    } DiagSlice2D;

    /* Writes CSV with columns:
     *   u_um, v_um, value
     */
    int diag_run_slice_2d(const DiagContext *ctx, const DiagSlice2D *spec);

#ifdef __cplusplus
}
#endif

#endif /* LASERION_DIAGNOSTICS_H */
