#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "diag_h5.h"

#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502884
#endif

/* ---------------- grid fillers ---------------- */

static void fill_1d(float *out, size_t n, double xmin, double xmax)
{
    for (size_t i = 0; i < n; ++i)
    {
        double s = (n > 1) ? (double)i / (double)(n - 1) : 0.0;
        double x = xmin + s * (xmax - xmin);
        out[i] = (float)(sin(2.0 * M_PI * x) * exp(-0.2 * x * x));
    }
}

static void fill_2d(float *out, size_t n1, size_t n2,
                    double xmin, double xmax,
                    double ymin, double ymax)
{
    /* C-order: out[j*n1 + i] corresponds to (i along axis1, j along axis2) */
    for (size_t j = 0; j < n2; ++j)
    {
        double sy = (n2 > 1) ? (double)j / (double)(n2 - 1) : 0.0;
        double y = ymin + sy * (ymax - ymin);

        for (size_t i = 0; i < n1; ++i)
        {
            double sx = (n1 > 1) ? (double)i / (double)(n1 - 1) : 0.0;
            double x = xmin + sx * (xmax - xmin);

            double r2 = x * x + y * y;
            out[j * n1 + i] = (float)(cos(2.0 * M_PI * x) * sin(2.0 * M_PI * y) * exp(-0.25 * r2));
        }
    }
}

/* ---------------- particle example ---------------- */

static void fill_particles(DiagParticleData *pd,
                           float *x, float *y, float *z,
                           float *px, float *py, float *pz,
                           size_t n)
{
    /* Simple synthetic bunch:
     * positions in micrometers, momenta in m_e c
     */
    for (size_t i = 0; i < n; ++i)
    {
        double s = (n > 1) ? (double)i / (double)(n - 1) : 0.0;

        /* Position: small helix in (x,y) and a ramp in z */
        z[i] = (float)(-50.0 + 100.0 * s);                /* [-50, +50] um */
        x[i] = (float)(10.0 * cos(2.0 * M_PI * 5.0 * s)); /* ~10 um */
        y[i] = (float)(10.0 * sin(2.0 * M_PI * 5.0 * s));

        /* Momentum: moderate gamma */
        pz[i] = (float)(3.0 + 0.5 * sin(2.0 * M_PI * s)); /* mostly forward */
        px[i] = (float)(0.2 * cos(2.0 * M_PI * 3.0 * s));
        py[i] = (float)(0.2 * sin(2.0 * M_PI * 3.0 * s));
    }

    pd->n = n;
    pd->x = x;
    pd->y = y;
    pd->z = z;
    pd->px = px;
    pd->py = py;
    pd->pz = pz;

    pd->q = NULL;   /* writer will output -1.0 for all particles */
    pd->ene = NULL; /* writer will compute sqrt(p^2+1)-1 */
}

int main(void)
{
    /* ---------------- 1D grid: Ey vs x ---------------- */
    {
        const size_t n1 = 512;
        const double xmin = -5.0, xmax = 5.0;

        float *data = (float *)malloc(n1 * sizeof(float));
        if (!data)
        {
            fprintf(stderr, "alloc failed\n");
            return 1;
        }
        fill_1d(data, n1, xmin, xmax);

        DiagRequest req = (DiagRequest){0};
        req.kind = DIAG_KIND_GRID_1D;
        req.dataset_name = "Ey";
        req.units = "GV/m";
        req.label = "E_y";
        req.time_value = 0.0;
        req.iter_value = 0;

        req.u.grid1d.axis1.id = DIAG_AXIS_X;
        req.u.grid1d.axis1.long_name = "x";
        req.u.grid1d.axis1.units = "\\mu m"; /* you prefer um */
        req.u.grid1d.axis1.vmin = xmin;
        req.u.grid1d.axis1.vmax = xmax;

        req.u.grid1d.n1 = n1;

        int rc = diag_h5_write("grid1d_Ey_vs_x.h5", &req, data);
        free(data);
        if (rc)
        {
            fprintf(stderr, "grid1d write failed rc=%d\n", rc);
            return 2;
        }

        printf("Wrote grid1d_Ey_vs_x.h5\n");
    }

    /* ---------------- 2D grid: Ey vs (x,y) ---------------- */
    {
        const size_t n1 = 256; /* x */
        const size_t n2 = 192; /* y */
        const double xmin = -6.0, xmax = 6.0;
        const double ymin = -4.0, ymax = 4.0;

        float *data = (float *)malloc(n1 * n2 * sizeof(float));
        if (!data)
        {
            fprintf(stderr, "alloc failed\n");
            return 3;
        }
        fill_2d(data, n1, n2, xmin, xmax, ymin, ymax);

        DiagRequest req = (DiagRequest){0};
        req.kind = DIAG_KIND_GRID_2D;
        req.dataset_name = "Ey";
        req.units = "GV/m";
        req.label = "E_y";
        req.time_value = 0.0;
        req.iter_value = 0;

        req.u.grid2d.axis1.id = DIAG_AXIS_X;
        req.u.grid2d.axis1.long_name = "x";
        req.u.grid2d.axis1.units = "\\mu m";
        req.u.grid2d.axis1.vmin = xmin;
        req.u.grid2d.axis1.vmax = xmax;

        req.u.grid2d.axis2.id = DIAG_AXIS_Y;
        req.u.grid2d.axis2.long_name = "y";
        req.u.grid2d.axis2.units = "\\mu m";
        req.u.grid2d.axis2.vmin = ymin;
        req.u.grid2d.axis2.vmax = ymax;

        req.u.grid2d.n1 = n1;
        req.u.grid2d.n2 = n2;

        int rc = diag_h5_write("grid2d_Ey_vs_xy.h5", &req, data);
        free(data);
        if (rc)
        {
            fprintf(stderr, "grid2d write failed rc=%d\n", rc);
            return 4;
        }

        printf("Wrote grid2d_Ey_vs_xy.h5\n");
    }

    /* ---------------- Particle file: RAW-species_1-000000.h5 ---------------- */
    {
        const size_t N = 20000;

        float *x = (float *)malloc(N * sizeof(float));
        float *y = (float *)malloc(N * sizeof(float));
        float *z = (float *)malloc(N * sizeof(float));
        float *px = (float *)malloc(N * sizeof(float));
        float *py = (float *)malloc(N * sizeof(float));
        float *pz = (float *)malloc(N * sizeof(float));

        if (!x || !y || !z || !px || !py || !pz)
        {
            fprintf(stderr, "alloc failed\n");
            free(x);
            free(y);
            free(z);
            free(px);
            free(py);
            free(pz);
            return 5;
        }

        DiagParticleData pd = (DiagParticleData){0};
        fill_particles(&pd, x, y, z, px, py, pz, N);

        DiagRequest req = (DiagRequest){0};
        req.kind = DIAG_KIND_PARTICLE;
        req.dataset_name = "species_1"; /* root NAME */
        req.time_value = 0.0;
        req.iter_value = 0;

        /* Default mapping requested:
         * x1<-z, x2<-x, x3<-y, p1<-pz, p2<-px, p3<-py
         */
        req.u.particle.map = DIAG_PARTICLE_MAP_ZXY_PZPXPY;

        /* Use default particle schema (QUANTS/LABELS/UNITS/OFFSET_T) */
        req.u.particle.quants = NULL;
        req.u.particle.nquants = 0;

        int rc = diag_h5_write("RAW-species_1-000000.h5", &req, &pd);

        free(x);
        free(y);
        free(z);
        free(px);
        free(py);
        free(pz);

        if (rc)
        {
            fprintf(stderr, "particle write failed rc=%d\n", rc);
            return 6;
        }

        printf("Wrote RAW-species_1-000000.h5\n");
    }

    return 0;
}
