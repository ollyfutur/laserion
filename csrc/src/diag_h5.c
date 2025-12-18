/* diag_h5.c */
#include "diag_h5.h"

#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <hdf5.h>

/* ---------------- helpers ---------------- */

static const char *axis_id_name(DiagAxisID id)
{
    switch (id)
    {
    case DIAG_AXIS_T:
        return "t";
    case DIAG_AXIS_X:
        return "x";
    case DIAG_AXIS_Y:
        return "y";
    case DIAG_AXIS_Z:
        return "z";
    default:
        return "";
    }
}

static int write_attr_str_scalar(hid_t obj, const char *name, const char *value)
{
    hid_t atype = H5Tcopy(H5T_C_S1);
    if (atype < 0)
        return 1;
    H5Tset_size(atype, 256);
    H5Tset_strpad(atype, H5T_STR_NULLTERM);

    hid_t aspace = H5Screate(H5S_SCALAR);
    if (aspace < 0)
    {
        H5Tclose(atype);
        return 2;
    }

    hid_t attr = H5Acreate2(obj, name, atype, aspace, H5P_DEFAULT, H5P_DEFAULT);
    if (attr < 0)
    {
        H5Sclose(aspace);
        H5Tclose(atype);
        return 3;
    }

    char buf[256];
    memset(buf, 0, sizeof(buf));
    strncpy(buf, value ? value : "", sizeof(buf) - 1);

    herr_t st = H5Awrite(attr, atype, buf);

    H5Aclose(attr);
    H5Sclose(aspace);
    H5Tclose(atype);
    return (st < 0) ? 4 : 0;
}

static int write_attr_f64_scalar(hid_t obj, const char *name, double v)
{
    hid_t aspace = H5Screate(H5S_SCALAR);
    if (aspace < 0)
        return 1;
    hid_t attr = H5Acreate2(obj, name, H5T_IEEE_F64LE, aspace, H5P_DEFAULT, H5P_DEFAULT);
    if (attr < 0)
    {
        H5Sclose(aspace);
        return 2;
    }
    herr_t st = H5Awrite(attr, H5T_NATIVE_DOUBLE, &v);
    H5Aclose(attr);
    H5Sclose(aspace);
    return (st < 0) ? 3 : 0;
}

static int write_attr_i32_scalar(hid_t obj, const char *name, int v)
{
    hid_t aspace = H5Screate(H5S_SCALAR);
    if (aspace < 0)
        return 1;
    hid_t attr = H5Acreate2(obj, name, H5T_STD_I32LE, aspace, H5P_DEFAULT, H5P_DEFAULT);
    if (attr < 0)
    {
        H5Sclose(aspace);
        return 2;
    }
    herr_t st = H5Awrite(attr, H5T_NATIVE_INT, &v);
    H5Aclose(attr);
    H5Sclose(aspace);
    return (st < 0) ? 3 : 0;
}

static int write_attr_str_array(hid_t obj, const char *name, const char **values, size_t n)
{
    hid_t atype = H5Tcopy(H5T_C_S1);
    if (atype < 0)
        return 1;
    H5Tset_size(atype, 256);
    H5Tset_strpad(atype, H5T_STR_NULLTERM);

    hsize_t dims[1] = {(hsize_t)n};
    hid_t aspace = H5Screate_simple(1, dims, NULL);
    if (aspace < 0)
    {
        H5Tclose(atype);
        return 2;
    }

    hid_t attr = H5Acreate2(obj, name, atype, aspace, H5P_DEFAULT, H5P_DEFAULT);
    if (attr < 0)
    {
        H5Sclose(aspace);
        H5Tclose(atype);
        return 3;
    }

    char *buf = (char *)malloc(n * 256);
    if (!buf)
    {
        H5Aclose(attr);
        H5Sclose(aspace);
        H5Tclose(atype);
        return 4;
    }
    memset(buf, 0, n * 256);

    for (size_t i = 0; i < n; ++i)
    {
        const char *s = (values && values[i]) ? values[i] : "";
        strncpy(buf + i * 256, s, 255);
    }

    herr_t st = H5Awrite(attr, atype, buf);
    free(buf);

    H5Aclose(attr);
    H5Sclose(aspace);
    H5Tclose(atype);

    return (st < 0) ? 5 : 0;
}

static int write_attr_f64_array(hid_t obj, const char *name, const double *values, size_t n)
{
    hsize_t dims[1] = {(hsize_t)n};
    hid_t aspace = H5Screate_simple(1, dims, NULL);
    if (aspace < 0)
        return 1;

    hid_t attr = H5Acreate2(obj, name, H5T_IEEE_F64LE, aspace, H5P_DEFAULT, H5P_DEFAULT);
    if (attr < 0)
    {
        H5Sclose(aspace);
        return 2;
    }

    herr_t st = H5Awrite(attr, H5T_NATIVE_DOUBLE, values);

    H5Aclose(attr);
    H5Sclose(aspace);
    return (st < 0) ? 3 : 0;
}

/* Root attrs (grid). */
static int write_root_attrs_grid(hid_t file,
                                 const char *dataset_name,
                                 const char *units,
                                 const char *label,
                                 double time_value,
                                 int iter_value)
{
    if (write_attr_str_scalar(file, "NAME", dataset_name ? dataset_name : ""))
        return 1;
    if (write_attr_str_scalar(file, "TYPE", "grid"))
        return 2;
    if (write_attr_f64_scalar(file, "TIME", time_value))
        return 3;
    if (write_attr_i32_scalar(file, "ITER", iter_value))
        return 4;
    if (write_attr_f64_scalar(file, "DT", 0.0))
        return 5;
    if (write_attr_str_scalar(file, "TIME UNITS", "1/\\omega_p"))
        return 6;
    if (write_attr_str_scalar(file, "UNITS", units ? units : ""))
        return 7;
    if (write_attr_str_scalar(file, "LABEL", label ? label : ""))
        return 8;
    return 0;
}

/* Root attrs (particles). */
static int write_root_attrs_particles(hid_t file,
                                      const char *species_name,
                                      double time_value,
                                      int iter_value,
                                      const DiagParticleQuant *quants,
                                      size_t nquants)
{
    if (write_attr_str_scalar(file, "NAME", species_name ? species_name : ""))
        return 1;
    if (write_attr_str_scalar(file, "TYPE", "particles"))
        return 2;
    if (write_attr_f64_scalar(file, "TIME", time_value))
        return 3;
    if (write_attr_i32_scalar(file, "ITER", iter_value))
        return 4;

    /* Keep this attribute as in your reference file (not fs). */
    if (write_attr_str_scalar(file, "TIME UNITS", "1 / \\omega_p"))
        return 5;

    char **q_quants = (char **)calloc(nquants, sizeof(char *));
    char **q_labels = (char **)calloc(nquants, sizeof(char *));
    char **q_units = (char **)calloc(nquants, sizeof(char *));
    double *offsets = (double *)calloc(nquants, sizeof(double));
    if (!q_quants || !q_labels || !q_units || !offsets)
    {
        free(q_quants);
        free(q_labels);
        free(q_units);
        free(offsets);
        return 6;
    }

    for (size_t i = 0; i < nquants; ++i)
    {
        q_quants[i] = (char *)(quants[i].quant ? quants[i].quant : "");
        q_labels[i] = (char *)(quants[i].label ? quants[i].label : "");
        q_units[i] = (char *)(quants[i].units ? quants[i].units : "");
        offsets[i] = quants[i].offset_t;
    }

    int rc = 0;
    if (write_attr_str_array(file, "QUANTS", (const char **)q_quants, nquants))
        rc = 7;
    else if (write_attr_str_array(file, "LABELS", (const char **)q_labels, nquants))
        rc = 8;
    else if (write_attr_str_array(file, "UNITS", (const char **)q_units, nquants))
        rc = 9;
    else if (write_attr_f64_array(file, "OFFSET_T", offsets, nquants))
        rc = 10;

    free(q_quants);
    free(q_labels);
    free(q_units);
    free(offsets);
    return rc;
}

static int write_axis_one(hid_t parent, const char *dset_name, const DiagAxis *a)
{
    hsize_t dims[1] = {2};
    hid_t space = H5Screate_simple(1, dims, NULL);
    if (space < 0)
        return 10;

    hid_t dset = H5Dcreate2(parent, dset_name, H5T_IEEE_F64LE, space,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dset < 0)
    {
        H5Sclose(space);
        return 11;
    }

    double v[2] = {a->vmin, a->vmax};
    herr_t st = H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, v);
    if (st < 0)
    {
        H5Dclose(dset);
        H5Sclose(space);
        return 12;
    }

    if (write_attr_str_scalar(dset, "TYPE", "linear"))
        goto fail;
    if (write_attr_str_scalar(dset, "UNITS", a->units ? a->units : ""))
        goto fail;

    const char *nm = axis_id_name(a->id);
    if (write_attr_str_scalar(dset, "NAME", nm))
        goto fail;
    if (write_attr_str_scalar(dset, "LONG_NAME", a->long_name ? a->long_name : nm))
        goto fail;

    H5Dclose(dset);
    H5Sclose(space);
    return 0;

fail:
    H5Dclose(dset);
    H5Sclose(space);
    return 13;
}

static int write_axis_group(hid_t file, const DiagAxis *a1, const DiagAxis *a2_or_null)
{
    hid_t g = H5Gcreate2(file, "/AXIS", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (g < 0)
        return 1;

    int rc = write_axis_one(g, "AXIS1", a1);
    if (rc)
    {
        H5Gclose(g);
        return 2;
    }

    if (a2_or_null)
    {
        rc = write_axis_one(g, "AXIS2", a2_or_null);
        if (rc)
        {
            H5Gclose(g);
            return 3;
        }
    }

    H5Gclose(g);
    return 0;
}

/* ---------------- grid public writers ---------------- */

int diag_h5_write_grid_1d(const char *path,
                          const char *dataset_name,
                          const char *units,
                          const char *label,
                          double time_value,
                          int iter_value,
                          const float *data,
                          size_t n,
                          const DiagAxis *axis1,
                          const DiagFixedCoords *fixed)
{
    (void)fixed;

    hid_t f = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (f < 0)
        return 1;

    if (write_root_attrs_grid(f, dataset_name, units, label, time_value, iter_value))
    {
        H5Fclose(f);
        return 2;
    }

    hsize_t dims[1] = {(hsize_t)n};
    hid_t space = H5Screate_simple(1, dims, NULL);
    if (space < 0)
    {
        H5Fclose(f);
        return 3;
    }

    hid_t dset = H5Dcreate2(f, dataset_name, H5T_IEEE_F32LE, space,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dset < 0)
    {
        H5Sclose(space);
        H5Fclose(f);
        return 4;
    }

    herr_t st = H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);

    H5Dclose(dset);
    H5Sclose(space);

    if (st < 0)
    {
        H5Fclose(f);
        return 5;
    }

    if (write_axis_group(f, axis1, NULL))
    {
        H5Fclose(f);
        return 6;
    }

    H5Fclose(f);
    return 0;
}

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
                          const DiagFixedCoords *fixed)
{
    (void)fixed;

    hid_t f = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (f < 0)
        return 1;

    if (write_root_attrs_grid(f, dataset_name, units, label, time_value, iter_value))
    {
        H5Fclose(f);
        return 2;
    }

    hsize_t dims[2] = {(hsize_t)n2, (hsize_t)n1};
    hid_t space = H5Screate_simple(2, dims, NULL);
    if (space < 0)
    {
        H5Fclose(f);
        return 3;
    }

    hid_t dset = H5Dcreate2(f, dataset_name, H5T_IEEE_F32LE, space,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dset < 0)
    {
        H5Sclose(space);
        H5Fclose(f);
        return 4;
    }

    herr_t st = H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);

    H5Dclose(dset);
    H5Sclose(space);

    if (st < 0)
    {
        H5Fclose(f);
        return 5;
    }

    if (write_axis_group(f, axis1, axis2))
    {
        H5Fclose(f);
        return 6;
    }

    H5Fclose(f);
    return 0;
}

/* ---------------- particle writer ---------------- */

static int write_dataset_f32_1d(hid_t file, const char *name, const float *data, size_t n)
{
    hsize_t dims[1] = {(hsize_t)n};
    hid_t space = H5Screate_simple(1, dims, NULL);
    if (space < 0)
        return 1;

    hid_t dset = H5Dcreate2(file, name, H5T_IEEE_F32LE, space,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dset < 0)
    {
        H5Sclose(space);
        return 2;
    }

    herr_t st = H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);

    H5Dclose(dset);
    H5Sclose(space);
    return (st < 0) ? 3 : 0;
}

static void map_particle_arrays(DiagParticleMap map,
                                const DiagParticleData *p,
                                const float **x1, const float **x2, const float **x3,
                                const float **p1, const float **p2, const float **p3)
{
    /* File datasets: x1,x2,x3 and p1,p2,p3 */
    if (map == DIAG_PARTICLE_MAP_XYZ_PXPYPZ)
    {
        *x1 = p->x;
        *x2 = p->y;
        *x3 = p->z;
        *p1 = p->px;
        *p2 = p->py;
        *p3 = p->pz;
    }
    else
    {
        /* Default requested: x1<-z, x2<-x, x3<-y, p1<-pz, p2<-px, p3<-py */
        *x1 = p->z;
        *x2 = p->x;
        *x3 = p->y;
        *p1 = p->pz;
        *p2 = p->px;
        *p3 = p->py;
    }
}

static int compute_ene_from_p(float *ene_out, size_t n,
                              const float *p1, const float *p2, const float *p3)
{
    for (size_t i = 0; i < n; ++i)
    {
        const double a = (double)p1[i];
        const double b = (double)p2[i];
        const double c = (double)p3[i];
        const double g = sqrt(a * a + b * b + c * c + 1.0);
        ene_out[i] = (float)(g - 1.0);
    }
    return 0;
}

int diag_h5_write_particles(const char *path,
                            const DiagRequest *req,
                            const DiagParticleData *pdata)
{
    if (!req || !pdata)
        return 1;
    if (req->kind != DIAG_KIND_PARTICLE)
        return 2;
    if (pdata->n == 0)
        return 3;

    /* Default schema matching your expectations, with requested units and labels. */
    const DiagParticleQuant default_q[8] = {
        /* order can be arbitrary; must be consistent across QUANTS/LABELS/UNITS/OFFSET_T */
        {.quant = "ene", .label = "\\epsilon_k", .units = "m_e c^2", .offset_t = 0.0},
        {.quant = "p1", .label = "p_z", .units = "m_e c", .offset_t = 0.0},
        {.quant = "p2", .label = "p_x", .units = "m_e c", .offset_t = 0.0},
        {.quant = "p3", .label = "p_y", .units = "m_e c", .offset_t = 0.0},
        {.quant = "x1", .label = "z", .units = "\\mu m", .offset_t = 0.0},
        {.quant = "x2", .label = "x", .units = "\\mu m", .offset_t = 0.0},
        {.quant = "x3", .label = "y", .units = "\\mu m", .offset_t = 0.0},
        {.quant = "q", .label = "q", .units = "e", .offset_t = 0.0},
    };

    const DiagParticleQuant *quants = req->u.particle.quants ? req->u.particle.quants : default_q;
    const size_t nquants = req->u.particle.quants ? req->u.particle.nquants : 8;

    const DiagParticleMap map = req->u.particle.map;

    hid_t f = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (f < 0)
        return 4;

    if (write_root_attrs_particles(f,
                                   req->dataset_name,
                                   req->time_value,
                                   req->iter_value,
                                   quants,
                                   nquants))
    {
        H5Fclose(f);
        return 5;
    }

    /* Map physical arrays to file datasets. */
    const float *x1 = NULL, *x2 = NULL, *x3 = NULL;
    const float *p1 = NULL, *p2 = NULL, *p3 = NULL;
    map_particle_arrays(map, pdata, &x1, &x2, &x3, &p1, &p2, &p3);

    /* Validate required pointers. */
    if (!x1 || !x2 || !x3 || !p1 || !p2 || !p3)
    {
        H5Fclose(f);
        return 6;
    }

    /* q: if not provided, write -1.0 for all particles. */
    float *qtmp = NULL;
    const float *qptr = pdata->q;
    if (!qptr)
    {
        qtmp = (float *)malloc(pdata->n * sizeof(float));
        if (!qtmp)
        {
            H5Fclose(f);
            return 7;
        }
        for (size_t i = 0; i < pdata->n; ++i)
            qtmp[i] = -1.0f;
        qptr = qtmp;
    }

    /* ene: if not provided, compute kinetic energy from mapped p1/p2/p3. */
    float *enetmp = NULL;
    const float *eneptr = pdata->ene;
    if (!eneptr)
    {
        enetmp = (float *)malloc(pdata->n * sizeof(float));
        if (!enetmp)
        {
            free(qtmp);
            H5Fclose(f);
            return 8;
        }
        compute_ene_from_p(enetmp, pdata->n, p1, p2, p3);
        eneptr = enetmp;
    }

    /* Write datasets. (Names must match your existing tools exactly.) */
    int rc = 0;
    rc = write_dataset_f32_1d(f, "ene", eneptr, pdata->n);
    if (rc)
    {
        rc = 20;
        goto done;
    }
    rc = write_dataset_f32_1d(f, "p1", p1, pdata->n);
    if (rc)
    {
        rc = 21;
        goto done;
    }
    rc = write_dataset_f32_1d(f, "p2", p2, pdata->n);
    if (rc)
    {
        rc = 22;
        goto done;
    }
    rc = write_dataset_f32_1d(f, "p3", p3, pdata->n);
    if (rc)
    {
        rc = 23;
        goto done;
    }
    rc = write_dataset_f32_1d(f, "x1", x1, pdata->n);
    if (rc)
    {
        rc = 24;
        goto done;
    }
    rc = write_dataset_f32_1d(f, "x2", x2, pdata->n);
    if (rc)
    {
        rc = 25;
        goto done;
    }
    rc = write_dataset_f32_1d(f, "x3", x3, pdata->n);
    if (rc)
    {
        rc = 26;
        goto done;
    }
    rc = write_dataset_f32_1d(f, "q", qptr, pdata->n);
    if (rc)
    {
        rc = 27;
        goto done;
    }

done:
    free(qtmp);
    free(enetmp);
    H5Fclose(f);
    return rc;
}

/* ---------------- dispatcher ---------------- */

int diag_h5_write(const char *path, const DiagRequest *req, const void *data)
{
    if (!req || !path)
        return 1;

    switch (req->kind)
    {
    case DIAG_KIND_GRID_1D:
        return diag_h5_write_grid_1d(path,
                                     req->dataset_name,
                                     req->units,
                                     req->label,
                                     req->time_value,
                                     req->iter_value,
                                     (const float *)data,
                                     req->u.grid1d.n1,
                                     &req->u.grid1d.axis1,
                                     &req->u.grid1d.fixed);

    case DIAG_KIND_GRID_2D:
        return diag_h5_write_grid_2d(path,
                                     req->dataset_name,
                                     req->units,
                                     req->label,
                                     req->time_value,
                                     req->iter_value,
                                     (const float *)data,
                                     req->u.grid2d.n1,
                                     req->u.grid2d.n2,
                                     &req->u.grid2d.axis1,
                                     &req->u.grid2d.axis2,
                                     &req->u.grid2d.fixed);

    case DIAG_KIND_PARTICLE:
        return diag_h5_write_particles(path, req, (const DiagParticleData *)data);

    default:
        return 2;
    }
}
