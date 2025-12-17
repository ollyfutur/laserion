#include "diag_h5.h"
#include <stdio.h>
#include <string.h>
#include <hdf5.h>

static int write_attr_str(hid_t obj, const char *name, const char *value)
{
    /* OSIRIS stores fixed-length strings (often 256). We can do fixed 256 too. */
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

static int write_attr_f64(hid_t obj, const char *name, double v)
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

static int write_attr_i32(hid_t obj, const char *name, int v)
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

static int write_root_attrs(hid_t file,
                            const char *dataset_name,
                            const char *units,
                            double time_value,
                            int iter_value)
{
    if (write_attr_str(file, "NAME", dataset_name))
        return 1;
    if (write_attr_str(file, "TYPE", "grid"))
        return 2;
    if (write_attr_f64(file, "TIME", time_value))
        return 3;
    if (write_attr_i32(file, "ITER", iter_value))
        return 4;
    if (write_attr_f64(file, "DT", 0.0))
        return 5;
    if (write_attr_str(file, "TIME UNITS", "1/\\omega_p"))
        return 6; /* exactly as requested */
    if (write_attr_str(file, "UNITS", units))
        return 7;
    if (write_attr_str(file, "LABEL", dataset_name))
        return 8;
    return 0;
}

static int write_axis_one(hid_t parent,
                          const char *dset_name,
                          const DiagAxis1D *a)
{
    hsize_t dims[1] = {2};
    hid_t space = H5Screate_simple(1, dims, NULL);
    if (space < 0)
        return 10;

    hid_t dset = H5Dcreate2(parent, dset_name,
                            H5T_IEEE_F64LE,
                            space,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dset < 0)
    {
        H5Sclose(space);
        return 11;
    }

    double v[2] = {a->vmin, a->vmax};
    herr_t st = H5Dwrite(dset,
                         H5T_NATIVE_DOUBLE,
                         H5S_ALL, H5S_ALL,
                         H5P_DEFAULT,
                         v);
    if (st < 0)
    {
        H5Dclose(dset);
        H5Sclose(space);
        return 12;
    }

    if (write_attr_str(dset, "TYPE", "linear"))
        goto fail;
    if (write_attr_str(dset, "UNITS", a->units))
        goto fail;
    if (write_attr_str(dset, "NAME", a->name))
        goto fail;
    if (write_attr_str(dset, "LONG_NAME", a->long_name))
        goto fail;

    H5Dclose(dset);
    H5Sclose(space);
    return 0;

fail:
    H5Dclose(dset);
    H5Sclose(space);
    return 13;
}

static int write_axis_group(hid_t file,
                            const DiagAxis1D *a1,
                            const DiagAxis1D *a2_or_null)
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

int diag_h5_write_field_1d(const char *path,
                           const char *dataset_name,
                           const char *units,
                           double time_value,
                           int iter_value,
                           const float *data,
                           size_t n,
                           const DiagAxis1D *axis1)
{
    hid_t f = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (f < 0)
        return 1;

    if (write_root_attrs(f, dataset_name, units, time_value, iter_value))
    {
        H5Fclose(f);
        return 2;
    }

    /* dataset at root */
    hsize_t dims[1] = {(hsize_t)n};
    hid_t space = H5Screate_simple(1, dims, NULL);
    if (space < 0)
    {
        H5Fclose(f);
        return 3;
    }

    hid_t dset = H5Dcreate2(f, dataset_name, H5T_IEEE_F32LE, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
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

int diag_h5_write_field_2d(const char *path,
                           const char *dataset_name,
                           const char *units,
                           double time_value,
                           int iter_value,
                           const float *data,
                           size_t n1,
                           size_t n2,
                           const DiagAxis1D *axis1,
                           const DiagAxis1D *axis2)
{
    hid_t f = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (f < 0)
        return 1;

    if (write_root_attrs(f, dataset_name, units, time_value, iter_value))
    {
        H5Fclose(f);
        return 2;
    }

    /* dataset at root: shape [n2, n1] */
    hsize_t dims[2] = {(hsize_t)n2, (hsize_t)n1};
    hid_t space = H5Screate_simple(2, dims, NULL);
    if (space < 0)
    {
        H5Fclose(f);
        return 3;
    }

    hid_t dset = H5Dcreate2(f, dataset_name, H5T_IEEE_F32LE, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
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
