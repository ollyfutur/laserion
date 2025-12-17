#pragma once
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const char *name;       // "x","y","z","t"
    const char *long_name;  // "x","y","z","t" (or "x_1" style if you prefer)
    const char *units;      // "\\mu m" or "fs"
    double vmin;
    double vmax;
} DiagAxis1D;

/* Writes a 1D OSIRIS-like minimal HDF5:
 *  / (root attrs per your spec)
 *  /<dataset_name>           float32 [n]
 *  /AXIS/AXIS1               float64 [2] = [vmin,vmax] + attrs
 */
int diag_h5_write_field_1d(const char *path,
                           const char *dataset_name,   // "ex","ey","ez","ax","ay","az"
                           const char *units,          // "GV/m" or "GV/m fs"
                           double time_value,          // TIME attr
                           int iter_value,             // ITER attr
                           const float *data,
                           size_t n,
                           const DiagAxis1D *axis1);

/* Writes a 2D OSIRIS-like minimal HDF5:
 *  /<dataset_name>           float32 [n2,n1] row-major (C order)
 *  /AXIS/AXIS1, /AXIS/AXIS2  float64 [2]
 */
int diag_h5_write_field_2d(const char *path,
                           const char *dataset_name,
                           const char *units,
                           double time_value,
                           int iter_value,
                           const float *data,          // size n2*n1
                           size_t n1,                  // fastest axis (AXIS1)
                           size_t n2,                  // slow axis (AXIS2)
                           const DiagAxis1D *axis1,
                           const DiagAxis1D *axis2);

#ifdef __cplusplus
}
#endif

