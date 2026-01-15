#ifndef MDF_PARTICLES_H
#define MDF_PARTICLES_H

#include <stddef.h>
#include <mpi.h>

#include "laser.h"
#include "mdf.h"
#include "ionization_model.h"

/*
 * Generate particles by sampling the MDF in a spatial region discretized
 * into an (nx,ny,nz) grid, with ppc particles per cell.
 *
 * Output is a single "particles" HDF5 file written by the root rank using
 * diag_h5_write_particles() (datasets: ene, p1,p2,p3, x1,x2,x3, q).
 */

typedef struct MDFParticlesOptions
{
    /* Species / ionization setup (same meaning as MDF_build) */
    const char *species;
    const int  *Z_list;
    size_t      nZ;
    const IonizationModel *ion_model;
    double      envelope_cut;

    /* Time window for MDF_build (fs) */
    double tmin_fs;
    double tmax_fs;
    double dt_fs;

    /* Spatial region in microns and spatial grid resolution */
    double xmin, xmax;
    double ymin, ymax;
    double zmin, zmax;
    int nx, ny, nz;

    /* Particles per spatial cell */
    int ppc;

    /*
     * If 0: build MDF once per cell at the cell center and sample p from it.
     * If 1: build MDF independently for each particle position inside the cell.
     *       (More accurate, significantly more expensive.)
     */
    int mdf_at_particle_position;

    /* Output naming */
    const char *dataset_name;  /* e.g. "electrons" */
    const char *file_suffix;   /* e.g. "_particles.h5" */

    /* Particle mapping for diag_h5_write_particles (your default is z,x,y and pz,px,py) */
    int particle_map;          /* use DiagParticleMap values */

    /* MPI and RNG */
    int root_rank;             /* default 0 if <0 */
    unsigned long long seed;   /* 0 -> auto-seed from rank */
} MDFParticlesOptions;

int mdf_particles_run(const LaserPulse *pulse,
                      const MDFParticlesOptions *opt,
                      const char *path_prefix,
                      MPI_Comm comm);

#endif /* MDF_PARTICLES_H */

