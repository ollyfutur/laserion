#ifndef DRIVER_H
#define DRIVER_H

#include <mpi.h>

int run_from_inputdeck(const char *toml_path,
                       const char *field_cache_prefix,
                       MPI_Comm comm);

#endif

