#ifndef DRIVER_H
#define DRIVER_H

#include <mpi.h>

int run_from_inputdeck(const char *toml_path, MPI_Comm comm);

#endif
