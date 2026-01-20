#include <stdio.h>
#include <mpi.h>
#include "driver.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank=0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc != 3) {
        if (rank == 0)
            fprintf(stderr, "usage: %s <input.toml> <field_cache_prefix>\n", argv[0]);
        MPI_Finalize();
        return 2;
    }

    int rc = run_from_inputdeck(argv[1], argv[2], MPI_COMM_WORLD);

    MPI_Finalize();
    return rc;
}

