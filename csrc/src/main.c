#include <stdio.h>
#include <mpi.h>
#include <unistd.h> /* access() */
#include "run.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const char *input = NULL;

    if (argc == 2)
    {
        input = argv[1];
    }
    else if (argc == 1)
    {
        /* default input */
        if (access("input.toml", R_OK) == 0)
        {
            input = "input.toml";
        }
        else
        {
            if (rank == 0)
                fprintf(stderr, "usage: %s <input.toml>\n", argv[0]);

            MPI_Finalize();
            return 2;
        }
    }
    else
    {
        if (rank == 0)
            fprintf(stderr, "usage: %s <input.toml>\n", argv[0]);

        MPI_Finalize();
        return 2;
    }

    int rc = run_from_inputdeck(input, MPI_COMM_WORLD);

    MPI_Finalize();
    return rc;
}
