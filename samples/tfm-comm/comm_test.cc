#include <cstdlib>
#include <iostream>

#include "errchk_mpi.h"
#include <mpi.h>

int
main(void)
{
    ERRCHK_MPI_API(MPI_Init(NULL, NULL));
    try {
    }
    catch (std::exception& e) {
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    ERRCHK_MPI_API(MPI_Finalize());
    return EXIT_SUCCESS;
}
