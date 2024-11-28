#pragma once
// Header for accessing Astaroth internals

#include <mpi.h>

/**
Returns the MPI communicator used by all Astaroth processes.

If MPI was initialized with MPI_Init* instead of ac_MPI_Init, this will return MPI_COMM_WORLD
 */
MPI_Comm acGridMPIComm();
