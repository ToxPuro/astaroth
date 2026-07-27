#pragma once
// Header for accessing Astaroth internals

#include <mpi.h>

#include "func_define.h"

AC_BEGIN_C_DECLARATIONS

/**
Returns the MPI communicator used by all Astaroth processes.

If MPI was initialized with MPI_Init* instead of ac_MPI_Init, this will return MPI_COMM_WORLD
 */
//FUNC_DEFINE(MPI_Comm, acGridMPIComm,());

AC_END_C_DECLARATIONS
