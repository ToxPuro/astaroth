#pragma once

#include "func_define.h"
#include "host_datatypes.h"

#if AC_MPI_ENABLED && __has_include(<mpi.h>)
#include <mpi.h>
#endif

//TP: opaque pointer for the MPI comm to enable having the opaque type in modules which do not about MPI_Comm
typedef struct AcCommunicator AcCommunicator;

#if AC_MPI_ENABLED && __has_include(<mpi.h>)

struct AcCommunicator {
    MPI_Comm handle;
};

typedef struct AcSubCommunicators {
    MPI_Comm all;
    MPI_Comm x;
    MPI_Comm y;
    MPI_Comm z;

    MPI_Comm reverse_x;
    MPI_Comm reverse_y;
    MPI_Comm reverse_z;

    MPI_Comm xy;
    MPI_Comm xz;
    MPI_Comm yz;
} AcSubCommunicators;

#else

struct AcCommunicator {
    // OM: Padding is necessary because C and C++ both make structs a different size if they are
    // left empty.
    void* padding[8];
};

#endif
