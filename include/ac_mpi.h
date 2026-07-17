#pragma once

#if AC_MPI_ENABLED

#define AC_MPI_H

#include <mpi.h>

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
