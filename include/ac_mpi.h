#pragma once

#include <mpi.h>

struct AcCommunicator
{
	MPI_Comm handle;
};

typedef struct AcSubCommunicators {
	MPI_Comm all;
	MPI_Comm x;
	MPI_Comm y;
	MPI_Comm z;

	MPI_Comm xy;
	MPI_Comm xz;
	MPI_Comm yz;
} AcSubCommunicators;
