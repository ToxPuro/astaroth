#include "astaroth_helpers.h"
#include "ac_helpers.h"
#include "ac_mpi.h"

int
ac_mpi_rank(const AcMeshInfo)
{
	return 0;
}

bool
ac_mpi_root(const AcMeshInfo info)
{
	return ac_mpi_rank(info) == 0;
}
