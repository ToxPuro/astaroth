#include "astaroth_helpers.h"
#include "ac_helpers.h"
#include "ac_mpi.h"

int
ac_mpi_rank(const AcMeshInfo info)
{
    ERRCHK_ALWAYS(info.comm != NULL);
    int pid;
    MPI_Comm_rank(info.comm->handle, &pid);
    return pid;
}

bool
ac_mpi_root(const AcMeshInfo info)
{
	return ac_mpi_rank(info) == 0;
}
