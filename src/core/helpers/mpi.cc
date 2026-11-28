#include "astaroth_helpers.h"
#include "ac_helpers.h"
#include "ac_mpi.h"
#include "astaroth_grid.h"

int
ac_mpi_rank(const AcMeshInfo info)
{
    ERRCHK_ALWAYS(info.comm != NULL);
    if(info.comm->handle == MPI_COMM_NULL)
    {
        if(acGridInitialized()) return ac_MPI_Comm_rank();
    }
    int pid;
    MPI_Comm_rank(info.comm->handle, &pid);
    return pid;
}

bool
ac_mpi_root(const AcMeshInfo info)
{
	return ac_mpi_rank(info) == 0;
}

void
ac_mpi_barrier(const AcMeshInfo info)
{
   ERRCHK_ALWAYS(info.comm != NULL);
   MPI_Barrier(info.comm->handle);
}

void
ac_mpi_bcast(int* res, const AcMeshInfo info)
{
   ERRCHK_ALWAYS(info.comm != NULL);
   MPI_Bcast(res, 1, MPI_INT, 0, info.comm->handle);
}
