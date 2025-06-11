#include "astaroth.h"
#include "ac_helpers.h"
#include <sys/resource.h>
const char*
acLibraryVersion(const char* library, const int counter, const AcMeshInfo info)
{
	if(counter == 0) return library;
	static char new_library[40000];
	sprintf(new_library,"%s_v%d",library,counter);
	int pid = 0;
#if AC_MPI_ENABLED
	ERRCHK_ALWAYS(info.comm != NULL && info.comm->handle != MPI_COMM_NULL);
	MPI_Comm_rank(info.comm->handle,&pid);
#endif
	if(pid == 0)
	{
		char cmd[60000];
		sprintf(cmd,"cp %s %s",library,new_library);
		int ret = system(cmd);
		if(ret != 0)
		{
			fprintf(stderr,"Was not able to generate new version number of library: %s!\n",new_library);
		}
	}
#if AC_MPI_ENABLED
	MPI_Barrier(info.comm->handle);
#endif
	return new_library;
}

size_t
acGetSizeFromDim(const int dim, const Volume dims)
{
    	const auto size   = dim == X_ORDER_INT ? dims.x :
        		    dim == Y_ORDER_INT ? dims.y :
        		    dims.z;
        return size;
}

Volume
acGetVolumeFromShape(const AcShape shape)
{
	return {shape.x,shape.y,shape.z};
}

int acMemUsage()
{
	struct rusage usage;
	int res=getrusage(RUSAGE_SELF,&usage);
	ERRCHK_ALWAYS(res == 0);

	return usage.ru_maxrss;
}
