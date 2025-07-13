#include "host_datatypes.h"
#include "device_headers.h"
#include "ac_helpers.h"
#include <sys/resource.h>
#include "astaroth_cuda_wrappers.h"
#include <stdio.h>
#include <math.h>
#include "errchk.h"
#if AC_MPI_ENABLED
#include <mpi.h>
struct AcCommunicator
{
	MPI_Comm handle;
};
#endif
const char*
acLibraryVersion(const char* library, const int counter, const AcCommunicator* comm)
{
	if(counter == 0) return library;
	static char new_library[40000];
	sprintf(new_library,"%s_v%d",library,counter);
	int pid = 0;
#if AC_MPI_ENABLED
	ERRCHK_ALWAYS(comm != NULL && comm->handle != MPI_COMM_NULL);
	MPI_Comm_rank(comm->handle,&pid);
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
	MPI_Barrier(comm->handle);
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

size_t
acGetAmountOfDeviceMemoryFree()
{
	size_t free_mem, total_mem;
	ERRCHK_CUDA_ALWAYS(acMemGetInfo(&free_mem,&total_mem));
	return free_mem;
}

int3
ceil(AcReal3 a)
{
	return (int3){(int)ceil(a.x), (int)ceil(a.y), (int)ceil(a.z)};
}

size3_t
ceil_div(const size3_t& a, const int3& b)
{
	const int3 factors = ceil((AcReal3){(AcReal)a.x/(AcReal)b.x, (AcReal)a.y/(AcReal)b.y, (AcReal)a.z/(AcReal)b.z});
	return (size3_t){(unsigned int)factors.x,(unsigned int)factors.y,(unsigned int)factors.z};
}

size3_t
ceil_div(const size3_t& a, const size3_t& b)
{
	const int3 factors = ceil((AcReal3){(AcReal)a.x/(AcReal)b.x, (AcReal)a.y/(AcReal)b.y, (AcReal)a.z/(AcReal)b.z});
	return (size3_t){(unsigned int)factors.x,(unsigned int)factors.y,(unsigned int)factors.z};
}
int3
ceil_div(const int3& a, const int3& b)
{
	const int3 factors = ceil((AcReal3){(AcReal)a.x/(AcReal)b.x, (AcReal)a.y/(AcReal)b.y, (AcReal)a.z/(AcReal)b.z});
	return (int3){factors.x,factors.y,factors.z};
}

size_t
ceil_div(const size_t& a, const size_t& b)
{
	return (size_t)ceil((AcReal)(1. * a) / b);
}

cudaDeviceProp
get_device_prop()
{
  cudaDeviceProp props;
  (void)acGetDeviceProperties(&props, 0);
  return props;
}

Volume
get_bpg(Volume dims, const Volume tpb)
{
  //TP: but dependency on implementation on comment for now
  //    since not used and enables this to be compiled in helpers
  /**
  switch (IMPLEMENTATION) {
  case IMPLICIT_CACHING:             // Fallthrough
  case EXPLICIT_CACHING:             // Fallthrough
  case EXPLICIT_CACHING_3D_BLOCKING: // Fallthrough
  case EXPLICIT_CACHING_4D_BLOCKING: // Fallthrough
  case EXPLICIT_PINGPONG_txw:        // Fallthrough
  case EXPLICIT_PINGPONG_txy:        // Fallthrough
  case EXPLICIT_PINGPONG_txyblocked: // Fallthrough
  case EXPLICIT_PINGPONG_txyz:       // Fallthrough
  case EXPLICIT_ROLLING_PINGPONG: {
  **/
  return (Volume){
      as_size_t(ceil_div(dims.x,tpb.x)),
      as_size_t(ceil_div(dims.y,tpb.y)),
      as_size_t(ceil_div(dims.z,tpb.z)),
  };
  /**
  }
  default: {
    ERROR("Invalid IMPLEMENTATION in get_bpg");
    return (Volume){0, 0, 0};
  }
  **/
}

AcMeshOrder
acGetMeshOrderForProfile(const AcProfileType type)
{
    	switch(type)
    	{
    	        case(PROFILE_X):
    	    	    return ZYX;
    	        case(PROFILE_Y):
		    return XZY;
    	        case(PROFILE_Z):
			return XYZ;
    	        case(PROFILE_XY):
			return ZXY;
    	        case(PROFILE_XZ):
			return YXZ;
    	        case(PROFILE_YX):
			return ZYX;
    	        case(PROFILE_YZ):
			return XYZ;
    	        case(PROFILE_ZX):
			return YZX;
    	        case(PROFILE_ZY):
			return XZY;
		case(PROFILE_NONE):
			return XYZ;
    	}
	return XYZ;
};

size_t
acShapeSize(const AcShape shape)
{
  return shape.x * shape.y * shape.z * shape.w;
}


