#include "host_datatypes.h"
#include "device_headers.h"
#include "device_details.h"
#include "ac_helpers.h"
#include "transpose.h"

#include "astaroth_cuda_wrappers.h"
#include "errchk.h"
#include "math_utils_base.h"

template <typename T>
void
transpose_base(const AcReal* src, AcReal* dst, const Volume dims, const Volume start, const Volume end, const T TRANSPOSED_IDX)
{
        auto IDX = [&](const int x, const int y, const int z)
        {
		return x + dims.x*y + dims.x*dims.y*z;
        };
    	for(size_t k = start.z; k < end.z;  ++k)
    	{
    		for(size_t j = start.y; j < end.y; ++j)
    		{
    			for(size_t i = start.x; i < end.x; ++i)
    	    		{
				dst[TRANSPOSED_IDX(i,j,k)] = src[IDX(i,j,k)];
    	    		}
    	    	}
    	}
}

static void 
transpose_xyz_to_zyx(const AcReal* src, AcReal* dst, const Volume dims, const Volume start, const Volume end)
{
	transpose_base(src,dst,dims,start,end,
    			[&](const int z, const int y, const int x)
    			{
    			    return x + dims.z*y + dims.y*dims.z*z;
    			}
		);
}

static void 
transpose_xyz_to_xzy(const AcReal* src, AcReal* dst, const Volume dims, const Volume start, const Volume end)
{
	transpose_base(src,dst,dims,start,end,
    			[&](const int x, const int z, const int y)
    			{
    			    return x + dims.x*y + dims.x*dims.z*z;
    			}
		);
}



static void 
transpose_xyz_to_yxz(const AcReal* src, AcReal* dst, const Volume dims, const Volume start, const Volume end)
{
	transpose_base(src,dst,dims,start,end,
    			[&](const size_t y, const size_t x, const size_t z)
    			{
    			    return x + dims.y*y + dims.y*dims.x*z;
    			}
		);
}

static void 
transpose_xyz_to_yzx(const AcReal* src, AcReal* dst, const Volume dims, const Volume start, const Volume end)
{
	transpose_base(src,dst,dims,start,end,
    			[&](const size_t z, const size_t x, const size_t y)
    			{
    			    return x + dims.y*y + dims.y*dims.z*z;
    			}
		);
}
static void 
transpose_xyz_to_zxy(const AcReal* src, AcReal* dst, const Volume dims, const Volume start, const Volume end)
{
	transpose_base(src,dst,dims,start,end,
    			[&](const size_t y, const size_t z, const size_t x)
    			{
    			    return x + dims.z*y + dims.x*dims.z*z;
    			}
		);
}


static AcResult
acTransposeXYZ_ZYX(const AcReal* src, AcReal* dst, const Volume dims, const Volume start, const Volume end, const cudaStream_t stream)
{
	(void)stream;
	transpose_xyz_to_zyx(src,dst,dims,start,end);
	return AC_SUCCESS;
}

static AcResult
acTransposeXYZ_ZXY(const AcReal* src, AcReal* dst, const Volume dims, const Volume start, const Volume end, const cudaStream_t stream)
{
	(void)stream;
	transpose_xyz_to_zxy(src,dst,dims,start,end);
	return AC_SUCCESS;
}

static AcResult
acTransposeXYZ_YXZ(const AcReal* src, AcReal* dst, const Volume dims, const Volume start, const Volume end, const cudaStream_t stream)
{
	(void)stream;
	transpose_xyz_to_yxz(src,dst,dims,start,end);
	return AC_SUCCESS;
}

static AcResult
acTransposeXYZ_YZX(const AcReal* src, AcReal* dst, const Volume dims, const Volume start, const Volume end, const cudaStream_t stream)
{
	(void)stream;
	transpose_xyz_to_yzx(src,dst,dims,start,end);
	return AC_SUCCESS;
}

static AcResult
acTransposeXYZ_XZY(const AcReal* src, AcReal* dst, const Volume dims, const Volume start, const Volume end, const cudaStream_t stream)
{
	(void)stream;
	transpose_xyz_to_xzy(src,dst,dims,start,end);
	return AC_SUCCESS;
}

static AcResult
acTransposeXYZ_XYZ(const AcReal* src, AcReal* dst, const Volume dims, const Volume start, const Volume end, const cudaStream_t stream)
{
	(void)stream;
	const Volume sub_dims = end-start;
	const size_t bytes = sub_dims.x*sub_dims.y*sub_dims.z*sizeof(AcReal);
	src = &src[start.x + dims.x*start.y + dims.x*dims.y*start.z];
	dst = &dst[start.x + dims.x*start.y + dims.x*dims.y*start.z];
	ERRCHK_CUDA_ALWAYS(acMemcpyAsync(dst,src,bytes,cudaMemcpyDeviceToDevice,stream));
	return AC_SUCCESS;
}

AcResult
acTransposeWithBounds(const AcMeshOrder order, const AcReal* src, AcReal* dst, const Volume dims, const Volume start, const Volume end, const cudaStream_t stream)
{
	switch(order)
	{
		case(XYZ):
			return acTransposeXYZ_XYZ(src,dst,dims,start,end,stream);
		case (XZY):
			return acTransposeXYZ_XZY(src,dst,dims,start,end,stream);
		case (YXZ):
			return acTransposeXYZ_YXZ(src,dst,dims,start,end,stream);
		case (YZX):
			return acTransposeXYZ_YZX(src,dst,dims,start,end,stream);
		case(ZXY):
			return acTransposeXYZ_ZXY(src,dst,dims,start,end,stream);
		case(ZYX):
			return acTransposeXYZ_ZYX(src,dst,dims,start,end,stream);
	}
        ERRCHK_CUDA_KERNEL();
	return AC_SUCCESS;
}

AcResult
acTranspose(const AcMeshOrder order, const AcReal* src, AcReal* dst, const Volume dims, const cudaStream_t stream)
{
	return acTransposeWithBounds(order,src,dst,dims,(Volume){0,0,0},dims,stream);
}
