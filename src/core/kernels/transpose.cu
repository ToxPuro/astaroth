#include "host_datatypes.h"
#include "device_headers.h"
#include "device_details.h"
#include "ac_helpers.h"
#include "transpose.h"

#include "astaroth_cuda_wrappers.h"
#include "errchk.h"
#include "math_utils_base.h"


#define TILE_DIM (32)

void  __global__
transpose_xyz_to_zyx(const AcReal* src, AcReal* dst, const Volume dims, const Volume start, const Volume end)
{
	__shared__ AcReal tile[TILE_DIM][TILE_DIM];
	const dim3 block_offset =
	{
		blockIdx.x*TILE_DIM,
		blockIdx.y,
		blockIdx.z*TILE_DIM
	};

	const dim3 vertexIdx = 
	{
		(int)start.x + threadIdx.x + block_offset.x,
		(int)start.y + threadIdx.y + block_offset.y,
		(int)start.z + threadIdx.z + block_offset.z
	};
	const dim3 out_vertexIdx = 
	{
		(int)start.z + threadIdx.x + block_offset.z,
		(int)start.y + threadIdx.y + block_offset.y,
		(int)start.x + threadIdx.z + block_offset.x
	};
	const bool in_oob  =  vertexIdx.x  >= end.x    ||  vertexIdx.y >= end.y     || vertexIdx.z >= end.z;
	const bool out_oob =  out_vertexIdx.x >= end.z ||  out_vertexIdx.y >= end.y || out_vertexIdx.z >= end.x;



	tile[threadIdx.z][threadIdx.x] = !in_oob ? src[vertexIdx.x + dims.x*(vertexIdx.y + dims.y*vertexIdx.z)] : (AcReal)0.0;
	__syncthreads();
	if(!out_oob)
		dst[out_vertexIdx.x +dims.z*out_vertexIdx.y + dims.z*dims.y*out_vertexIdx.z] = tile[threadIdx.x][threadIdx.z];
}


void __global__ 
transpose_xyz_to_zxy(const AcReal* src, AcReal* dst, const Volume dims, const Volume start, const Volume end)
{
	__shared__ AcReal tile[TILE_DIM][TILE_DIM];
	const dim3 block_offset =
	{
		blockIdx.x*TILE_DIM,
		blockIdx.y,
		blockIdx.z*TILE_DIM
	};

	const dim3 vertexIdx = 
	{
		(int) start.x + threadIdx.x + block_offset.x,
		(int) start.y + threadIdx.y + block_offset.y,
		(int) start.z + threadIdx.z + block_offset.z
	};
	const dim3 out_vertexIdx = 
	{
		(int)start.z + threadIdx.x + block_offset.z,
		(int)start.y + threadIdx.y + block_offset.y,
		(int)start.x + threadIdx.z + block_offset.x
	};
	const bool in_oob  =  vertexIdx.x  >= end.x    ||  vertexIdx.y >= end.y     || vertexIdx.z >= end.z;
	const bool out_oob =  out_vertexIdx.x >= end.z ||  out_vertexIdx.y >= end.y || out_vertexIdx.z >= end.x;



	tile[threadIdx.z][threadIdx.x] = !in_oob ? src[vertexIdx.x + dims.x*(vertexIdx.y + dims.y*vertexIdx.z)] : (AcReal)0.0;
	__syncthreads();
	if(!out_oob)
		dst[out_vertexIdx.x +dims.z*out_vertexIdx.z + dims.z*dims.x*out_vertexIdx.y] = tile[threadIdx.x][threadIdx.z];
}

void __global__ 
transpose_xyz_to_yxz(const AcReal* src, AcReal* dst, const Volume dims, const Volume start, const Volume end)
{
	__shared__ AcReal tile[TILE_DIM][TILE_DIM];
	const dim3 block_offset =
	{
		blockIdx.x*TILE_DIM,
		blockIdx.y*TILE_DIM,
		blockIdx.z
	};

	const dim3 vertexIdx = 
	{
		(int) start.x + threadIdx.x + block_offset.x,
		(int) start.y + threadIdx.y + block_offset.y,
		(int) start.z + threadIdx.z + block_offset.z
	};
	const dim3 out_vertexIdx = 
	{
		(int)start.y + threadIdx.x + block_offset.y,
		(int)start.x + threadIdx.y + block_offset.x,
		(int)start.z + threadIdx.z + block_offset.z
	};
	const bool in_oob  =  vertexIdx.x  >= end.x    ||  vertexIdx.y >= end.y     || vertexIdx.z >= end.z;
	const bool out_oob =  out_vertexIdx.x >= end.y ||  out_vertexIdx.y >= end.x || out_vertexIdx.z >= end.z;



	tile[threadIdx.y][threadIdx.x] = !in_oob ? src[vertexIdx.x + dims.x*(vertexIdx.y + dims.y*vertexIdx.z)] : (AcReal)0.0;
	__syncthreads();
	if(!out_oob)
		dst[out_vertexIdx.x +dims.y*out_vertexIdx.y + dims.x*dims.y*out_vertexIdx.z] = tile[threadIdx.x][threadIdx.y];
}

void __global__ 
transpose_xyz_to_yzx(const AcReal* src, AcReal* dst, const Volume dims, const Volume start, const Volume end)
{
	__shared__ AcReal tile[TILE_DIM][TILE_DIM];
	const dim3 block_offset =
	{
		blockIdx.x*TILE_DIM,
		blockIdx.y*TILE_DIM,
		blockIdx.z
	};

	const dim3 vertexIdx = 
	{
		(int)start.x +threadIdx.x + block_offset.x,
		(int)start.y +threadIdx.y + block_offset.y,
		(int)start.z +threadIdx.z + block_offset.z
	};
	const dim3 out_vertexIdx = 
	{
		(int)start.y + threadIdx.x + block_offset.y,
		(int)start.x + threadIdx.y + block_offset.x,
		(int)start.z + threadIdx.z + block_offset.z
	};
	const bool in_oob  =  vertexIdx.x  >= end.x    ||  vertexIdx.y >= end.y     || vertexIdx.z >= end.z;
	const bool out_oob =  out_vertexIdx.x >= end.y ||  out_vertexIdx.y >= end.x || out_vertexIdx.z >= end.z;



	tile[threadIdx.y][threadIdx.x] = !in_oob ? src[vertexIdx.x + dims.x*(vertexIdx.y + dims.y*vertexIdx.z)] : (AcReal)0.0;
	__syncthreads();
	if(!out_oob)
		dst[out_vertexIdx.x +dims.y*out_vertexIdx.z + dims.y*dims.z*out_vertexIdx.y] = tile[threadIdx.x][threadIdx.y];
}

void __global__ 
transpose_xyz_to_xzy(const AcReal* src, AcReal* dst, const Volume dims, const Volume start, const Volume end)
{
	const dim3 in_block_offset =
	{
		blockIdx.x*blockDim.x,
		blockIdx.y*blockDim.y,
		blockIdx.z*blockDim.z
	};

	const dim3 vertexIdx = 
	{
		(int)start.x + threadIdx.x + in_block_offset.x,
		(int)start.y + threadIdx.y + in_block_offset.y,
		(int)start.z + threadIdx.z + in_block_offset.z
	};

	const bool oob  =  vertexIdx.x  >= end.x    ||  vertexIdx.y >= end.y     || vertexIdx.z >= end.z;
	if(oob) return;
	dst[vertexIdx.x + dims.x*vertexIdx.z + dims.x*dims.z*vertexIdx.y] 
		= src[vertexIdx.x + dims.x*(vertexIdx.y + dims.y*vertexIdx.z)];
}

static AcResult
acTransposeXYZ_ZYX(const AcReal* src, AcReal* dst, const Volume dims, const Volume start, const Volume end, const cudaStream_t stream)
{
	(void)stream;
	const dim3 tpb = {32,1,32};
	const Volume sub_dims = end-start;
	[[maybe_unused]] const dim3 bpg = to_dim3(get_bpg(sub_dims,to_volume(tpb)));
  	KERNEL_LAUNCH(transpose_xyz_to_zyx,bpg, tpb, 0, stream)(src,dst,dims,start,end);
	ERRCHK_CUDA_KERNEL();
	return AC_SUCCESS;
}

static AcResult
acTransposeXYZ_ZXY(const AcReal* src, AcReal* dst, const Volume dims, const Volume start, const Volume end, const cudaStream_t stream)
{
	(void)stream;
	const dim3 tpb = {32,1,32};
	const Volume sub_dims = end-start;
	[[maybe_unused]] const dim3 bpg = to_dim3(get_bpg(sub_dims,to_volume(tpb)));
  	KERNEL_LAUNCH(transpose_xyz_to_zxy,bpg, tpb, 0, stream)(src,dst,dims,start,end);
	ERRCHK_CUDA_KERNEL();
	return AC_SUCCESS;
}

static AcResult
acTransposeXYZ_YXZ(const AcReal* src, AcReal* dst, const Volume dims, const Volume start, const Volume end, const cudaStream_t stream)
{
	(void)stream;
	const dim3 tpb = {32,32,1};
	const Volume sub_dims = end-start;
	[[maybe_unused]] const dim3 bpg = to_dim3(get_bpg(sub_dims,to_volume(tpb)));
  	KERNEL_LAUNCH(transpose_xyz_to_yxz,bpg, tpb, 0, stream)(src,dst,dims,start,end);
	ERRCHK_CUDA_KERNEL();
	return AC_SUCCESS;
}

static AcResult
acTransposeXYZ_YZX(const AcReal* src, AcReal* dst, const Volume dims, const Volume start, const Volume end, const cudaStream_t stream)
{
	(void)stream;
	const dim3 tpb = {32,32,1};
	const Volume sub_dims = end-start;
	[[maybe_unused]] const dim3 bpg = to_dim3(get_bpg(sub_dims,to_volume(tpb)));
  	KERNEL_LAUNCH(transpose_xyz_to_yzx,bpg, tpb, 0, stream)(src,dst,dims,start,end);
	ERRCHK_CUDA_KERNEL();
	return AC_SUCCESS;
}

static AcResult
acTransposeXYZ_XZY(const AcReal* src, AcReal* dst, const Volume dims, const Volume start, const Volume end, const cudaStream_t stream)
{
	(void)stream;
	const dim3 tpb = {32,32,1};
	const Volume sub_dims = end-start;
	[[maybe_unused]] const dim3 bpg = to_dim3(get_bpg(sub_dims,to_volume(tpb)));
  	KERNEL_LAUNCH(transpose_xyz_to_xzy,bpg, tpb, 0, stream)(src,dst,dims,start,end);
	ERRCHK_CUDA_KERNEL();
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
