/*
    Copyright (C) 2014-2021, Johannes Pekkila, Miikka Vaisala.

    This file is part of Astaroth.

    Astaroth is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Astaroth is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Astaroth.  If not, see <http://www.gnu.org/licenses/>.
*/
#include "vtxbuf_is_communicated_func.h"
#pragma once
struct GpuVtxBufHandles 
{
	//we pad by one in case there is no communicated fields
	VertexBufferHandle data[NUM_COMMUNICATED_FIELDS+1];
};
static __global__ void
kernel_pack_data(const DeviceVertexBufferArray vba, const int3 vba_start, const int3 dims,
                 AcRealPacked* packed)
{ 
    KERNEL_DIMS_PREFIX
    const int i_packed = threadIdx.x + blockIdx.x * blockDim.x;
    const int j_packed = threadIdx.y + blockIdx.y * blockDim.y;
    const int k_packed = threadIdx.z + blockIdx.z * blockDim.z;

    // If within the start-end range (this allows threadblock dims that are not
    // divisible by end - start)
    if (i_packed >= dims.x || //
        j_packed >= dims.y || //
        k_packed >= dims.z) {
        return;
    }

    const int i_unpacked = i_packed + vba_start.x;
    const int j_unpacked = j_packed + vba_start.y;
    const int k_unpacked = k_packed + vba_start.z;

    const int packed_idx   = i_packed +        //
                           j_packed * dims.x + //
                           k_packed * dims.x * dims.y;

    const size_t vtxbuf_offset = dims.x * dims.y * dims.z;

    //#pragma unroll
    int i = 0;
    for (int j = 0; j < NUM_VTXBUF_HANDLES; ++j)
    {
      const int unpacked_idx = DEVICE_VARIABLE_VTXBUF_IDX(i_unpacked, j_unpacked, k_unpacked,VAL(vtxbuf_device_dims[j]));
      const int dst_idx = packed_idx + i * vtxbuf_offset;
      i += is_communicated(static_cast<Field>(j));
      if(is_communicated(static_cast<Field>(j))) packed[dst_idx] = vba.in[j][unpacked_idx];
    }
    KERNEL_POSTFIX
}

static inline __device__ int3
is_on_boundary()
{
    const int x = (VAL(AC_domain_coordinates).x == VAL(AC_domain_decomposition).x - 1 ||
                                VAL(AC_domain_coordinates).x == 0);
    const int y = (VAL(AC_domain_coordinates).y == VAL(AC_domain_decomposition).y - 1 ||
                                VAL(AC_domain_coordinates).y == 0);
    const int z = (VAL(AC_domain_coordinates).z == VAL(AC_domain_decomposition).z - 1 ||
                                VAL(AC_domain_coordinates).z == 0);
    return (int3){x,y,z};
}

//TP: this could be rewritten to return real2 or real3 but works the way it is for now
static inline __device__ AcReal
lagrangian_correction(const int j, const Field2 coords, const int3 indexes)
{
	const int3 on_boundary = is_on_boundary();
	const AcReal x_coeff = on_boundary.x*(j == coords.x)*VAL(AC_len).x;
	const AcReal y_coeff = on_boundary.y*(j == coords.y)*VAL(AC_len).y;
        return  x_coeff*((indexes.x >= VAL(AC_nlocal_max).x) - (indexes.x < VAL(AC_nmin).x))
              + y_coeff*((indexes.y >= VAL(AC_nlocal_max).y) - (indexes.y < VAL(AC_nmin).y));
}
static inline __device__ AcReal
lagrangian_correction(const int j, const Field3 coords, const int3 indexes)
{
	const int3 on_boundary = is_on_boundary();
	const AcReal x_coeff = on_boundary.x*(j == coords.x)*VAL(AC_len).x;
	const AcReal y_coeff = on_boundary.y*(j == coords.y)*VAL(AC_len).y;
	const AcReal z_coeff = on_boundary.z*(j == coords.z)*VAL(AC_len).z;
        return  x_coeff*((indexes.x >= VAL(AC_nlocal_max).x) - (indexes.x < VAL(AC_nmin).x))
              + y_coeff*((indexes.y >= VAL(AC_nlocal_max).y) - (indexes.y < VAL(AC_nmin).y))
              + z_coeff*((indexes.z >= VAL(AC_nlocal_max).z) - (indexes.z < VAL(AC_nmin).z));
}


static __global__ void
kernel_unpack_data(const AcRealPacked* packed, const int3 vba_start, const int3 dims,
                   DeviceVertexBufferArray vba)
{
    KERNEL_DIMS_PREFIX
    const int i_packed = threadIdx.x + blockIdx.x * blockDim.x;
    const int j_packed = threadIdx.y + blockIdx.y * blockDim.y;
    const int k_packed = threadIdx.z + blockIdx.z * blockDim.z;

    // If within the start-end range (this allows threadblock dims that are not
    // divisible by end - start)
    if (i_packed >= dims.x || //
        j_packed >= dims.y || //
        k_packed >= dims.z) {
        return;
    }

    const int i_unpacked = i_packed + vba_start.x;
    const int j_unpacked = j_packed + vba_start.y;
    const int k_unpacked = k_packed + vba_start.z;

    const int packed_idx   = i_packed +        //
                           j_packed * dims.x + //
                           k_packed * dims.x * dims.y;

    const size_t vtxbuf_offset = dims.x * dims.y * dims.z;

    int i = 0;
    for (int j = 0; j < NUM_VTXBUF_HANDLES; ++j)
    {
      if(is_communicated(static_cast<Field>(j)))
      {
        const int unpacked_idx = DEVICE_VARIABLE_VTXBUF_IDX(i_unpacked, j_unpacked, k_unpacked,VAL(vtxbuf_device_dims[j]));
        vba.in[j][unpacked_idx] = packed[packed_idx + i * vtxbuf_offset];
#if AC_LAGRANGIAN_GRID
      	vba.in[j][unpacked_idx] += lagrangian_correction(j, AC_COORDS, (int3){i_unpacked,j_unpacked,k_unpacked});
#endif
      }
      i += is_communicated(static_cast<Field>(j));
    }
    KERNEL_POSTFIX
}


static __global__ void
kernel_partial_pack_data(const DeviceVertexBufferArray vba, const int3 vba_start, const int3 dims,
                         AcRealPacked* packed, GpuVtxBufHandles vtxbufs, size_t num_vtxbufs)
{
    KERNEL_DIMS_PREFIX
    const int i_packed = threadIdx.x + blockIdx.x * blockDim.x;
    const int j_packed = threadIdx.y + blockIdx.y * blockDim.y;
    const int k_packed = threadIdx.z + blockIdx.z * blockDim.z;


    // If within the start-end range (this allows threadblock dims that are not
    // divisible by end - start)
    if (i_packed >= dims.x || //
        j_packed >= dims.y || //
        k_packed >= dims.z) {
        return;
    }

    const int i_unpacked = i_packed + vba_start.x;
    const int j_unpacked = j_packed + vba_start.y;
    const int k_unpacked = k_packed + vba_start.z;

    const int packed_idx   = i_packed +        //
                           j_packed * dims.x + //
                           k_packed * dims.x * dims.y;

    const size_t vtxbuf_offset = dims.x * dims.y * dims.z;

    //#pragma unroll
    for (size_t i = 0; i < num_vtxbufs; ++i)
    {
	const int j = vtxbufs.data[i];
        const int unpacked_idx = DEVICE_VARIABLE_VTXBUF_IDX(i_unpacked, j_unpacked, k_unpacked,VAL(vtxbuf_device_dims[j]));
	const int dst_idx = packed_idx + i * vtxbuf_offset;
        packed[dst_idx] = vba.in[j][unpacked_idx];
    }
    KERNEL_POSTFIX
}

static __global__ void
kernel_partial_move_data(const DeviceVertexBufferArray vba, const int3 src_start, const int3 dst_start, const int3 dims,
                         GpuVtxBufHandles vtxbufs, size_t num_vtxbufs)
{
    KERNEL_DIMS_PREFIX
    const int i_packed = threadIdx.x + blockIdx.x * blockDim.x;
    const int j_packed = threadIdx.y + blockIdx.y * blockDim.y;
    const int k_packed = threadIdx.z + blockIdx.z * blockDim.z;

    // If within the start-end range (this allows threadblock dims that are not
    // divisible by end - start)
    if (i_packed >= dims.x || //
        j_packed >= dims.y || //
        k_packed >= dims.z) {
        return;
    }

    const int i_unpacked = i_packed + src_start.x;
    const int j_unpacked = j_packed + src_start.y;
    const int k_unpacked = k_packed + src_start.z;

    const int i_dst = i_packed + dst_start.x;
    const int j_dst = j_packed + dst_start.y;
    const int k_dst = k_packed + dst_start.z;



    //#pragma unroll
    for (size_t i = 0; i < num_vtxbufs; ++i)
    {
	const int j = vtxbufs.data[i];
        const int unpacked_idx = DEVICE_VARIABLE_VTXBUF_IDX(i_unpacked, j_unpacked, k_unpacked,VAL(vtxbuf_device_dims[j]));
        const int dst_idx = DEVICE_VARIABLE_VTXBUF_IDX(i_dst, j_dst, k_dst,VAL(vtxbuf_device_dims[j]));
        vba.in[j][dst_idx] = vba.in[j][unpacked_idx];
#if AC_LAGRANGIAN_GRID
        	vba.in[j][dst_idx] += lagrangian_correction(vtxbufs.data[i], AC_COORDS, (int3){i_dst, j_dst, k_dst});
#endif
    }
    KERNEL_POSTFIX
}

//TP: does not work with variable dimensions for now!!
static __global__ void
kernel_shear_partial_unpack_data(const AcRealPacked* packed, const int3 vba_start, const int3 dims,
                           DeviceVertexBufferArray vba, GpuVtxBufHandles vtxbufs , size_t num_vtxbufs
			   , const AcShearInterpolationCoeffs coeffs, const int offset
			   )
{
    KERNEL_DIMS_PREFIX
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;

    // If within the start-end range (this allows threadblock dims that are not
    // divisible by end - start)
    if (i >= dims.x || //
        j >= dims.y || //
        k >= dims.z) {
        return;
    }

    const int i_unpacked = i + vba_start.x;
    const int j_unpacked = j + vba_start.y;
    const int k_unpacked = k + vba_start.z;

    const bool on_left_boundary = i_unpacked < NGHOST;

    const int unpacked_idx = DEVICE_VTXBUF_IDX(i_unpacked, j_unpacked, k_unpacked);

   
    //TP: message dimensions do not line up with kernel dims since we get volumes with ny but want to populate whole my
    const int3 message_dims = 
    {
	    dims.x,
	    VAL(AC_nlocal).y,
	    dims.z
    };
    //TP: the wonky indexing is because we get 3d halos from multiple processes and these are contiguous in y,
    //    with each process halo coming after another
    const auto get_index = [&](const int x, const int y, const int z, const int vtxbuf_index)
    {
	    return 
		    x +
		    (y % message_dims.y) * message_dims.x +
		    z * message_dims.x * message_dims.y   +
		    vtxbuf_index * message_dims.x*message_dims.y*message_dims.z +
		    (y / message_dims.y) * message_dims.x*message_dims.y*message_dims.z*num_vtxbufs;
    };
    
    const int x = i;
    const int y = j + offset;
    const int z = k;
    for(size_t vtxbuf_index = 0; vtxbuf_index < num_vtxbufs; ++vtxbuf_index)
    {
    	if(on_left_boundary)
    	{
    	    AcReal res  = coeffs.c1*packed[get_index(x,y+2,z,vtxbuf_index)];
    	    res        += coeffs.c2*packed[get_index(x,y+1,z,vtxbuf_index)];
    	    res        += coeffs.c3*packed[get_index(x,y+0,z,vtxbuf_index)];
    	    res        += coeffs.c4*packed[get_index(x,y-1,z,vtxbuf_index)];
    	    res        += coeffs.c5*packed[get_index(x,y-2,z,vtxbuf_index)];
    	    res        += coeffs.c6*packed[get_index(x,y-3,z,vtxbuf_index)];
	    vba.in[vtxbufs.data[vtxbuf_index]][unpacked_idx] = res;
    	}
    	else
    	{
    	    AcReal res  = coeffs.c1*packed[get_index(x,y-2,z,vtxbuf_index)];
    	    res        += coeffs.c2*packed[get_index(x,y-1,z,vtxbuf_index)];
    	    res        += coeffs.c3*packed[get_index(x,y-0,z,vtxbuf_index)];
    	    res        += coeffs.c4*packed[get_index(x,y+1,z,vtxbuf_index)];
    	    res        += coeffs.c5*packed[get_index(x,y+2,z,vtxbuf_index)];
    	    res        += coeffs.c6*packed[get_index(x,y+3,z,vtxbuf_index)];
	    vba.in[vtxbufs.data[vtxbuf_index]][unpacked_idx] = res;
    	}
    }
    KERNEL_POSTFIX
}

static __global__ void
kernel_partial_unpack_data(const AcRealPacked* packed, const int3 vba_start, const int3 dims,
                           DeviceVertexBufferArray vba, GpuVtxBufHandles vtxbufs , size_t num_vtxbufs)
{
    KERNEL_DIMS_PREFIX
    const int i_packed = threadIdx.x + blockIdx.x * blockDim.x;
    const int j_packed = threadIdx.y + blockIdx.y * blockDim.y;
    const int k_packed = threadIdx.z + blockIdx.z * blockDim.z;

    // If within the start-end range (this allows threadblock dims that are not
    // divisible by end - start)
    if (i_packed >= dims.x || //
        j_packed >= dims.y || //
        k_packed >= dims.z) {
        return;
    }

    const int i_unpacked = i_packed + vba_start.x;
    const int j_unpacked = j_packed + vba_start.y;
    const int k_unpacked = k_packed + vba_start.z;

    const int packed_idx   = i_packed +        //
                           j_packed * dims.x + //
                           k_packed * dims.x * dims.y;

    const size_t vtxbuf_offset = dims.x * dims.y * dims.z;

    //#pragma unroll
     for (size_t i = 0; i < num_vtxbufs; ++i)
     {
	     const int j = vtxbufs.data[i];
    	     const int unpacked_idx = DEVICE_VARIABLE_VTXBUF_IDX(i_unpacked, j_unpacked, k_unpacked,VAL(vtxbuf_device_dims[j]));
	     vba.in[j][unpacked_idx] = packed[packed_idx + i * vtxbuf_offset];
#if AC_LAGRANGIAN_GRID
             	vba.in[j][unpacked_idx] += lagrangian_correction(j, AC_COORDS, (int3){i_unpacked, j_unpacked, k_unpacked});
#endif
     }
     KERNEL_POSTFIX
}

AcResult
acKernelPackDataFull(const cudaStream_t stream, const VertexBufferArray vba, const Volume vba_start,
                 const Volume dims, AcRealPacked* packed)
{
    (void)stream;
    const dim3 tpb{32, 8, 1};
    [[maybe_unused]] const dim3 bpg{(unsigned int)ceil(dims.x / (double)tpb.x),
                   (unsigned int)ceil(dims.y / (double)tpb.y),
                   (unsigned int)ceil(dims.z / (double)tpb.z)};

    KERNEL_LAUNCH(kernel_pack_data,bpg,tpb,0,stream)(vba.on_device, to_int3(vba_start), to_int3(dims), packed);
    ERRCHK_CUDA_KERNEL();

    return AC_SUCCESS;
}

AcResult
acKernelUnpackDataFull(const cudaStream_t stream, const AcRealPacked* packed, const Volume vba_start,
                   const Volume dims, VertexBufferArray vba)
{
    (void)stream;
    const dim3 tpb{32, 8, 1};
    [[maybe_unused]] const dim3 bpg{(unsigned int)ceil(dims.x / (double)tpb.x),
                   (unsigned int)ceil(dims.y / (double)tpb.y),
                   (unsigned int)ceil(dims.z / (double)tpb.z)};

    KERNEL_LAUNCH(kernel_unpack_data,bpg,tpb,0,stream)(packed, to_int3(vba_start), to_int3(dims), vba.on_device);
    ERRCHK_CUDA_KERNEL();
    return AC_SUCCESS;
}

AcResult
acKernelPackData(const cudaStream_t stream, const VertexBufferArray vba,
                        const Volume vba_start, const Volume dims, AcRealPacked* packed,
                        const VertexBufferHandle* vtxbufs, const size_t num_vtxbufs)
{
    (void)stream;
    //done to ensure performance backwards compatibility
    if(num_vtxbufs == NUM_COMMUNICATED_FIELDS)
	    return acKernelPackDataFull(stream,vba,vba_start,dims,packed);
    const dim3 tpb{32, 8, 1};
    [[maybe_unused]] const dim3 bpg{(unsigned int)ceil(dims.x / (double)tpb.x),
                   (unsigned int)ceil(dims.y / (double)tpb.y),
                   (unsigned int)ceil(dims.z / (double)tpb.z)};

    GpuVtxBufHandles gpu_handles;
    for(size_t i=0; i<num_vtxbufs; ++i)
	    gpu_handles.data[i] = vtxbufs[i];
    KERNEL_LAUNCH(kernel_partial_pack_data,bpg,tpb,0,stream)(vba.on_device, to_int3(vba_start), to_int3(dims), packed, gpu_handles,
                                                      num_vtxbufs);
    ERRCHK_CUDA_KERNEL();

    return AC_SUCCESS;
}


AcResult
acKernelPackData(const cudaStream_t stream, const VertexBufferArray vba,
                        const Volume vba_start, const Volume dims, AcRealPacked* packed)
{
	return acKernelPackDataFull(stream,vba,vba_start,dims,packed);
}

AcResult
acKernelUnpackData(const cudaStream_t stream, const AcRealPacked* packed,
                          const Volume vba_start, const Volume dims, VertexBufferArray vba,
                          const VertexBufferHandle* vtxbufs, const size_t num_vtxbufs)
{
    //done to ensure performance backwards compatibility
    if(num_vtxbufs == NUM_COMMUNICATED_FIELDS)
	    return acKernelUnpackDataFull(stream,packed,vba_start,dims,vba);
    const dim3 tpb{32, 8, 1};
    [[maybe_unused]] const dim3 bpg{(unsigned int)ceil(dims.x / (double)tpb.x),
                   (unsigned int)ceil(dims.y / (double)tpb.y),
                   (unsigned int)ceil(dims.z / (double)tpb.z)};
    GpuVtxBufHandles gpu_handles;
    for(size_t i=0; i<num_vtxbufs; ++i)
	    gpu_handles.data[i] = vtxbufs[i];
    KERNEL_LAUNCH(kernel_partial_unpack_data,bpg,tpb,0,stream)(packed, to_int3(vba_start), to_int3(dims), vba.on_device, gpu_handles,
                                                        num_vtxbufs);
    ERRCHK_CUDA_KERNEL();
    return AC_SUCCESS;
}

AcResult
acKernelShearUnpackData(const cudaStream_t stream, const AcRealPacked* packed,
                          const Volume vba_start, const Volume dims, VertexBufferArray vba,
                          const VertexBufferHandle* vtxbufs, const size_t num_vtxbufs,
			  const AcShearInterpolationCoeffs coeffs, const int offset
			  )
{ 
    (void)stream;
    //done to ensure performance backwards compatibility
    const dim3 tpb{32, 8, 1};
    [[maybe_unused]] const dim3 bpg{(unsigned int)ceil(dims.x / (double)tpb.x),
                   (unsigned int)ceil(dims.y / (double)tpb.y),
                   (unsigned int)ceil(dims.z / (double)tpb.z)};
    GpuVtxBufHandles gpu_handles;
    for(size_t i=0; i<num_vtxbufs; ++i)
	    gpu_handles.data[i] = vtxbufs[i];
    KERNEL_LAUNCH(kernel_shear_partial_unpack_data,bpg,tpb,0,stream)(packed, to_int3(vba_start), to_int3(dims), vba.on_device, gpu_handles,
                                                        num_vtxbufs, coeffs, offset);
    ERRCHK_CUDA_KERNEL();
    return AC_SUCCESS;
}

AcResult
acKernelMoveData(const cudaStream_t stream, const Volume src_start, const Volume dst_start, const Volume src_dims, const Volume dst_dims, VertexBufferArray vba,
                          const VertexBufferHandle* vtxbufs, const size_t num_vtxbufs)
{
    (void)stream;
    if(src_dims != dst_dims)
    {
	    fprintf(stderr,"src and dst dims have to be the same\n");
	    fflush(stderr);
	    exit(EXIT_FAILURE);
    }
    const dim3 tpb{32, 8, 1};
    [[maybe_unused]] const dim3 bpg{(unsigned int)ceil(src_dims.x / (double)tpb.x),
                   (unsigned int)ceil(src_dims.y / (double)tpb.y),
                   (unsigned int)ceil(src_dims.z / (double)tpb.z)};
    GpuVtxBufHandles gpu_handles;
    for(size_t i=0; i<num_vtxbufs; ++i)
	    gpu_handles.data[i] = vtxbufs[i];
    KERNEL_LAUNCH(kernel_partial_move_data,bpg,tpb,0,stream)(vba.on_device,to_int3(src_start), to_int3(dst_start), to_int3(src_dims), gpu_handles,
                                                        num_vtxbufs);
    ERRCHK_CUDA_KERNEL();
    return AC_SUCCESS;
}

AcResult
acKernelUnpackData(const cudaStream_t stream, const AcRealPacked* packed,
                          const Volume vba_start, const Volume dims, VertexBufferArray vba)
{
	return acKernelUnpackDataFull(stream,packed,vba_start,dims,vba);
}
