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
kernel_pack_data(const VertexBufferArray vba, const int3 vba_start, const int3 dims,
                 AcRealPacked* packed)
{
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

    const int unpacked_idx = DEVICE_VTXBUF_IDX(i_unpacked, j_unpacked, k_unpacked);
    const int packed_idx   = i_packed +        //
                           j_packed * dims.x + //
                           k_packed * dims.x * dims.y;

    const size_t vtxbuf_offset = dims.x * dims.y * dims.z;

    //#pragma unroll
    int i = 0;
    for (int j = 0; j < NUM_VTXBUF_HANDLES; ++j)
    {
      const int dst_idx = packed_idx + i * vtxbuf_offset;
      packed[dst_idx] = vba.in[j][unpacked_idx];
      i += is_communicated(static_cast<Field>(j));
    }
}

static __global__ void
kernel_unpack_data(const AcRealPacked* packed, const int3 vba_start, const int3 dims,
                   VertexBufferArray vba)
{
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

    const int unpacked_idx = DEVICE_VTXBUF_IDX(i_unpacked, j_unpacked, k_unpacked);
    const int packed_idx   = i_packed +        //
                           j_packed * dims.x + //
                           k_packed * dims.x * dims.y;

    const size_t vtxbuf_offset = dims.x * dims.y * dims.z;

    int i = 0;
#if AC_LAGRANGIAN_GRID
    const int on_x_boundary = (DCONST(AC_domain_coordinates).x == DCONST(AC_domain_decomposition).x - 1 || 
	      		        DCONST(AC_domain_coordinates).x == 0);
    const int on_y_boundary = (DCONST(AC_domain_coordinates).y == DCONST(AC_domain_decomposition).y - 1 || 
	      		        DCONST(AC_domain_coordinates).y == 0);
    const int on_z_boundary = (DCONST(AC_domain_coordinates).z == DCONST(AC_domain_decomposition).z - 1 || 
	      		        DCONST(AC_domain_coordinates).z == 0);
#endif
    for (int j = 0; j < NUM_VTXBUF_HANDLES; ++j)
    {
      vba.in[j][unpacked_idx] = packed[packed_idx + i * vtxbuf_offset];
#if AC_LAGRANGIAN_GRID
      const AcReal x_coeff = on_x_boundary*(j == COORDS_X)*DCONST(AC_xlen);
      const AcReal y_coeff = on_y_boundary*(j == COORDS_Y)*DCONST(AC_ylen);
      const AcReal z_coeff = on_z_boundary*(j == COORDS_Z)*DCONST(AC_zlen);

      const AcReal add = x_coeff*((i_unpacked > DCONST(AC_nx_max)) - (i_unpacked < DCONST(AC_nx_min)))
      		       + y_coeff*((j_unpacked > DCONST(AC_ny_max)) - (j_unpacked < DCONST(AC_ny_min)))
      		       + z_coeff*((k_unpacked > DCONST(AC_nz_max)) - (k_unpacked < DCONST(AC_nz_min)));

      vba.in[j][unpacked_idx] += add;
#endif
      i += is_communicated(static_cast<Field>(j));
    }
}

static __global__ void
kernel_partial_pack_data(const VertexBufferArray vba, const int3 vba_start, const int3 dims,
                         AcRealPacked* packed, GpuVtxBufHandles vtxbufs, size_t num_vtxbufs)
{
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

    const int unpacked_idx = DEVICE_VTXBUF_IDX(i_unpacked, j_unpacked, k_unpacked);
    const int packed_idx   = i_packed +        //
                           j_packed * dims.x + //
                           k_packed * dims.x * dims.y;

    const size_t vtxbuf_offset = dims.x * dims.y * dims.z;

    //#pragma unroll
    for (size_t i = 0; i < num_vtxbufs; ++i)
    {
	const int dst_idx = packed_idx + i * vtxbuf_offset;
        packed[dst_idx] = vba.in[vtxbufs.data[i]][unpacked_idx];
    }
}

static __global__ void
kernel_partial_move_data(const VertexBufferArray vba, const int3 src_start, const int3 dst_start, const int3 dims,
                         GpuVtxBufHandles vtxbufs, size_t num_vtxbufs)
{
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


    const int unpacked_idx = DEVICE_VTXBUF_IDX(i_unpacked, j_unpacked, k_unpacked);
    const int dst_idx = DEVICE_VTXBUF_IDX(i_dst, j_dst, k_dst);

    //#pragma unroll
    for (size_t i = 0; i < num_vtxbufs; ++i)
    {
        vba.in[vtxbufs.data[i]][dst_idx] = vba.in[vtxbufs.data[i]][unpacked_idx];
    }
#if AC_LAGRANGIAN_GRID
    vba.in[COORDS_X][dst_idx] += DCONST(AC_xlen)*((i_dst > DCONST(AC_nx_max)) -(i_dst < DCONST(AC_nx_min)));
    vba.in[COORDS_Y][dst_idx] += DCONST(AC_ylen)*((j_dst > DCONST(AC_ny_max)) -(j_dst < DCONST(AC_ny_min)));
    vba.in[COORDS_Z][dst_idx] += DCONST(AC_zlen)*((k_dst > DCONST(AC_nz_max)) -(k_dst < DCONST(AC_nz_min)));
#endif
}

static __global__ void
kernel_partial_unpack_data(const AcRealPacked* packed, const int3 vba_start, const int3 dims,
                           VertexBufferArray vba, GpuVtxBufHandles vtxbufs , size_t num_vtxbufs)
{
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

    const int unpacked_idx = DEVICE_VTXBUF_IDX(i_unpacked, j_unpacked, k_unpacked);
    const int packed_idx   = i_packed +        //
                           j_packed * dims.x + //
                           k_packed * dims.x * dims.y;

    const size_t vtxbuf_offset = dims.x * dims.y * dims.z;

    //#pragma unroll
    // Note explicit cast size_t to int
    /**
    for (int i = 0; i < (int)num_vtxbufs; ++i) {
        int vtxbuf_id                   = vtxbufs[i];
        vba.in[vtxbuf_id][unpacked_idx] = packed[packed_idx + i * vtxbuf_offset];
    }
    for (int i = 0; i < NUM_COMMUNICATED_FIELDS; ++i)
        vba.in[i][unpacked_idx] = packed[packed_idx + i * vtxbuf_offset];
    **/
#if AC_LAGRANGIAN_GRID
    const int on_x_boundary = (DCONST(AC_domain_coordinates).x == DCONST(AC_domain_decomposition).x - 1 || 
	      		        DCONST(AC_domain_coordinates).x == 0);
    const int on_y_boundary = (DCONST(AC_domain_coordinates).y == DCONST(AC_domain_decomposition).y - 1 || 
	      		        DCONST(AC_domain_coordinates).y == 0);
    const int on_z_boundary = (DCONST(AC_domain_coordinates).z == DCONST(AC_domain_decomposition).z - 1 || 
	      		        DCONST(AC_domain_coordinates).z == 0);
#endif

     for (size_t i = 0; i < num_vtxbufs; ++i)
     {
	     const int j = vtxbufs.data[i];
	     vba.in[j][unpacked_idx] = packed[packed_idx + i * vtxbuf_offset];
#if AC_LAGRANGIAN_GRID
             const AcReal x_coeff = on_x_boundary*(j == COORDS_X)*DCONST(AC_xlen);
             const AcReal y_coeff = on_y_boundary*(j == COORDS_Y)*DCONST(AC_ylen);
             const AcReal z_coeff = on_z_boundary*(j == COORDS_Z)*DCONST(AC_zlen);

             const AcReal add = x_coeff*((i_unpacked > DCONST(AC_nx_max)) - (i_unpacked < DCONST(AC_nx_min)))
             		       + y_coeff*((j_unpacked > DCONST(AC_ny_max)) - (j_unpacked < DCONST(AC_ny_min)))
             		       + z_coeff*((k_unpacked > DCONST(AC_nz_max)) - (k_unpacked < DCONST(AC_nz_min)));
             vba.in[j][unpacked_idx] += add;
#endif
     }

}

AcResult
acKernelPackDataFull(const cudaStream_t stream, const VertexBufferArray vba, const int3 vba_start,
                 const int3 dims, AcRealPacked* packed)
{
    const dim3 tpb(32, 8, 1);
    const dim3 bpg((unsigned int)ceil(dims.x / (double)tpb.x),
                   (unsigned int)ceil(dims.y / (double)tpb.y),
                   (unsigned int)ceil(dims.z / (double)tpb.z));

    kernel_pack_data<<<bpg, tpb, 0, stream>>>(vba, vba_start, dims, packed);
    ERRCHK_CUDA_KERNEL();

    return AC_SUCCESS;
}

AcResult
acKernelUnpackDataFull(const cudaStream_t stream, const AcRealPacked* packed, const int3 vba_start,
                   const int3 dims, VertexBufferArray vba)
{
    const dim3 tpb(32, 8, 1);
    const dim3 bpg((unsigned int)ceil(dims.x / (double)tpb.x),
                   (unsigned int)ceil(dims.y / (double)tpb.y),
                   (unsigned int)ceil(dims.z / (double)tpb.z));

    kernel_unpack_data<<<bpg, tpb, 0, stream>>>(packed, vba_start, dims, vba);
    ERRCHK_CUDA_KERNEL();
    return AC_SUCCESS;
}

AcResult
acKernelPackData(const cudaStream_t stream, const VertexBufferArray vba,
                        const int3 vba_start, const int3 dims, AcRealPacked* packed,
                        const VertexBufferHandle* vtxbufs, const size_t num_vtxbufs)
{
    //done to ensure performance backwards compatibility
    if(num_vtxbufs == NUM_COMMUNICATED_FIELDS)
	    return acKernelPackDataFull(stream,vba,vba_start,dims,packed);
    const dim3 tpb(32, 8, 1);
    const dim3 bpg((unsigned int)ceil(dims.x / (double)tpb.x),
                   (unsigned int)ceil(dims.y / (double)tpb.y),
                   (unsigned int)ceil(dims.z / (double)tpb.z));

    GpuVtxBufHandles gpu_handles;
    for(size_t i=0; i<num_vtxbufs; ++i)
	    gpu_handles.data[i] = vtxbufs[i];
    kernel_partial_pack_data<<<bpg, tpb, 0, stream>>>(vba, vba_start, dims, packed, gpu_handles,
                                                      num_vtxbufs);
    ERRCHK_CUDA_KERNEL();

    return AC_SUCCESS;
}

AcResult
acKernelPackData(const cudaStream_t stream, const VertexBufferArray vba,
                        const int3 vba_start, const int3 dims, AcRealPacked* packed)
{
	return acKernelPackDataFull(stream,vba,vba_start,dims,packed);
}

AcResult
acKernelUnpackData(const cudaStream_t stream, const AcRealPacked* packed,
                          const int3 vba_start, const int3 dims, VertexBufferArray vba,
                          const VertexBufferHandle* vtxbufs, const size_t num_vtxbufs)
{
    //done to ensure performance backwards compatibility
    if(num_vtxbufs == NUM_COMMUNICATED_FIELDS)
	    return acKernelUnpackDataFull(stream,packed,vba_start,dims,vba);
    const dim3 tpb(32, 8, 1);
    const dim3 bpg((unsigned int)ceil(dims.x / (double)tpb.x),
                   (unsigned int)ceil(dims.y / (double)tpb.y),
                   (unsigned int)ceil(dims.z / (double)tpb.z));
    GpuVtxBufHandles gpu_handles;
    for(size_t i=0; i<num_vtxbufs; ++i)
	    gpu_handles.data[i] = vtxbufs[i];
    kernel_partial_unpack_data<<<bpg, tpb, 0, stream>>>(packed, vba_start, dims, vba, gpu_handles,
                                                        num_vtxbufs);
    ERRCHK_CUDA_KERNEL();
    return AC_SUCCESS;
}

AcResult
acKernelMoveData(const cudaStream_t stream, const int3 src_start, const int3 dst_start, const int3 src_dims, const int3 dst_dims, VertexBufferArray vba,
                          const VertexBufferHandle* vtxbufs, const size_t num_vtxbufs)
{
    if(src_dims != dst_dims)
    {
	    fprintf(stderr,"src and dst dims have to be the same\n");
	    fflush(stderr);
	    exit(EXIT_FAILURE);
    }
    const dim3 tpb(32, 8, 1);
    const dim3 bpg((unsigned int)ceil(src_dims.x / (double)tpb.x),
                   (unsigned int)ceil(src_dims.y / (double)tpb.y),
                   (unsigned int)ceil(src_dims.z / (double)tpb.z));
    GpuVtxBufHandles gpu_handles;
    for(size_t i=0; i<num_vtxbufs; ++i)
	    gpu_handles.data[i] = vtxbufs[i];
    kernel_partial_move_data<<<bpg, tpb, 0, stream>>>(vba,src_start, dst_start, src_dims, gpu_handles,
                                                        num_vtxbufs);
    ERRCHK_CUDA_KERNEL();
    return AC_SUCCESS;
}

AcResult
acKernelUnpackData(const cudaStream_t stream, const AcRealPacked* packed,
                          const int3 vba_start, const int3 dims, VertexBufferArray vba)
{
	return acKernelUnpackDataFull(stream,packed,vba_start,dims,vba);
}
