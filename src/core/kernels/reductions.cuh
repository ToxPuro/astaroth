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
#pragma once
#include <assert.h>




template <typename T>
static void
swap_ptrs(T** a, T** b)
{
    T* tmp = *a;
    *a          = *b;
    *b          = tmp;
}

static Volume
get_map_tpb(void)
{
    return (Volume){32, 4, 1};
}

static Volume
get_map_bpg(const int3 dims, const Volume tpb)
{
    return (Volume){
        as_size_t(int(ceil(double(dims.x) / tpb.x))),
        as_size_t(int(ceil(double(dims.y) / tpb.y))),
        as_size_t(int(ceil(double(dims.z) / tpb.z))),
    };
}

size_t
acKernelReduceGetMinimumScratchpadSize(const int3 max_dims)
{
    const Volume tpb   = get_map_tpb();
    const Volume bpg   = get_map_bpg(max_dims, tpb);
    const size_t count = tpb.x * bpg.x * tpb.y * bpg.y * tpb.z * bpg.z;
    return count;
}

size_t
acKernelReduceGetMinimumScratchpadSizeBytes(const int3 max_dims)
{
    return sizeof(AcReal) * acKernelReduceGetMinimumScratchpadSize(max_dims);
}

AcReal
acKernelReduceScal(const cudaStream_t stream, const AcReduction reduction, const VertexBufferHandle vtxbuf,
                   const int3 start, const int3 end, const int scratchpad_index,
                   const size_t scratchpad_size, VertexBufferArray vba)
{
    // NOTE synchronization here: we have only one scratchpad at the moment and multiple reductions
    // cannot be parallelized due to race conditions to this scratchpad Communication/memcopies
    // could be done in parallel, but allowing that also exposes the users to potential bugs with
    // race conditions
    ERRCHK_CUDA_ALWAYS(cudaDeviceSynchronize());

    AcReal* in  = vba.on_device.reduce_scratchpads_real[scratchpad_index];

    // Compute block dimensions
    const int3 dims            = end - start;
    const size_t initial_count = dims.x * dims.y * dims.z;
    ERRCHK_ALWAYS(initial_count <= scratchpad_size);

    const AcKernel map_kernel = reduction.map_vtxbuf_single;
    if(map_kernel == AC_NULL_KERNEL)
    {
      ERROR("Invalid reduction type in acKernelReduceScal");
      return AC_FAILURE;
    }
    acLoadKernelParams(vba.on_device.kernel_input_params,map_kernel,vtxbuf,in); 
    acLaunchKernel(map_kernel,stream,start,end,vba);

    AcReal* out = vba.reduce_res_real[scratchpad_index];
    acReduce(stream,in,initial_count,out,reduction.reduce_op,vba.reduce_cub_tmp_real[scratchpad_index],vba.reduce_cub_tmp_size_real[scratchpad_index]);
    AcReal result;
    ERRCHK_CUDA(cudaMemcpyAsync(&result, out, sizeof(out[0]), cudaMemcpyDeviceToHost, stream));
    ERRCHK_CUDA_ALWAYS(cudaStreamSynchronize(stream));
    
    // NOTE synchronization here: we have only one scratchpad at the moment and multiple reductions
    // cannot be parallelized due to race conditions to this scratchpad Communication/memcopies
    // could be done in parallel, but allowing that also exposes the users to potential bugs with
    // race conditions
    ERRCHK_CUDA_ALWAYS(cudaDeviceSynchronize());
    return result;
}

AcReal
acKernelReduceVec(const cudaStream_t stream, const AcReduction reduction, const int3 start,
                  const int3 end, const Field3 vector, VertexBufferArray vba, const int scratchpad_index,
                  const size_t scratchpad_size)
{
    // NOTE synchronization here: we have only one scratchpad at the moment and multiple reductions
    // cannot be parallelized due to race conditions to this scratchpad Communication/memcopies
    // could be done in parallel, but allowing that also exposes the users to potential bugs with
    // race conditions
    ERRCHK_CUDA_ALWAYS(cudaDeviceSynchronize());

    AcReal* in  = vba.on_device.reduce_scratchpads_real[scratchpad_index];

    // Set thread block dimensions
    const int3 dims            = end - start;
    const size_t initial_count = dims.x * dims.y * dims.z;
    ERRCHK_ALWAYS(initial_count <= scratchpad_size);

    const AcKernel map_kernel = reduction.map_vtxbuf_vec;
    if(map_kernel == AC_NULL_KERNEL)
    {
      ERROR("Invalid reduction type in acKernelReduceVec");
      return AC_FAILURE;
    }
    acLoadKernelParams(vba.on_device.kernel_input_params,map_kernel,vector,in); 
    acLaunchKernel(map_kernel,stream,start,end,vba);

    AcReal* out = vba.reduce_res_real[scratchpad_index];
    acReduce(stream,in,initial_count,out,reduction.reduce_op,vba.reduce_cub_tmp_real[scratchpad_index],vba.reduce_cub_tmp_size_real[scratchpad_index]);
    AcReal result;
    ERRCHK_CUDA(cudaMemcpyAsync(&result, out, sizeof(out[0]), cudaMemcpyDeviceToHost, stream));
    ERRCHK_CUDA_ALWAYS(cudaStreamSynchronize(stream));
    
    // NOTE synchronization here: we have only one scratchpad at the moment and multiple reductions
    // cannot be parallelized due to race conditions to this scratchpad Communication/memcopies
    // could be done in parallel, but allowing that also exposes the users to potential bugs with
    // race conditions
    ERRCHK_CUDA_ALWAYS(cudaDeviceSynchronize());
    return result;
}

AcReal
acKernelReduceVecScal(const cudaStream_t stream, const AcReduction reduction, const int3 start,
                      const int3 end, const Field4 vtxbufs,VertexBufferArray vba,
                      const int scratchpad_index, const size_t scratchpad_size)
{
    // NOTE synchronization here: we have only one scratchpad at the moment and multiple reductions
    // cannot be parallelized due to race conditions to this scratchpad Communication/memcopies
    // could be done in parallel, but allowing that also exposes the users to potential bugs with
    // race conditions
    ERRCHK_CUDA_ALWAYS(cudaDeviceSynchronize());

    AcReal* in  = vba.on_device.reduce_scratchpads_real[scratchpad_index];

    // Set thread block dimensions
    const int3 dims            = end - start;
    const size_t initial_count = dims.x * dims.y * dims.z;
    ERRCHK_ALWAYS(initial_count <= scratchpad_size);
    const AcKernel map_kernel = reduction.map_vtxbuf_vec_scal;
    if(map_kernel == AC_NULL_KERNEL)
    {
      ERROR("Invalid reduction type in acKernelReduceVecScal");
      return AC_FAILURE;
    }

    acLoadKernelParams(vba.on_device.kernel_input_params,map_kernel,vtxbufs,in); 
    acLaunchKernel(map_kernel,stream,start,end,vba);

    AcReal* out = vba.reduce_res_real[scratchpad_index];
    acReduce(stream,in,initial_count,out,reduction.reduce_op,vba.reduce_cub_tmp_real[scratchpad_index],vba.reduce_cub_tmp_size_real[scratchpad_index]);
    AcReal result;
    ERRCHK_CUDA(cudaMemcpyAsync(&result, out, sizeof(out[0]), cudaMemcpyDeviceToHost, stream));
    ERRCHK_CUDA_ALWAYS(cudaStreamSynchronize(stream));
    
    // NOTE synchronization here: we have only one scratchpad at the moment and multiple reductions
    // cannot be parallelized due to race conditions to this scratchpad Communication/memcopies
    // could be done in parallel, but allowing that also exposes the users to potential bugs with
    // race conditions
    ERRCHK_CUDA_ALWAYS(cudaDeviceSynchronize());
    return result;
}
