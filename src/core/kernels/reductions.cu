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
#include "acc_runtime.h"
#include "device_details.h"
#include "kernels.h"
#include "astaroth_cuda_wrappers.h"
#include "math_utils.h"

#include <assert.h>
#include <unordered_map>

typedef struct
{
	void* data;
	size_t bytes;
} AcDeviceTmpBuffer;

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
                   const Volume start, const Volume end, const int scratchpad_index,
                   VertexBufferArray vba)
{
    // NOTE synchronization here: we have only one scratchpad at the moment and multiple reductions
    // cannot be parallelized due to race conditions to this scratchpad Communication/memcopies
    // could be done in parallel, but allowing that also exposes the users to potential bugs with
    // race conditions
    ERRCHK_CUDA_ALWAYS(acDeviceSynchronize());


    // Compute block dimensions
    const Volume dims            = end - start;
    const size_t initial_count = dims.x * dims.y * dims.z;

    const AcKernel map_kernel = reduction.map_vtxbuf_single;
    if(map_kernel == AC_NULL_KERNEL)
    {
      ERROR("Invalid reduction type in acKernelReduceScal");
      return AC_FAILURE;
    }
    ac_resize_scratchpad_real(scratchpad_index, initial_count*sizeof(AcReal), reduction.reduce_op);
    AcReal* in  = *(vba.reduce_buffer_real[scratchpad_index].src);
    acLoadKernelParams(vba.on_device.kernel_input_params,map_kernel,vtxbuf,in); 
    acLaunchKernel(map_kernel,stream,start,end,vba);

    AcReal* out = vba.reduce_buffer_real[scratchpad_index].res;
    acReduce(stream,reduction.reduce_op, vba.reduce_buffer_real[scratchpad_index],initial_count);
    AcReal result;
    ERRCHK_CUDA(acMemcpyAsync(&result, out, sizeof(out[0]), cudaMemcpyDeviceToHost, stream));
    ERRCHK_CUDA_ALWAYS(acStreamSynchronize(stream));
    
    // NOTE synchronization here: we have only one scratchpad at the moment and multiple reductions
    // cannot be parallelized due to race conditions to this scratchpad Communication/memcopies
    // could be done in parallel, but allowing that also exposes the users to potential bugs with
    // race conditions
    ERRCHK_CUDA_ALWAYS(acDeviceSynchronize());
    return result;
}

AcReal
acKernelReduceVec(const cudaStream_t stream, const AcReduction reduction, const Volume start,
                  const Volume end, const Field3 vector, VertexBufferArray vba, const int scratchpad_index
                  )
{
    // NOTE synchronization here: we have only one scratchpad at the moment and multiple reductions
    // cannot be parallelized due to race conditions to this scratchpad Communication/memcopies
    // could be done in parallel, but allowing that also exposes the users to potential bugs with
    // race conditions
    ERRCHK_CUDA_ALWAYS(acDeviceSynchronize());


    // Set thread block dimensions
    const Volume dims            = end - start;
    const size_t initial_count = dims.x * dims.y * dims.z;

    const AcKernel map_kernel = reduction.map_vtxbuf_vec;
    if(map_kernel == AC_NULL_KERNEL)
    {
      ERROR("Invalid reduction type in acKernelReduceVec");
      return AC_FAILURE;
    }
    ac_resize_scratchpad_real(scratchpad_index, initial_count*sizeof(AcReal), reduction.reduce_op);
    AcReal* in  = *(vba.reduce_buffer_real[scratchpad_index].src);
    acLoadKernelParams(vba.on_device.kernel_input_params,map_kernel,vector,in); 
    acLaunchKernel(map_kernel,stream,start,end,vba);

    AcReal* out = vba.reduce_buffer_real[scratchpad_index].res;
    acReduce(stream,reduction.reduce_op, vba.reduce_buffer_real[scratchpad_index],initial_count);
    AcReal result;
    ERRCHK_CUDA(acMemcpyAsync(&result, out, sizeof(out[0]), cudaMemcpyDeviceToHost, stream));
    ERRCHK_CUDA_ALWAYS(acStreamSynchronize(stream));
    
    // NOTE synchronization here: we have only one scratchpad at the moment and multiple reductions
    // cannot be parallelized due to race conditions to this scratchpad Communication/memcopies
    // could be done in parallel, but allowing that also exposes the users to potential bugs with
    // race conditions
    ERRCHK_CUDA_ALWAYS(acDeviceSynchronize());
    return result;
}

AcReal
acKernelReduceVecScal(const cudaStream_t stream, const AcReduction reduction, const Volume start,
                      const Volume end, const Field4 vtxbufs,VertexBufferArray vba,
                      const int scratchpad_index)
{
    // NOTE synchronization here: we have only one scratchpad at the moment and multiple reductions
    // cannot be parallelized due to race conditions to this scratchpad Communication/memcopies
    // could be done in parallel, but allowing that also exposes the users to potential bugs with
    // race conditions
    ERRCHK_CUDA_ALWAYS(acDeviceSynchronize());


    // Set thread block dimensions
    const Volume dims            = end - start;
    const size_t initial_count = dims.x * dims.y * dims.z;
    const AcKernel map_kernel = reduction.map_vtxbuf_vec_scal;
    if(map_kernel == AC_NULL_KERNEL)
    {
      ERROR("Invalid reduction type in acKernelReduceVecScal");
      return AC_FAILURE;
    }

    ac_resize_scratchpad_real(scratchpad_index, initial_count*sizeof(AcReal), reduction.reduce_op);
    AcReal* in  = *(vba.reduce_buffer_real[scratchpad_index].src);
    acLoadKernelParams(vba.on_device.kernel_input_params,map_kernel,vtxbufs,in); 
    acLaunchKernel(map_kernel,stream,start,end,vba);

    AcReal* out = vba.reduce_buffer_real[scratchpad_index].res;
    acReduce(stream,reduction.reduce_op, vba.reduce_buffer_real[scratchpad_index],initial_count);
    AcReal result;
    ERRCHK_CUDA(acMemcpyAsync(&result, out, sizeof(out[0]), cudaMemcpyDeviceToHost, stream));
    ERRCHK_CUDA_ALWAYS(acStreamSynchronize(stream));
    
    // NOTE synchronization here: we have only one scratchpad at the moment and multiple reductions
    // cannot be parallelized due to race conditions to this scratchpad Communication/memcopies
    // could be done in parallel, but allowing that also exposes the users to potential bugs with
    // race conditions
    ERRCHK_CUDA_ALWAYS(acDeviceSynchronize());
    return result;
}

typedef struct
{
	size_t x;
	size_t y;
} size_t2;

static HOST_DEVICE_INLINE bool
operator==(const size_t2& a, const size_t2& b)
{
  return a.x == b.x && a.y == b.y;
}

struct size_t2Hash {
    std::size_t operator()(const size_t2& v) const {
        return std::hash<size_t>()(v.x) ^ std::hash<size_t>()(v.y) << 1;
    }
};

std::unordered_map<size_t2,size_t*,size_t2Hash> segmented_reduce_offsets{};

#if AC_CPU_BUILD

AcResult
acSegmentedReduce(const cudaStream_t, const AcReal*,
                  const size_t, const size_t, AcReal*, AcReal**, size_t*)
{
	printf("acSegmentedReduce not supported yet for CPU-only BUILD!\n");
	return AC_FAILURE;
}

template <typename T>
AcResult
acReduceBase(const cudaStream_t, const AcReduceOp reduce_op, T buffer, const size_t count)
{
  const auto data = *buffer.src;  
  auto tmp =  (reduce_op == REDUCE_SUM) ? data[0] - data[0] : data[0];
  for(size_t i = 0; i < count; ++i)
  {
	  if(reduce_op == REDUCE_SUM) tmp += data[i];
	  if(reduce_op == REDUCE_MAX) tmp = max(data[i],tmp);
	  if(reduce_op == REDUCE_MIN) tmp = min(data[i],tmp);
  }
  *buffer.res = tmp;
  return AC_SUCCESS;
}

#else

#if AC_USE_HIP
#include <hipcub/hipcub.hpp>
#define cub hipcub
#else
#include <cub/cub.cuh>
#endif

//TP: will return a cached allocation if one is found
size_t*
get_offsets(const size_t count, const size_t num_segments)
{
  const size_t2 key = {count,num_segments};
  if(segmented_reduce_offsets.find(key) != segmented_reduce_offsets.end())
	  return segmented_reduce_offsets[key];

  size_t* offsets = (size_t*)malloc(sizeof(offsets[0]) * (num_segments + 1));
  ERRCHK_ALWAYS(num_segments > 0);
  ERRCHK_ALWAYS(offsets);
  ERRCHK_ALWAYS(count % num_segments == 0);
  for (size_t i = 0; i <= num_segments; ++i) {
    offsets[i] = i * (count / num_segments);
    ERRCHK_ALWAYS(offsets[i] <= count);
  }
  size_t* d_offsets = NULL;
  ERRCHK_CUDA_ALWAYS(acMalloc((void**)&d_offsets, sizeof(d_offsets[0]) * (num_segments + 1)));
  ERRCHK_ALWAYS(d_offsets);
  ERRCHK_CUDA(acMemcpy((void*)d_offsets, offsets, sizeof(d_offsets[0]) * (num_segments + 1),cudaMemcpyHostToDevice));
  free(offsets);
  segmented_reduce_offsets[key] = d_offsets;
  return d_offsets;
}

template <typename T>
void
cub_reduce(AcDeviceTmpBuffer& temp_storage, const cudaStream_t stream, const T* d_in, const size_t count, T* d_out,  AcReduceOp reduce_op)
{
  switch(reduce_op)
  {
	  case(REDUCE_SUM):
	  	ERRCHK_CUDA(cub::DeviceReduce::Sum(temp_storage.data, temp_storage.bytes, d_in, d_out, count,stream));
	  	break;
	  case(REDUCE_MIN):
	  	ERRCHK_CUDA(cub::DeviceReduce::Min(temp_storage.data, temp_storage.bytes, d_in, d_out, count,stream));
	  	break;
	  case(REDUCE_MAX):
	  	ERRCHK_CUDA(cub::DeviceReduce::Max(temp_storage.data, temp_storage.bytes, d_in, d_out, count,stream));
	  	break;
	default:
		ERRCHK_ALWAYS(reduce_op != NO_REDUCE);
  }
  if (acGetLastError() != cudaSuccess) {
          ERRCHK_CUDA_KERNEL_ALWAYS();
          ERRCHK_CUDA_ALWAYS(acGetLastError());
  }
}

template <typename T>
AcResult
acReduceBase(const cudaStream_t stream, const AcReduceOp reduce_op, T buffer, const size_t count)
{
  ERRCHK(*(buffer.buffer_size)/sizeof(*(buffer.src)[0]) >= count);
  ERRCHK(buffer.src   != NULL);
  ERRCHK(buffer.src   != NULL);

  AcDeviceTmpBuffer temp_storage{NULL,0};
  cub_reduce(temp_storage,stream,*(buffer.src),count,buffer.res,reduce_op);

  *buffer.cub_tmp_size = acDeviceResize((void**)buffer.cub_tmp,*buffer.cub_tmp_size,temp_storage.bytes);
  temp_storage.data = (void*)(*buffer.cub_tmp);
  cub_reduce(temp_storage,stream,*(buffer.src),count,buffer.res,reduce_op);
  return AC_SUCCESS;
}

AcResult
acSegmentedReduce(const cudaStream_t stream, const AcReal* d_in,
                  const size_t count, const size_t num_segments, AcReal* d_out, AcReal** tmp_buffer, size_t* tmp_size)
{

  size_t* d_offsets = get_offsets(count,num_segments);

  void* d_temp_storage      = NULL;
  size_t temp_storage_bytes = 0;
  ERRCHK_CUDA(cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, d_in,
                                  d_out, num_segments, d_offsets, d_offsets + 1,
                                  stream));

  *tmp_size = acDeviceResize((void**)tmp_buffer,*tmp_size,temp_storage_bytes);
  ERRCHK_CUDA(cub::DeviceSegmentedReduce::Sum((void*)(*tmp_buffer), temp_storage_bytes, d_in,
                            d_out, num_segments, d_offsets, d_offsets + 1,
                            stream));
  ERRCHK_CUDA_KERNEL();
  return AC_SUCCESS;
}

#endif

AcResult
acReduceReal(const cudaStream_t stream, const AcReduceOp op, const AcRealScalarReduceBuffer buffer, const size_t count)
{
	return acReduceBase(stream,op,buffer,count);
}

#if AC_DOUBLE_PRECISION
AcResult
acReduceFloat(const cudaStream_t stream, const AcReduceOp op, const AcFloatScalarReduceBuffer buffer, const size_t count)
{
	return acReduceBase(stream,op,buffer,count);
}
#endif

AcResult
acReduceInt(const cudaStream_t stream, const AcReduceOp op, const AcIntScalarReduceBuffer buffer, const size_t count)
{
	return acReduceBase(stream,op,buffer,count);
}

AcResult
acReduceClean()
{
	segmented_reduce_offsets.clear();
	return AC_SUCCESS;
}

AcResult
acReduceProfileWithBounds(const Profile prof, AcReduceBuffer buffer, AcReal* dst, const cudaStream_t stream, const Volume start, const Volume end, const Volume start_after_transpose, const Volume end_after_transpose)
{
    if constexpr (NUM_PROFILES == 0) return AC_FAILURE;
    if(buffer.src.data == NULL)      return AC_NOT_ALLOCATED;
    const AcProfileType type = prof_types[prof];
    const AcMeshOrder order    = acGetMeshOrderForProfile(type);


    acTransposeWithBounds(order,buffer.src.data,buffer.transposed.data,acGetVolumeFromShape(buffer.src.shape),start,end,stream);

    const Volume dims = end_after_transpose-start_after_transpose;

    const size_t num_segments = (type & ONE_DIMENSIONAL_PROFILE) ? dims.z*buffer.transposed.shape.w
	                                                         : dims.y*dims.z*buffer.transposed.shape.w;

    const size_t count = buffer.transposed.shape.w*dims.x*dims.y*dims.z;

    const AcReal* reduce_src = buffer.transposed.data
	    		      + start_after_transpose.x + start_after_transpose.y*buffer.transposed.shape.x + start_after_transpose.z*buffer.transposed.shape.x*buffer.transposed.shape.y;

    acSegmentedReduce(stream, reduce_src, count, num_segments, dst,buffer.cub_tmp,buffer.cub_tmp_size);
    return AC_SUCCESS;
}

AcResult
acReduceProfile(const Profile prof, AcReduceBuffer buffer, AcReal* dst, const cudaStream_t stream)
{
	return acReduceProfileWithBounds(prof,buffer,dst,stream,(Volume){0,0,0},acGetVolumeFromShape(buffer.src.shape),(Volume){0,0,0},acGetVolumeFromShape(buffer.transposed.shape));
}
