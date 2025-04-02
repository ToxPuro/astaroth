#include "cubreduce.h"

#include "errchk_cuda.h"
#include "type_conversion.h"

#if defined(ACM_HIP_ENABLED)
#include <hipcub/hipcub.hpp>
#define cub hipcub
#elif defined(ACM_CUDA_ENABLED)
#include <cub/cub.cuh>
#endif

template <typename T>
void
segmented_reduce_sum(void* d_tmp_storage, size_t& tmp_storage_bytes, T* d_in, T* d_out,
                     size_t num_segments, size_t* d_offsets, size_t* d_offsets_next)
{
    const cudaStream_t stream{nullptr};
    ERRCHK_CUDA_API(cub::DeviceSegmentedReduce::Sum(d_tmp_storage,
                                                    tmp_storage_bytes,
                                                    d_in,
                                                    d_out,
                                                    as<int>(num_segments),
                                                    d_offsets,
                                                    d_offsets_next,
                                                    stream));
    ERRCHK_CUDA_API(cudaStreamSynchronize(stream));
}

template void segmented_reduce_sum<double>(void* d_tmp_storage, size_t& tmp_storage_bytes,
                                           double* d_in, double* d_out, size_t num_segments,
                                           size_t* d_offsets, size_t* d_offsets_next);
template void segmented_reduce_sum<uint64_t>(void* d_tmp_storage, size_t& tmp_storage_bytes,
                                             uint64_t* d_in, uint64_t* d_out, size_t num_segments,
                                             size_t* d_offsets, size_t* d_offsets_next);
