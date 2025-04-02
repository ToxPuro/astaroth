#pragma once
#include <cstddef>
#include <cstdint>

template <typename T>
void segmented_reduce_sum(void* d_tmp_storage, size_t& tmp_storage_bytes, T* d_in, T* d_out,
                          size_t num_segments, size_t* d_offsets, size_t* d_offsets_next);

extern template void segmented_reduce_sum<double>(void* d_tmp_storage, size_t& tmp_storage_bytes,
                                                  double* d_in, double* d_out, size_t num_segments,
                                                  size_t* d_offsets, size_t* d_offsets_next);

extern template void segmented_reduce_sum<uint64_t>(void* d_tmp_storage, size_t& tmp_storage_bytes,
                                                    uint64_t* d_in, uint64_t* d_out,
                                                    size_t num_segments, size_t* d_offsets,
                                                    size_t* d_offsets_next);
