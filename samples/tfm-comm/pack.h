#pragma once
#include <stddef.h>

void segment_copy(const size_t ndims, const size_t* input_dims, const size_t* input_offset,
                  const double* input, const size_t* output_dims, const size_t* output_offset,
                  double* output);

void test_pack(void);

// typedef struct {
//     const size_t ndims;
//     size_t* dims;
//     size_t* offset;
//     double* data;
// } SegmentInfo;

// SegmentInfo segment_info_create(const size_t ndims);

// void segment_info_destroy(SegmentInfo* info);

// void segment_copy_batched(const size_t ninput_segments, SegmentInfo* input_segments,
//                           const size_t noutput_segments, SegmentInfo* output_segments);
