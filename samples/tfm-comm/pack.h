#pragma once
#include <stddef.h>

typedef struct {
    size_t ndims;
    size_t* dims;    // Dims of the parent body
    size_t* subdims; // Dims of the segment
    size_t* offsets; // Offset of the segment in the parent body
} AcHaloSegmentInfo;

void pack(const size_t nbuffers, const double* data[nbuffers], const size_t nsegments,
          const AcHaloSegmentInfo segments[nsegments], double* packed_buffers[nsegments]);

void unpack(const size_t nsegments, const AcHaloSegmentInfo segments[nsegments],
            const double* packed_buffers[nsegments], const size_t nbuffers, double* data[nbuffers]);
