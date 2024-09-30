#include "pack.h"

void
pack(const size_t nbuffers, const double* data[nbuffers], const size_t nsegments,
     const AcHaloSegmentInfo segments[nsegments], double* packed_buffers[nsegments])
{
}

void
unpack(const size_t nsegments, const AcHaloSegmentInfo segments[nsegments],
       const double* packed_buffers[nsegments], const size_t nbuffers, double* data[nbuffers])
{
}
