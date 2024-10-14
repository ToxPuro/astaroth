#include <stddef.h>
#include <stdint.h>

typedef struct {
    size_t length;
    double* data;
} Buffer;

#define MAX_NDIMS ((uint64_t)4)

typedef struct {
    size_t ndims;
    uint64_t dims[MAX_NDIMS];
    uint64_t subdims[MAX_NDIMS];
    uint64_t offset[MAX_NDIMS]
} Segment;

#define BUFFER_ARRAY_MAX_NBUFFERS ((uint64_t)8)
typedef struct {
    size_t nbuffers;
    double* buffers[BUFFER_ARRAY_MAX_NBUFFERS];
} BufferArray;

#include <mpi.h>

typedef struct {
    Segment segment; // Information of the segment
    Buffer buffer;   // Buffer holding the data
    MPI_Request req; // MPI request holding synchronization information
} Packet;

typedef struct {
    size_t npackets;
    Packet* packets;
} PacketBatch;

////

// typedef struct {
//     Segment segment;
//     double* buffer;
// } DataSegment;

// typedef struct {
//     uint64_t nsegments;
//     DataSegments segments[MAX_LEN];
// } DataSegmentArray;

// // Buffer may not be needed

// pack(const StaticArray inputs, double* output);
// Inputs contain only one segment but several buffers

// typedef struct {
//     size_t nbuffers;
//     double* buffers[BUFFERARRAY_MAX_LEN];
// } BufferArray;

// BufferArray buffer_array_create(const size_t nbuffers, double* buffers[BUFFERARRAY_MAX_LEN]);

// const double* buffers[] = {buf0, buf1};
// BufferArray buf = {.nbuffers = ARRAY_SIZE(buffers), .buffers = buffers};

// pack(const Segment segment, const uint64_t ninputs, double* inputs[], const size_t count, const
// double* output); pack(const Segment segment, const BufferArray inputs, double* output);
