#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

template <typename T, size_t N> struct StaticArray {
    T elements[N];

    constexpr size_t len(void) const { return N; }
};

template <size_t ndims> struct AcHaloSegmentInfo {
    StaticArray<size_t, ndims> dims;
    StaticArray<size_t, ndims> subdims;
    StaticArray<size_t, ndims> offset;
    size_t nbuffers;
};

template <typename T, size_t nbuffers, size_t ndims, size_t nsegments>
void
pack(const StaticArray<T, nbuffers> inputs,
     const StaticArray<AcHaloSegmentInfo<ndims>, nsegments> segments,
     StaticArray<T, nsegments> outputs)
{
    printf("Inputs len %zu\n", inputs.len());
    printf("segments len %zu\n", segments.len());
    printf("outputs len %zu\n", outputs.len());
}

int
main(void)
{
    constexpr size_t nbuffers  = 4;
    constexpr size_t ndims     = 3;
    constexpr size_t nsegments = 2;
    StaticArray<double*, nbuffers> inputs;
    StaticArray<AcHaloSegmentInfo<ndims>, nsegments> segments;
    StaticArray<double*, nsegments> outputs;

    pack<double*, nbuffers, ndims, nsegments>(inputs, segments, outputs);

    return EXIT_SUCCESS;
}
