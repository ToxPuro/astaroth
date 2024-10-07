#include "pack.h"

#include "errchk.h"
#include "misc.h"
#include <cassert>
#include <iostream>

#define MAX_NDIMS (3)
#define MAX_NBUFFERS (7)

typedef struct {
    size_t ndims;
    size_t mm[MAX_NDIMS];
    size_t block_shape[MAX_NDIMS];
    size_t block_offset[MAX_NDIMS];
    size_t nbuffers;
    double* buffers[MAX_NBUFFERS];
    double* pack_buffer;
} PackKernelParams;

template <typename T>
static void
ncopy(const size_t count, const T* in, T* out)
{
    memmove(out, in, sizeof(in[0]) * count);
}

template <typename T>
void
print(const char* label, const T value)
{
    std::cout << label << ": " << value << "\n";
}

template <typename T>
void
print_array(const char* label, const size_t count, const T* arr)
{
    std::cout << label << ": ";
    for (size_t i = 0; i < count; ++i)
        std::cout << arr[i] << (i + 1 < count ? ", " : "\n");
}

size_t
prod(const size_t count, const size_t* arr)
{
    size_t res = 1;
    for (size_t i = 0; i < count; ++i)
        res *= arr[i];
    return res;
}

size_t
equals(const size_t count, const size_t* a, const size_t* b)
{
    for (size_t i = 0; i < count; ++i)
        if (a[i] != b[i])
            return false;
    return true;
}

static PackKernelParams
to_static_pack_kernel_params(const size_t ndims, const size_t* mm, const size_t* block_shape,
                             const size_t* block_offset, const size_t nbuffers, double* buffers[],
                             double* pack_buffer)
{
    PackKernelParams params;

    params.ndims = ndims;
    ncopy(ndims, mm, params.mm);
    ncopy(ndims, block_shape, params.block_shape);
    ncopy(ndims, block_offset, params.block_offset);

    params.nbuffers = nbuffers;
    ncopy(nbuffers, buffers, params.buffers);
    params.pack_buffer = pack_buffer;

    return params;
}

void
pack(const size_t ndims, const size_t* mm, const size_t* block_shape, const size_t* block_offset,
     const size_t ninputs, double* inputs[], double* output)
{
    PackKernelParams kp = to_static_pack_kernel_params(ndims, mm, block_shape, block_offset,
                                                       ninputs, inputs, output);
    print("ndims", kp.ndims);

    // assert(ndims == kp.ndims);
}

void
unpack(const double* input, const size_t ndims, const size_t* mm, const size_t* block_shape,
       const size_t* block_offset, const size_t noutputs, double* outputs[])
{
    // StaticPackInfo spi = to_static_pack_info(ndims, mm, block_shape, block_offset, noutputs,
    //                                          outputs);
    // unpack_(input, spi);
}

void
test_pack(void)
{
    std::cout << "Hello from pack\n";
    print("Test", 1.0);

    const size_t mm[]           = {8, 8};
    const size_t block_shape[]  = {6, 6};
    const size_t block_offset[] = {1, 1};
    const size_t ndims          = ARRAY_SIZE(mm);
    print_array("mm", ndims, mm);

    double* buffers[]     = {NULL};
    const size_t nbuffers = ARRAY_SIZE(buffers);

    double* pack_buffer = NULL;

    pack(ndims, mm, block_shape, block_offset, nbuffers, buffers, pack_buffer);
}
