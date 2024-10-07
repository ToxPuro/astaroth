#include "pack.h"

#include "errchk.h"
#include "misc.h"

#include <cassert>
#include <iostream>
#include <cstring>

#define MAX_NDIMS (3)
#define MAX_NBUFFERS (7)
#define range(i, min, max) (size_t i = (min); (i) < (max); ++(i))

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
    ERRCHK(in != NULL);
    ERRCHK(out != NULL);
    memmove(out, in, sizeof(in[0]) * count);
}

template <typename T>
static void
print(const char* label, const T value)
{
    std::cout << label << ": " << value << "\n";
}

template <typename T>
static void
print_array(const char* label, const size_t count, const T* arr)
{
    std::cout << label << ": ";
    for (size_t i = 0; i < count; ++i)
        std::cout << arr[i] << (i + 1 < count ? ", " : "\n");
}

#define printd(x) print(#x, (x))
#define printd_array(count, arr) print_array(#arr, (count), (arr))

static size_t
prod(const size_t count, const size_t* arr)
{
    ERRCHK(arr != NULL);
    size_t res = 1;
    for (size_t i = 0; i < count; ++i)
        res *= arr[i];
    return res;
}

template <typename T>
static bool
equals(const size_t count, const T* a, const T* b)
{
    ERRCHK(a != NULL);
    ERRCHK(b != NULL);
    for (size_t i = 0; i < count; ++i)
        if (a[i] != b[i])
            return false;
    return true;
}

static void
add_arrays(const size_t count, const size_t* a, const size_t* b, size_t* c)
{
    for (size_t i = 0; i < count; ++i)
        c[i] = a[i] + b[i];
}

static void
to_spatial(const size_t index, const size_t ndims, const size_t* shape, size_t* output)
{
    for (size_t j = 0; j < ndims; ++j) {
        size_t divisor = 1;
        for (size_t i = 0; i < j; ++i)
            divisor *= shape[i];
        output[j] = (index / divisor) % shape[j];
    }
}

static size_t
to_linear(const size_t ndims, const size_t* index, const size_t* shape)
{
    size_t result = 0;
    for (size_t j = 0; j < ndims; ++j) {
        size_t factor = 1;
        for (size_t i = 0; i < j; ++i)
            factor *= shape[i];
        result += index[j] * factor;
    }
    return result;
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

static void pack_(const PackKernelParams params)
{
    ERRCHK(params.pack_buffer != NULL);
    ERRCHK(params.buffers != NULL);

    const size_t block_elems = prod(params.ndims, params.block_shape);
    for (size_t i = 0; i < block_elems; ++i){
        for (size_t j = 0; j < params.nbuffers; ++j) {
            // Block coords
            size_t block_coords[MAX_NDIMS];
            to_spatial(i, params.ndims, params.block_shape, block_coords);

            // Input coords
            size_t in_coords[MAX_NDIMS];
            add_arrays(params.ndims, params.block_offset, block_coords, in_coords);
            
            const size_t in_idx = to_linear(params.ndims, in_coords, params.mm);
            ERRCHK(in_idx < prod(params.ndims, params.mm));
            ERRCHK(in_idx < prod(params.ndims, params.mm));
            ERRCHK(params.buffers[j] != NULL);

            params.pack_buffer[i + j * block_elems] = params.buffers[j][in_idx];
        }
    }
}

static void unpack_(const PackKernelParams params)
{
    ERRCHK(params.pack_buffer != NULL);
    ERRCHK(params.buffers != NULL);

    const size_t block_elems = prod(params.ndims, params.block_shape);
    for (size_t i = 0; i < block_elems; ++i){
        for (size_t j = 0; j < params.nbuffers; ++j) {
            // Block coords
            size_t block_coords[MAX_NDIMS];
            to_spatial(i, params.ndims, params.block_shape, block_coords);

            // Output coords
            size_t out_coords[MAX_NDIMS];
            add_arrays(params.ndims, params.block_offset, block_coords, out_coords);

            const size_t out_idx = to_linear(params.ndims, out_coords, params.mm);
            ERRCHK(out_idx < prod(params.ndims, params.mm));
            ERRCHK(params.buffers[j] != NULL);

            params.buffers[j][out_idx] = params.pack_buffer[i + j * block_elems];
        }
    }
}

void
pack(const size_t ndims, const size_t* mm, const size_t* block_shape, const size_t* block_offset,
     const size_t ninputs, double* inputs[], double* output)
{
    PackKernelParams kp = to_static_pack_kernel_params(ndims, mm, block_shape, block_offset,
                                                       ninputs, inputs, output);
    // printd(kp.ndims);
    // printd(ninputs);

    ERRCHK(ndims == kp.ndims);
    ERRCHK(equals(ndims, mm, kp.mm));
    ERRCHK(equals(ndims, block_shape, kp.block_shape));
    ERRCHK(equals(ndims, block_offset, kp.block_offset));
    ERRCHK(ninputs == kp.nbuffers);
    ERRCHK(equals(ninputs, inputs, kp.buffers));
    ERRCHK(output == kp.pack_buffer);

    // printd_array(ninputs, inputs);
    // printd_array(kp.nbuffers, kp.buffers);
    // printd(kp.buffers[0] == inputs[0]);

    pack_(kp);
}

void
unpack(double* input, const size_t ndims, const size_t* mm, const size_t* block_shape,
       const size_t* block_offset, const size_t noutputs, double* outputs[])
{
    PackKernelParams kp = to_static_pack_kernel_params(ndims, mm, block_shape, block_offset,
                                                       noutputs, outputs, input);
    unpack_(kp);
}

void
test_pack(void)
{

    const size_t mm[]           = {8, 8};
    const size_t block_shape[]  = {6, 6};
    const size_t block_offset[] = {1, 1};
    const size_t ndims          = ARRAY_SIZE(mm);
    // print_array("mm", ndims, mm);

    const size_t mm_count = prod(ndims, mm);
    double* buf0 = new double[mm_count];
    double* buf1 = new double[mm_count];
    double* buf2 = new double[mm_count];

    for range(i, 0, mm_count) {
        buf0[i] = double(i + 0);
        buf1[i] = double(i + 1*mm_count);
        buf2[i] = double(i + 2*mm_count);
    }

    double* buffers[]     = {buf0, buf1, buf2};
    const size_t nbuffers = ARRAY_SIZE(buffers);

    const size_t pack_count = nbuffers * prod(ndims, block_shape);
    double* pack_buffer = new double[pack_count];

    pack(ndims, mm, block_shape, block_offset, nbuffers, buffers, pack_buffer);
    unpack(pack_buffer, ndims, mm, block_shape, block_offset, nbuffers, buffers);

    // printd_array(mm_count, buf0);
    // printd_array(mm_count, buf1);
    // printd_array(mm_count, buf2);
    // printd_array(pack_count, pack_buffer);

    delete pack_buffer;
    delete buf0;
    delete buf1;
    delete buf2;
}
