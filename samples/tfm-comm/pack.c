#include "pack.h"

#include "math_utils.h"
#include "misc.h"
#include "nalloc.h"
#include "print.h"

static void
check_input_valid(const size_t ndims, const size_t* input_dims, const size_t* input_offset,
                  const size_t* output_dims)
{
    size_t* max_coords;
    nalloc(ndims, max_coords);

    add_arrays(ndims, input_offset, output_dims, max_coords);
    subtract_value(1, ndims, max_coords);
    ERRCHK(all_less_than(ndims, max_coords, input_dims));

    ndealloc(max_coords);
}

void
segment_copy(const size_t ndims,                                                        //
             const size_t* input_dims, const size_t* input_offset, const double* input, //
             const size_t* output_dims, const size_t* output_offset, double* output)
{
    check_input_valid(ndims, input_dims, input_offset, output_dims);

    size_t *coords, *out_coords, *in_coords;
    nalloc(ndims, coords);
    nalloc(ndims, out_coords);
    nalloc(ndims, in_coords);

    const size_t count = prod(ndims, output_dims);
    for (size_t i = 0; i < count; ++i) {
        to_spatial(i, ndims, output_dims, coords);
        add_arrays(ndims, output_offset, coords, out_coords);
        add_arrays(ndims, input_offset, coords, in_coords);

        const size_t out_idx = to_linear(ndims, out_coords, output_dims);
        const size_t in_idx  = to_linear(ndims, in_coords, input_dims);
        output[out_idx]      = input[in_idx];
    }
    ndealloc(coords);
    ndealloc(out_coords);
    ndealloc(in_coords);
}

void
test_pack(void)
{
    {
        const size_t input_dims[]    = {8, 8};
        const size_t input_offset[]  = {0, 0};
        const size_t output_dims[]   = {2, 2};
        const size_t output_offset[] = {0, 0};
        const size_t ndims           = ARRAY_SIZE(input_dims);
        const size_t input_count     = prod(ndims, input_dims);
        const size_t output_count    = prod(ndims, output_dims);
        const double model_output[]  = {0, 1, 8, 9};

        double *input, *output;
        nalloc(input_count, input);
        nalloc(output_count, output);

        for (size_t i = 0; i < input_count; ++i)
            input[i] = (double)i;

        segment_copy(ndims, input_dims, input_offset, input, output_dims, output_offset, output);
        ERRCHK(ncmp(ndims, output, model_output));

        // print_ndarray("input", ndims, input_dims, input);
        // print_ndarray("output", ndims, output_dims, output);

        ndealloc(input);
        ndealloc(output);
    }
    {
        const size_t input_dims[]    = {4, 3, 2};
        const size_t input_offset[]  = {1, 1, 0};
        const size_t output_dims[]   = {2, 1, 2};
        const size_t output_offset[] = {0, 0, 0};
        const size_t ndims           = ARRAY_SIZE(input_dims);
        const size_t input_count     = prod(ndims, input_dims);
        const size_t output_count    = prod(ndims, output_dims);
        const double model_output[]  = {5, 6, 17, 18};

        double *input, *output;
        nalloc(input_count, input);
        nalloc(output_count, output);

        for (size_t i = 0; i < input_count; ++i)
            input[i] = (double)i;

        segment_copy(ndims, input_dims, input_offset, input, output_dims, output_offset, output);
        ERRCHK(ncmp(ndims, output, model_output));

        // print_ndarray("input", ndims, input_dims, input);
        // print_ndarray("output", ndims, output_dims, output);

        ndealloc(input);
        ndealloc(output);
    }
    {
        const size_t input_dims[]    = {8, 7, 4};
        const size_t input_offset[]  = {2, 2, 1};
        const size_t output_dims[]   = {3, 2, 3};
        const size_t output_offset[] = {0, 0, 0};
        const size_t ndims           = ARRAY_SIZE(input_dims);
        const size_t input_count     = prod(ndims, input_dims);
        const size_t output_count    = prod(ndims, output_dims);
        const double model_output[]  = {
            74, 75, 76, 82, 83, 84, 45, 46, 47, 53, 54, 55, 186, 187, 188, 194, 195, 196,
        };

        double *input, *output;
        nalloc(input_count, input);
        nalloc(output_count, output);

        for (size_t i = 0; i < input_count; ++i)
            input[i] = (double)i;

        segment_copy(ndims, input_dims, input_offset, input, output_dims, output_offset, output);
        segment_copy(ndims, input_dims, (size_t[]){5, 5, 0}, input, (size_t[]){3, 2, 1},
                     (size_t[]){0, 0, 1}, output);
        ERRCHK(ncmp(ndims, output, model_output));

        // print_ndarray("input", ndims, input_dims, input);
        // print_ndarray("output", ndims, output_dims, output);

        ndealloc(input);
        ndealloc(output);
    }
}
