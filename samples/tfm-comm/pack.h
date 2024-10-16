#pragma once
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void pack(const size_t ndims, const size_t* mm, const size_t* block_shape,
          const size_t* block_offset, const size_t ninputs, double* inputs[], double* output);

void unpack(double* input, const size_t ndims, const size_t* mm, const size_t* block_shape,
            const size_t* block_offset, const size_t noutputs, double* outputs[]);

void test_pack(void);

#ifdef __cplusplus
}
#endif
