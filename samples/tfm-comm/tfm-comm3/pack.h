#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void pack(const size_t ndims, const uint64_t* mm, const uint64_t* block_shape,
          const uint64_t* block_offset, const size_t ninputs, double* inputs[], double* output);

void unpack(double* input, const size_t ndims, const uint64_t* mm, const uint64_t* block_shape,
            const uint64_t* block_offset, const size_t noutputs, double* outputs[]);

void test_pack(void);

#ifdef __cplusplus
}
#endif
