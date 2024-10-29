#pragma once

#include "errchk.h"

#define ERRCHK_CUDA_API(errcode)                                                                   \
    do {                                                                                           \
        const cudaError_t _tmp_cuda_api_errcode_ = (errcode);                                      \
        if (_tmp_cuda_api_errcode_ != CUDA_SUCCESS) {                                              \
            errchk_print_error(__func__, __FILE__, __LINE__, #errcode,                             \
                               cudaGetErrorString(_tmp_cuda_api_errcode_));                        \
            errchk_print_stacktrace();                                                             \
            throw std::runtime_error("CUDA API error");                                            \
        }                                                                                          \
    } while (0)

#define ERRCHK_CUDA_KERNEL()                                                                       \
    do {                                                                                           \
        ERRCHK_CUDA_API(cudaPeekAtLastError());                                                    \
        ERRCHK_CUDA_API(cudaDeviceSynchronize());                                                  \
    } while (0)
