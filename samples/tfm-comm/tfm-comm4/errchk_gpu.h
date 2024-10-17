#pragma once

#include "errchk.h"

#define ERRCHK_GPU_API(errorcode)                                                                  \
    do {                                                                                           \
        if ((errorcode) != cudaSuccess) {                                                          \
            ERRCHKK((errorcode) == cudaSuccess, cudaGetErrorString(errorcode));                    \
        }                                                                                          \
    } while (0)

#define ERRCHK_GPU_KERNEL()                                                                        \
    do {                                                                                           \
        ERRCHK_GPU_API(cudaPeekAtLastError());                                                     \
        ERRCHK_GPU_API(cudaDeviceSynchronize());                                                   \
    } while (0)
