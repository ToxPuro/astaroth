#pragma once

#include "errchk.h"

#define ERRCHK_GPU_API(errorcode)                                                                  \
    if ((errorcode) != cudaSuccess) {                                                              \
        ERRCHKK((errorcode) == cudaSuccess, cudaGetErrorString(errorcode));                        \
    }

#define ERRCHK_GPU_KERNEL()                                                                        \
    {                                                                                              \
        ERRCHK_GPU_API(cudaPeekAtLastError());                                                     \
        ERRCHK_GPU_API(cudaDeviceSynchronize());                                                   \
    }
