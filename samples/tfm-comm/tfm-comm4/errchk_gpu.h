#pragma once

#include "errchk.h"

#define ERRCHK_GPU_API(errorcode)                                                                  \
    ((errorcode) == cudaSuccess ? 0                                                                \
                                : (errchk_raise_error(__func__, __FILE__, __LINE__, #errorcode,    \
                                                      cudaGetErrorString(errorcode)),              \
                                   -1))

#define ERRCHK_GPU_KERNEL()                                                                        \
    ((ERRCHK_GPU_API(cudaPeekAtLastError())) && (ERRCHK_GPU_API(cudaDeviceSynchronize())) == 0     \
         ? 0                                                                                       \
         : -1)

// #define ERRCHK_GPU_API(errorcode)                                                                  \
//     (((errorcode) == MPI_SUCCESS)                                                                  \
//          ? 0                                                                                       \
//          : ((handle_gpu_api_error((errorcode), __func__, __FILE__, __LINE__, #errorcode)), -1))

// #define ERRCHK_GPU(expr) ((expr) ? 0 : ((handle_error(__func__, __FILE__, __LINE__, #expr)), -1))

// #define ERRCHK_GPU_API(errorcode)                                                                  \
//     do {                                                                                           \
//         if ((errorcode) != cudaSuccess) {                                                          \
//             ERRCHKK((errorcode) == cudaSuccess, cudaGetErrorString(errorcode));                    \
//         }                                                                                          \
//     } while (0)

// #define ERRCHK_GPU_KERNEL()                                                                        \
//     do {                                                                                           \
//         ERRCHK_GPU_API(cudaPeekAtLastError());                                                     \
//         ERRCHK_GPU_API(cudaDeviceSynchronize());                                                   \
//     } while (0)
