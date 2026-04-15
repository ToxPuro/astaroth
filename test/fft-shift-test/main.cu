/*
    Copyright (C) 2014-2021, Johannes Pekkila, Miikka Vaisala.
    This file is part of Astaroth.
    Astaroth is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    Astaroth is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with Astaroth.  If not, see <http://www.gnu.org/licenses/>.
*/
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <complex>
#include <vector>
#include <cufftXt.h>
#include <cuComplex.h>
// #include "cufft_utils.h"
// #define STENCIL_ORDER (2)
#include "astaroth.h"
#include "astaroth_utils.h"
#include  "math_utils.h"
// CUDA API error checking
#ifndef CUDA_RT_CALL
#define CUDA_RT_CALL( call )                                                                                           \
    {                                                                                                                  \
        auto status = static_cast<cudaError_t>( call );                                                                \
        if ( status != cudaSuccess )                                                                                   \
            fprintf( stderr,                                                                                           \
                     "ERROR: CUDA RT call \"%s\" in line %d of file %s failed "                                        \
                     "with "                                                                                           \
                     "%s (%d).\n",                                                                                     \
                     #call,                                                                                            \
                     __LINE__,                                                                                         \
                     __FILE__,                                                                                         \
                     cudaGetErrorString( status ),                                                                     \
                     status );                                                                                         \
    }
#endif  // CUDA_RT_CALL
// cufft API error chekcing
#ifndef CUFFT_CALL
#define CUFFT_CALL( call )                                                                                             \
    {                                                                                                                  \
        auto status = static_cast<cufftResult>( call );                                                                \
        if ( status != CUFFT_SUCCESS )                                                                                 \
            fprintf( stderr,                                                                                           \
                     "ERROR: CUFFT call \"%s\" in line %d of file %s failed "                                          \
                     "with "                                                                                           \
                     "code (%d).\n",                                                                                   \
                     #call,                                                                                            \
                     __LINE__,                                                                                         \
                     __FILE__,                                                                                         \
                     status );                                                                                         \
    }
#endif  // CUFFT_CALL
// https://forums.developer.nvidia.com/t/additional-cucomplex-functions-cucnorm-cucsqrt-cucexp-and-some-complex-double-functions/36892
__host__ __device__ static __inline__ cuDoubleComplex cuCexp(cuDoubleComplex x)
{
	double factor = exp(x.x);
	return make_cuDoubleComplex(factor * cos(x.y), factor * sin(x.y));
}
/**
 * data: array with dimensions [size[0]][size[1]][size[2]]
 * : indexing is as [z][y][x]
 * sizes: {x, y, z} dimension of `data` buffer
 */
__global__
void cushift(cuDoubleComplex* data, const double shift_amount[3], const double sample_rate[3], const int sizes[3]) {
    const auto x = blockIdx.x * blockDim.x + threadIdx.x;
    const auto y = blockIdx.y * blockDim.y + threadIdx.y;
    const auto z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x > sizes[2] || y > sizes[1] || z > sizes[0]) return;
    const auto idx = ((z * sizes[1]) + y * sizes[2] ) + x;
    
    const double fullfreq[] = {
        1.0 / sample_rate[0],
        1.0 / sample_rate[1],
        1.0 / (2.0 * sample_rate[2]),
    };
    const double freq[] = {
        fullfreq[0] * static_cast<double>(z) / sizes[0],
        fullfreq[1] * static_cast<double>(y) / sizes[1],
        fullfreq[2] * static_cast<double>(x) / sizes[2],
    };
    const double pi = std::acos(-1.0);
    
    // -2j * pi * freq * shift_amount
    cuDoubleComplex exp[] = {
        make_cuDoubleComplex(0.0, -2.0 * pi * freq[0] * shift_amount[0]),
        make_cuDoubleComplex(0.0, -2.0 * pi * freq[1] * shift_amount[1]),
        make_cuDoubleComplex(0.0, -2.0 * pi * freq[2] * shift_amount[2])
    };
    // exp(-2j * pi * freq * shift_amount)
    cuDoubleComplex shift_factor[] = {
        cuCexp(exp[0]),
        cuCexp(exp[1]),
        cuCexp(exp[2]),
    };
    
    data[idx] = cuCmul(cuCmul(cuCmul(data[idx], shift_factor[0]), shift_factor[1]), shift_factor[2]);
}
void shift(double * buffer, const int3 size, const double shift_amount[3], const double sample_rate[3]) {
    const int dimension = 3;
    
    int * cusizes;
    double * cushift_amount, * cusample_rate;
    cudaMalloc(reinterpret_cast<void **>(&cushift_amount), dimension * sizeof(double));
    cudaMemcpy(cushift_amount, shift_amount, dimension * sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc(reinterpret_cast<void **>(&cusample_rate), dimension * sizeof(double));
    cudaMemcpy(cusample_rate, sample_rate, dimension * sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc(reinterpret_cast<void **>(&cusizes), dimension * sizeof(int));
    int dimensions[] = { size.z, size.y, ((size.x/2)+1)};    
    cudaMemcpy(cusizes, dimensions, dimension * sizeof(int), cudaMemcpyHostToDevice);
    // Copy input data to GPUs
    double *d_data = nullptr;
    size_t data_size = size.x * size.y * size.z;
    CUDA_RT_CALL(cudaMalloc(reinterpret_cast<void **>(&d_data), data_size * sizeof(double)));
    CUDA_RT_CALL(cudaMemcpy(d_data, buffer, data_size * sizeof(double), cudaMemcpyHostToDevice));
    size_t complex_domain_size = ((size.x/2)+1)* size.y * size.z;
    // NOTE: We need this, because in-place fft needs properly padded buffer.
    // Otherwise use buffer padded to (n/2 + 1) elements in least significant dimension.
    cufftDoubleComplex *d_fft = nullptr;
    CUFFT_CALL(cudaMalloc(reinterpret_cast<void **>(&d_fft), complex_domain_size * sizeof(cufftDoubleComplex)));
    // this should copy and pad the buffer. however I was not able to get correct results
    // CUDA_RT_CALL(cudaMemcpy2D(d_fft, complex_dimension * sizeof(cufftDoubleComplex), 
    //     candidate.vertex_buffer[current_buffer], dims[2] * sizeof(input_type),
    //     dims[2] * sizeof(input_type), dims[0] * dims[1],
    //     cudaMemcpyHostToDevice));
    acFFTForwardTransformSymmetricR2C(d_data, to_volume(size), to_volume(size), (Volume){0,0,0}, reinterpret_cast<AcComplex*>(d_fft));
    
    const auto BLOCK_SIZE = 6;
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((dimensions[2] / dimBlock.x) + 1, (size.y / dimBlock.y) + 1, (size.z / dimBlock.z) + 1);
    cushift<<<dimGrid, dimBlock>>>(d_fft, cushift_amount, cusample_rate, cusizes);
    acFFTBackwardTransformSymmetricC2R(reinterpret_cast<const AcComplex*>(d_fft), to_volume(size), to_volume(size), (Volume){0,0,0}, d_data);
    cudaFree(d_fft);
    CUDA_RT_CALL(cudaMemcpy(buffer, d_data, data_size * sizeof(double), cudaMemcpyDeviceToHost));
    cudaFree(d_data);
}
int
main(void)
{
    using input_type = double;
    
    AcMeshInfo info;
    acLoadConfig(AC_DEFAULT_CONFIG, &info);
    info[AC_nlocal].x = STENCIL_ORDER;
    info[AC_nlocal].y = STENCIL_ORDER;
    info[AC_nlocal].z = 20 - 2 * NGHOST;
    acHostUpdateParams(&info);
    // Sample rates for each dimensions
    info[AC_ds].x = 1.0 / static_cast<input_type>(info[AC_mlocal].x);
    info[AC_ds].x = 1.0 / static_cast<input_type>(info[AC_mlocal].y);
    info[AC_ds].x = 1.0 / static_cast<input_type>(info[AC_mlocal].z);
    acHostUpdateParams(&info);
    // Alloc
    AcMesh candidate;
    acHostMeshCreate(info, &candidate);
    // Init
    const auto current_buffer = VTXBUF_LNRHO;
    
    int3 nn = info[AC_nlocal];
    int3 mm = info[AC_mlocal];
    acGridInit(info);
    acPrintMeshInfo(info);
    
    std::printf("---\nInput array:\n");
    for (auto z = 0; z < mm.z; z++) {
        // std::printf("");
        for (auto y = 0; y < mm.y; y++) {
            // std::printf("");
            for (auto x = 0; x < mm.x; x++) {
                auto idx = acVertexBufferIdx(x, y, z, info);
                // std::printf("\nidx: %d\n", idx);
                const double dsx = info[AC_ds].x;
                const double dsy = info[AC_ds].y;
                const double dsz = info[AC_ds].z;
                const double two_pi = 2.0 * 3.1415;
                const double angle_x = two_pi * static_cast<input_type>(x) * dsx;
                const double angle_y = two_pi * static_cast<input_type>(y) * dsy;
                const double angle_z = two_pi * static_cast<input_type>(z) * dsz;
                candidate.vertex_buffer[current_buffer][idx] = static_cast<input_type>( sin( two_pi * angle_x ) * sin( two_pi * angle_y ) /* * sin( two_pi * angle_z ) */ );
                std::printf("%f", candidate.vertex_buffer[current_buffer][idx]);
                if (x != mm.x - 1) {std::printf("\t");}
            }
            std::printf("\n");
        }
        std::printf("\n");
    }
    std::printf("---\n");
    
    const double shift_amount[] = { 0, 0, -1 };
    const double sample_rate[] = { info[AC_ds].z, info[AC_ds].y, info[AC_ds].x };
    shift(candidate.vertex_buffer[current_buffer], mm, shift_amount, sample_rate);
    std::printf("---\nOutput array after Inverse FFT:\n");
    for (auto z = 0; z < mm.z; z++) {
        for (auto y = 0; y < mm.y; y++) {
            for (auto x = 0; x < mm.x; x++) {
                auto idx = acVertexBufferIdx(x, y, z, info);
                std::printf("%f", candidate.vertex_buffer[current_buffer][idx]);
                if (x != mm.x - 1) {std::printf("\t");}
            }
            std::printf("\n");
        }
        std::printf("\n");
    }
    std::printf("---\n");
    // Destroy
    acGridQuit();
    acHostMeshDestroy(&candidate);
    return EXIT_SUCCESS;
}
