#include <iostream>

#include <array>

#include <algorithm>
#include <cuda_runtime.h>

#include <stdio.h>

#include "errchk_gpu.h"

constexpr size_t COUNT = 3;

template <typename T, size_t N> struct StaticArray {
    T data[N];

    static_assert(sizeof(T) * N <= 1024,
                  "Warning: tried to stack-allocate an array larger than 1024 bytes.");

    __host__ __device__ constexpr size_t count(void) const { return N; }

    __host__ __device__ T& operator[](size_t i) { return data[i]; }
    __host__ __device__ const T& operator[](size_t i) const { return data[i]; }
};

template <typename T, size_t N>
typename std::enable_if<std::is_arithmetic<T>::value, StaticArray<T, N>>::type __host__ __device__
operator+(const StaticArray<T, N>& a, const StaticArray<T, N>& b)
{
    StaticArray<T, N> c;
    for (size_t i = 0; i < N; ++i)
        c[i] = a[i] + b[i];
    return c;
}

// __global__ void
// kernel(const size_t count, const double* in, double* out, StaticArray<double, 3> arr)
// {
//     const int i = threadIdx.x + blockIdx.x * blockDim.x;
//     if (i < count)
//         out[i] = arr.data[2];
// }

struct SomeRandomStruct {
    int something;
    size_t something_else;
    StaticArray<uint64_t, COUNT> maybe_a_static_array_member;
};

template <typename T, size_t N>
__global__ void
kernel(const StaticArray<T, N> in, double* out, const StaticArray<SomeRandomStruct, 2> lala)
{
    const int i            = threadIdx.x + blockIdx.x * blockDim.x;
    StaticArray<T, N> test = in + in;
    if (i < in.count())
        out[i] = in[i] + lala[1].maybe_a_static_array_member[1];
}

int
main(void)
{
    // const size_t count = 10;
    // const size_t bytes = count * sizeof(double);
    // double* in         = (double*)malloc(bytes);
    // double* out        = (double*)malloc(bytes);

    // for (size_t i = 0; i < count; ++i)
    //     in[i] = (double)i;

    // for (size_t i = 0; i < count; ++i)
    //     printf("%zu: %g\n", i, in[i]);

    // // for (size_t i = 0; i < count; ++i)
    // //     out[i] = in[i];
    // double *din, *dout;
    // ERRCHK_GPU_API(cudaMalloc(&din, bytes));
    // ERRCHK_GPU_API(cudaMalloc(&dout, bytes));

    // ERRCHK_GPU_API(cudaMemcpy(din, in, bytes, cudaMemcpyHostToDevice));
    // kernel<<<1, 10>>>(count, din, dout);
    // ERRCHK_GPU_KERNEL();
    // ERRCHK_GPU_API(cudaDeviceSynchronize());
    // ERRCHK_GPU_API(cudaMemcpy(out, dout, bytes, cudaMemcpyDeviceToHost));
    // ERRCHK_GPU_API(cudaDeviceSynchronize());

    // ERRCHK_GPU_API(cudaFree(din));
    // ERRCHK_GPU_API(cudaFree(dout));

    // for (size_t i = 0; i < count; ++i)
    //     printf("%zu: %g\n", i, out[i]);

    // free(in);
    // free(out);

    // std::cout << "test\n" << std::endl;

    // StaticArray<double, 3> arr = {1, 2, 3};
    // printf("val %g\n", arr.data[2]);

    // const size_t count = 10;
    // double *in, *out;
    // cudaMallocManaged(&in, count * sizeof(in[0]));
    // cudaMallocManaged(&out, count * sizeof(out[0]));

    // for (size_t i = 0; i < count; ++i) {
    //     in[i] = (double)i;
    //     printf("%zu: %g\n", i, in[i]);
    // }

    // kernel<<<1, count>>>(count, in, out, arr);
    // cudaDeviceSynchronize();

    // for (size_t i = 0; i < count; ++i)
    //     printf("%zu: %g\n", i, out[i]);

    // cudaFree(in);
    // cudaFree(out);

    StaticArray<double, COUNT> in;
    for (size_t i = 0; i < in.count(); ++i) {
        in[i] = (double)i;
        printf("%zu: %g\n", i, in.data[i]);
    }

    StaticArray<SomeRandomStruct, 2> lala;
    lala[1].maybe_a_static_array_member[1] = 500;

    double* out;
    cudaMallocManaged(&out, in.count() * sizeof(out[0]));

    kernel<double, COUNT><<<1, in.count()>>>(in, out, lala);
    cudaDeviceSynchronize();

    for (size_t i = 0; i < in.count(); ++i)
        printf("%zu: %g\n", i, out[i]);

    cudaFree(out);
    return EXIT_SUCCESS;
}
