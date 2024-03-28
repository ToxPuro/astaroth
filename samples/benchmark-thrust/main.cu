#include <stdio.h>
#include <stdlib.h>

#include "acc_runtime.h"
#include "timer_hires.h"

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

#include "astaroth_utils.h"

typedef struct {
    AcReal* data;
    size_t mx, my, mz, mw;
    bool on_device;
} AcTensor;

__host__ __device__ size_t
acTensorCount(const AcTensor& tensor)
{
    return tensor.mx * tensor.my * tensor.mz * tensor.mw;
}

typedef struct {
    size_t i, j, k, l;
} AcIndex;

__host__ __device__ size_t
acTensorIdx(const AcTensor& tensor, const AcIndex& idx)
{
    return idx.i +                         //
           idx.j * tensor.mx +             //
           idx.k * tensor.mx * tensor.my + //
           idx.l * tensor.mx * tensor.my * tensor.mz;
}

__host__ __device__ bool
acTensorIdxWithinBounds(const AcTensor& tensor, const AcIndex& idx)
{
    return (idx.i < tensor.mx) && //
           (idx.j < tensor.my) && //
           (idx.k < tensor.mz) && //
           (idx.l < tensor.mw);
}

__host__ __device__ bool
acTensorIdxWithinRange(const AcIndex& idx, const AcIndex& end)
{
    return (idx.i < end.i) && //
           (idx.j < end.j) && //
           (idx.k < end.k) && //
           (idx.l < end.l);
}

AcTensor
acTensorCreate(const size_t mx, const size_t my, const size_t mz, const size_t mw)
{
    AcTensor tensor = {
        .mx        = mx,
        .my        = my,
        .mz        = mz,
        .mw        = mw,
        .on_device = true,
    };
    const size_t bytes = sizeof(tensor.data[0]) * acTensorCount(tensor);
    ERRCHK_CUDA_ALWAYS(cudaMalloc((void**)&tensor.data, bytes));
    return tensor;
}

void
acTensorDestroy(AcTensor* t)
{
    cudaFree(t->data);
    t->data = NULL;
    t->mx   = 0;
    t->my   = 0;
    t->mz   = 0;
    t->mw   = 0;
}

__global__ void
reduce_dummy(const AcTensor input, AcTensor output)
{
    const AcIndex vertexIdx = (AcIndex){
        .i = threadIdx.x + blockIdx.x * blockDim.x,
        .j = threadIdx.y + blockIdx.y * blockDim.y,
        .k = threadIdx.z + blockIdx.z * blockDim.z,
        .l = 0, // Todo derive from 1D index
    };
    if (!acTensorIdxWithinBounds(input, vertexIdx))
        return;

    const size_t idx = acTensorIdx(input, vertexIdx);

    output.data[idx] = input.data[idx];
}

template <typename T> struct square {
    __host__ __device__ T operator()(const T& x) const { return x * x; }
};

static void
benchmark_thrust(const size_t mx, const size_t my, const size_t mz)
{
    std::vector<thrust::device_vector<double>> inputs(mz);
    for (size_t k = 0; k < mz; ++k)
        inputs[k] = thrust::device_vector<double>(mx * my, 1);

    thrust::device_vector<double> results(mz);

    for (size_t k = 0; k < mz; ++k)
        results[k] = thrust::reduce(inputs[k].begin(), inputs[k].end());

    Timer t;
    cudaDeviceSynchronize();
    timer_reset(&t);
    for (size_t k = 0; k < mz; ++k)
        results[k] = thrust::reduce(inputs[k].begin(), inputs[k].end());
    cudaDeviceSynchronize();
    timer_diff_print(t);
}

int
main(int argc, char** argv)
{
    cudaProfilerStop();

    fprintf(stderr, "Usage: ./benchmark-thrust <nx> <ny> <nz>\n");
    const size_t nx     = (argc > 1) ? (size_t)atol(argv[1]) : 32;
    const size_t ny     = (argc > 2) ? (size_t)atol(argv[2]) : 32;
    const size_t nz     = (argc > 3) ? (size_t)atol(argv[3]) : 32;
    const size_t radius = STENCIL_ORDER / 2;
    const size_t mx     = nx + 2 * radius;
    const size_t my     = ny + 2 * radius;
    const size_t mz     = nz + 2 * radius;

    printf("Input parameters:\n");
    printf("\tnx: %zu\n", nx);
    printf("\tny: %zu\n", ny);
    printf("\tnz: %zu\n", nz);
    printf("\tradius: %zu\n", radius);
    printf("\tmx: %zu\n", mx);
    printf("\tmy: %zu\n", my);
    printf("\tmz: %zu\n", mz);

    // Device
    cudaSetDevice(0);

#if 0
    cudaProfilerStart();
    benchmark_thrust(mx, my, mz);
    cudaProfilerStop();
#endif
#if 0
    // Mesh info
    AcMeshInfo info;
    acLoadConfig(AC_DEFAULT_CONFIG, &info);
    acSetMeshDims(nx, ny, nz, &info);
    acLoadMeshInfo(info, 0);

    // Random
    const size_t seed  = 12345;
    const size_t pid   = 0;
    const size_t count = acVertexBufferCompdomainSize(info);
    acRandInitAlt(seed, count, pid);

    AcMeshDims dims       = acGetMeshDims(info);
    VertexBufferArray vba = acVBACreate(mx, my, mz);
    acVBAReset(0, &vba);
    acLaunchKernel(randomize, 0, dims.n0, dims.n1, vba);
    acVBASwapBuffers(&vba);

    ProfileBufferArray pba = acPBACreate(mz);
    AcBufferArray ba0      = acBufferArrayCreate(12, nx * ny);
    AcBufferArray ba1      = acBufferArrayCreate(12, nx * ny);
    AcBufferArray ba2      = acBufferArrayCreate(12, nx * ny);
    // acMapCross(vba, 0, (int3){radius, radius, 0}, (int3){radius + nx, radius + ny, 1},
    // scratchpad);
    acMapCrossReduce(vba, 0, ba0, ba1, ba2, pba);

    Timer t;

    cudaProfilerStart();
    cudaDeviceSynchronize();
    timer_reset(&t);
    // for (size_t k = 0; k < mz; ++k)
    // acMapCross(vba, 0, (int3){radius, radius, k}, (int3){radius + nx, radius + ny, k +
    // 1},
    //            scratchpad);
    acMapCrossReduce(vba, 0, ba0, ba1, ba2, pba);

    cudaDeviceSynchronize();
    timer_diff_print(t);
    cudaProfilerStop();
    cudaDeviceSynchronize();

    ERRCHK_CUDA_KERNEL_ALWAYS();
    acRandQuit();
    acBufferArrayDestroy(&ba2);
    acBufferArrayDestroy(&ba1);
    acBufferArrayDestroy(&ba0);
    acPBADestroy(&pba);
    acVBADestroy(&vba);
#endif

#if 1
    const double gib         = 1024 * 1024 * 1024;
    const size_t num_fields  = 4 + 12;
    const size_t field_bytes = 2 * num_fields * mx * my * mz * sizeof(AcReal);
    printf("Total field memory: %g GiB\n", field_bytes / gib);

    const size_t num_averages  = 16;
    const size_t count         = nx * ny * mz * num_averages;
    const size_t average_bytes = 2 * sizeof(AcReal) * count;
    printf("Total average memory: %g GiB\n", average_bytes / gib);

    // AcTensor input  = acTensorCreate(nx, ny, mz, num_averages);
    // AcTensor output = acTensorCreate(nx, ny, mz, num_averages);
    AcTensor input  = acTensorCreate(nx, ny, mz * num_averages, 1);
    AcTensor output = acTensorCreate(nx, ny, mz * num_averages, 1);
    // AcTensor input  = acTensorCreate(num_fields * nx, ny, mz, 1);
    // AcTensor output = acTensorCreate(num_fields * nx, ny, mz, 1);

    Timer t;
    timer_reset(&t);

    const dim3 tpb = (dim3){64, 8, 1};
    // Integer round-up division
    const dim3 bpg = (dim3){
        (input.mx + tpb.x - 1) / tpb.x,
        (input.my + tpb.y - 1) / tpb.y,
        (input.mz + tpb.z - 1) / tpb.z,
    };
    reduce_dummy<<<bpg, tpb>>>(input, output);
    cudaDeviceSynchronize();
    timer_diff_print(t);

    acTensorDestroy(&input);
    acTensorDestroy(&output);
#endif

    return EXIT_SUCCESS;
}