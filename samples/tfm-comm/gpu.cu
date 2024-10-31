#include <cstdlib>

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#elif defined(__HIP_PLATFORM_AMD__)
#include "hip.h"
#include <hip/hip_runtime.h>
#else
static_assert(false);
#endif

#include <iostream>
#include <vector>

#include "errchk.h"
#include "errchk_cuda.h"
#include "static_array.h"

// #include "device_buffer.h"
// #include "dbuffer.h"
#include "buffer.h"
#include "buffer_transfer.h"

__global__ void
kernel(const size_t count, const double* in, double* out)
{
    const size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < count)
        out[i] = 2 * in[i];
}

int
main()
{
    const size_t count = 10;
    Buffer<double> host_buffer(count);
    host_buffer.fill_arange(0, count);
    std::cout << "Before: " << host_buffer << std::endl;
    Buffer<double> device_buffer(count, BUFFER_DEVICE);
    // host_buffer.migrate(device_buffer);
    // device_buffer.migrate(host_buffer);
    HostToDeviceBufferExchangeTask<double> htod(count);
    htod.launch(host_buffer);
    htod.wait(device_buffer);
    HostToDeviceBufferExchangeTask<double> dtoh(count);
    dtoh.launch(device_buffer);
    dtoh.wait(host_buffer);
    std::cout << "After: " << host_buffer << std::endl;

    double* hin  = (double*)malloc(count * sizeof(hin[0]));
    double* hout = (double*)malloc(count * sizeof(hout[0]));
    ERRCHK(hin);
    ERRCHK(hout);

    double *din, *dout;
    ERRCHK_CUDA_API(cudaMalloc(&din, count * sizeof(din[0])));
    ERRCHK_CUDA_API(cudaMalloc(&dout, count * sizeof(dout[0])));

    for (size_t i = 0; i < count; ++i)
        hin[i] = static_cast<double>(i);

    ERRCHK_CUDA_API(cudaMemcpy(din, hin, count * sizeof(hin[0]), cudaMemcpyHostToDevice));
    const size_t tpb = 256;
    const size_t bpg = (count + tpb - 1) / tpb;
    kernel<<<bpg, tpb>>>(count, din, dout);
    ERRCHK_CUDA_KERNEL();
    ERRCHK_CUDA_API(cudaMemcpy(hout, dout, count * sizeof(dout[0]), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < count; ++i)
        std::cout << "i: " << hout[i] << std::endl;

    ERRCHK_CUDA_API(cudaFree(dout));
    ERRCHK_CUDA_API(cudaFree(din));
    free(hout);
    free(hin);
    return EXIT_SUCCESS;
}

// int
// main_draft(void)
// {
//     const size_t count = 10;
//     Buffer<double> hin(count);
//     Buffer<double> hout(count);
//     // Buffer<double> din(count, true); // count, on_device, pinned
//     // Buffer<double> dout(count, true);
//     // Buffer<double> din(count, true);
//     din.pin();
//     din.unpin();
//     Buffer::migrate_async();
//     Buffer::sync();

//     hin.fill_arange();
//     Buffer::migrate(hin, din);
//     // Kernel
//     Buffer::migrate(dout, hout);

//     return EXIT_SUCCESS;
// }
