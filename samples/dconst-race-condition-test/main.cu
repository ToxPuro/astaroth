/**
 * A program for checking whether it is possible to get a race condition
 * when writing to a device constant that is shared between two kernels.
 *
 *
 *    Building and running:
 *      nvcc ../samples/dconst-race-condition-test/main.cu && ./a.out
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__device__ __constant__ int dconst;

__global__ void
kernel(int* output, int* dummy_output)
{
    volatile int j = 0;
    for (int i = 0; i < 1000000000; ++i)
        j += i;

    *output       = dconst;
    *dummy_output = j; // For ensuring that the compiler does not optimize out the loop above
}

static void
timestamp(const char* msg)
{
    time_t ltime = time(NULL);
    printf("%s - %s", msg, asctime(localtime(&ltime)));
    fflush(stdout);
}

int
main(void)
{
    cudaStream_t stream0, stream1;
    const int aa = 1;
    const int bb = 2;
    int *cc, *dd;
    cudaMallocManaged((void**)&cc, 1);
    cudaMallocManaged((void**)&dd, 1);

    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);

    timestamp("Calling cudaMemcpyToSymbolAsync stream0");
    cudaMemcpyToSymbolAsync(dconst, &aa, 1, 0, cudaMemcpyHostToDevice, stream0);
    timestamp("Calling kernel stream0");
    kernel<<<1, 1, 0, stream0>>>(cc, dd);
    timestamp("Calling cudaMemcpyToSymbolAsync stream1");
    cudaMemcpyToSymbolAsync(dconst, &bb, 1, 0, cudaMemcpyHostToDevice, stream1);

    timestamp("Calling cudaDeviceSynchronize");
    cudaDeviceSynchronize();
    timestamp("Synchronized");

    printf("-------------\n");
    if (*cc == aa)
        printf("OK! %d == %d\n", *cc, aa);
    else
        printf("FAILURE: %d != %d\n", *cc, aa);
    printf("-------------\n");

    cudaStreamDestroy(stream0);
    cudaStreamDestroy(stream1);
    cudaFree(cc);
    return EXIT_SUCCESS;
}