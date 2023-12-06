#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <cudnn.h>

#include "timer_hires.h"

#define CONV_TYPE CUDNN_CROSS_CORRELATION

#define USE_DOUBLE (0)
#if USE_DOUBLE
static cudnnDataType_t dtype = CUDNN_DATA_DOUBLE;
typedef double real;
#else
static cudnnDataType_t dtype = CUDNN_DATA_FLOAT;
typedef float real;
#endif

static inline void
cudnn_errchk(cudnnStatus_t code, const char* file, int line, bool abort)
{
  if (code != CUDNN_STATUS_SUCCESS) {
    time_t terr;
    time(&terr);
    fprintf(stderr, "%s", ctime(&terr));
    fprintf(stderr, "\tcuDNN error in file %s line %d: %d\n", file, line, code);
    fflush(stderr);
  }
}

#define ERRCHK(params)                                                         \
  {                                                                            \
    cudnn_errchk((params), __FILE__, __LINE__, true);                          \
  }

int
main(void)
{

  int ndevices;
  cudaGetDeviceCount(&ndevices);
  printf("Devices: %d\n", ndevices);
  assert(ndevices == 1);

  const int device = 0;
  cudaSetDevice(device);

  cudnnHandle_t nn;
  cudnnCreate(&nn);

  cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW;

  // Input
  const size_t fn = 1;
  const size_t fc = 1;
  const size_t fh = 4096;
  const size_t fw = 4096;
  cudnnTensorDescriptor_t input_desc;
  ERRCHK(cudnnCreateTensorDescriptor(&input_desc));
  ERRCHK(cudnnSetTensor4dDescriptor(input_desc, format, dtype, fn, fc, fh, fw));

  real* input;
  const size_t input_count = fn * fc * fh * fw;
  const size_t input_bytes = input_count * sizeof(input[0]);
  cudaMalloc((void**)&input, input_bytes);

  // Host
  real* host_input = (real*)malloc(input_bytes);
  for (size_t i = 0; i < input_count; ++i)
    host_input[i] = 1;
  cudaMemcpy(input, host_input, input_bytes, cudaMemcpyHostToDevice);

  // Kernel
  const size_t gk = 1;
  const size_t gc = 1;
  const size_t gh = 3;
  const size_t gw = 3;
  cudnnFilterDescriptor_t filter_desc;
  ERRCHK(cudnnCreateFilterDescriptor(&filter_desc));
  ERRCHK(
      cudnnSetFilter4dDescriptor(filter_desc, dtype, format, gk, gc, gh, gw));

  real* filter;
  const size_t filter_count = gk * gc * gh * gw;
  const size_t filter_bytes = filter_count * sizeof(filter[0]);
  cudaMalloc((void**)&filter, filter_bytes);

  // Host
  real* host_filter = (real*)malloc(filter_bytes);
  for (size_t i = 0; i < filter_count; ++i)
    host_filter[i] = 1;
  cudaMemcpy(filter, host_filter, filter_bytes, cudaMemcpyHostToDevice);

  // Convolution
  const size_t pad_h = 1;
  const size_t pad_w = 1;
  const size_t str_h = 1;
  const size_t str_w = 1;
  const size_t dil_h = 1;
  const size_t dil_w = 1;
  cudnnConvolutionDescriptor_t convolution_desc;
  ERRCHK(cudnnCreateConvolutionDescriptor(&convolution_desc));
  ERRCHK(cudnnSetConvolution2dDescriptor(convolution_desc, pad_h, pad_w, str_h,
                                         str_w, dil_h, dil_w, CONV_TYPE,
                                         dtype));

  // Output
  int fn_out;
  int fc_out;
  int fh_out;
  int fw_out;
  ERRCHK(cudnnGetConvolution2dForwardOutputDim(convolution_desc, input_desc,
                                               filter_desc, &fn_out, &fc_out,
                                               &fh_out, &fw_out));
  cudnnTensorDescriptor_t output_desc;
  ERRCHK(cudnnCreateTensorDescriptor(&output_desc));
  ERRCHK(cudnnSetTensor4dDescriptor(output_desc, format, dtype, fn_out, fc_out,
                                    fh_out, fw_out));
  real* output;
  const size_t output_count = fn_out * fc_out * fh_out * fw_out;
  const size_t output_bytes = output_count * sizeof(output[0]);
  cudaMalloc((void**)&output, output_bytes);

  // Algorithm
  //   const cudnnConvolutionFwdAlgo_t
  //       algorithm = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
  // const cudnnConvolutionFwdAlgo_t algorithm =
  // CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM; const
  // cudnnConvolutionFwdAlgo_t algorithm = CUDNN_CONVOLUTION_FWD_ALGO_FFT;
  // const cudnnConvolutionFwdAlgo_t algorithm =
  // CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;
  const int requested_algorithms = 1;
  cudnnConvolutionFwdAlgoPerf_t algorithms[requested_algorithms];
  int returned_algorithms;
  ERRCHK(cudnnFindConvolutionForwardAlgorithm(
      nn, input_desc, filter_desc, convolution_desc, output_desc,
      requested_algorithms, &returned_algorithms, algorithms));
  assert(returned_algorithms == requested_algorithms);
  const cudnnConvolutionFwdAlgo_t algorithm = algorithms[0].algo;

  // Workspace
  size_t workspace_size;
  ERRCHK(cudnnGetConvolutionForwardWorkspaceSize(nn, input_desc, filter_desc,
                                                 convolution_desc, output_desc,
                                                 algorithm, &workspace_size));

  real* workspace;
  cudaMalloc((void**)&workspace, workspace_size);

  // Compute ---------------------------------------
  const real alpha = 1;
  const real beta  = 0;

  // Warmup
  for (size_t i = 0; i < 10; ++i)
    ERRCHK(cudnnConvolutionForward(
        nn, &alpha, input_desc, input, filter_desc, filter, convolution_desc,
        algorithm, workspace, workspace_size, &beta, output_desc, output));

  // Benchmark
  Timer t;
  cudaDeviceSynchronize();
  timer_reset(&t);
  for (size_t i = 0; i < 1; ++i)
    cudnnConvolutionForward(nn, &alpha, input_desc, input, filter_desc, filter,
                            convolution_desc, algorithm, workspace,
                            workspace_size, &beta, output_desc, output);
  cudaDeviceSynchronize();
  timer_diff_print(t);

  // Check results ------------------------------
  real* output_host = (real*)malloc(output_bytes);
  cudaMemcpy(output_host, output, output_bytes, cudaMemcpyDeviceToHost);
  for (size_t j = (gh - 1) / 2; j < fh_out - (gh - 1) / 2; ++j) {
    for (size_t i = (gw - 1) / 2; i < fw_out - (gw - 1) / 2; ++i) {
      assert(output_host[i + j * fw_out] == 9);
      // printf("%lu: %g\n", i, output_host[i + j * fw_out]);
    }
  }

  // Free ---------------------------------------
  // Workspace
  cudaFree(workspace);

  // Output
  cudaFree(output);
  ERRCHK(cudnnDestroyTensorDescriptor(output_desc));

  // Convolution
  ERRCHK(cudnnDestroyConvolutionDescriptor(convolution_desc));

  // Filter
  cudaFree(filter);
  ERRCHK(cudnnDestroyFilterDescriptor(filter_desc));

  // Input
  cudaFree(input);
  ERRCHK(cudnnDestroyTensorDescriptor(input_desc));

  // cuDNN
  cudnnDestroy(nn);
  return EXIT_SUCCESS;
}