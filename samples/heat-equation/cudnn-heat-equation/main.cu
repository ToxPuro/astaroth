#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <cudnn.h>

#include "timer_hires.h"

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

  cudnnDataType_t dtype      = CUDNN_DATA_FLOAT;
  cudnnTensorFormat_t format = CUDNN_TENSOR_NHWC;

  // Input
  const size_t fn = 1;
  const size_t fc = 1;
  const size_t fh = 4096;
  const size_t fw = 4096;
  cudnnTensorDescriptor_t input_desc;
  cudnnCreateTensorDescriptor(&input_desc);
  cudnnSetTensor4dDescriptor(input_desc, format, dtype, fn, fc, fh, fw);

  float* input;
  cudaMalloc((void**)&input, fn * fc * fh * fw * sizeof(input[0]));

  // Kernel
  const size_t gk = 1;
  const size_t gc = 1;
  const size_t gh = 3;
  const size_t gw = 3;
  cudnnFilterDescriptor_t filter_desc;
  cudnnCreateFilterDescriptor(&filter_desc);
  cudnnSetFilter4dDescriptor(filter_desc, dtype, format, gk, gc, gh, gw);

  float* filter;
  cudaMalloc((void**)&filter, gk * gc * gh * gw * sizeof(filter[0]));

  // Convolution
  const size_t pad_h = 1;
  const size_t pad_w = 1;
  const size_t str_h = 1;
  const size_t str_w = 1;
  const size_t dil_h = 1;
  const size_t dil_w = 1;
  cudnnConvolutionDescriptor_t convolution_desc;
  cudnnCreateConvolutionDescriptor(&convolution_desc);
  cudnnSetConvolution2dDescriptor(convolution_desc, pad_h, pad_w, str_h, str_w,
                                  dil_h, dil_w, CUDNN_CONVOLUTION, dtype);

  // Output
  int fn_out;
  int fc_out;
  int fh_out;
  int fw_out;
  cudnnGetConvolution2dForwardOutputDim(convolution_desc, input_desc,
                                        filter_desc, &fn_out, &fc_out, &fh_out,
                                        &fw_out);

  cudnnTensorDescriptor_t output_desc;
  cudnnCreateTensorDescriptor(&output_desc);
  cudnnSetTensor4dDescriptor(output_desc, format, dtype, fn_out, fc_out, fh_out,
                             fw_out);
  float* output;
  cudaMalloc((void**)&output,
             fn_out * fc_out * fh_out * fw_out * sizeof(output[0]));

  // Algorithm
  const cudnnConvolutionFwdAlgo_t
      algorithm = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
  // const cudnnConvolutionFwdAlgo_t algorithm =
  // CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM; const
  // cudnnConvolutionFwdAlgo_t algorithm = CUDNN_CONVOLUTION_FWD_ALGO_FFT; const
  // cudnnConvolutionFwdAlgo_t algorithm = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;

  // Workspace
  size_t workspace_size;
  cudnnGetConvolutionForwardWorkspaceSize(nn, input_desc, filter_desc,
                                          convolution_desc, output_desc,
                                          algorithm, &workspace_size);

  float* workspace;
  cudaMalloc((void**)&workspace, workspace_size);

  // Compute ---------------------------------------
  const float alpha = 1.0f;
  const float beta  = 0.0f;

  // Warmup
  for (size_t i = 0; i < 10; ++i)
    cudnnConvolutionForward(nn, &alpha, input_desc, input, filter_desc, filter,
                            convolution_desc, algorithm, workspace,
                            workspace_size, &beta, output_desc, output);

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

  // Free ---------------------------------------
  // Workspace
  cudaFree(workspace);

  // Output
  cudaFree(output);
  cudnnDestroyTensorDescriptor(output_desc);

  // Convolution
  cudnnDestroyConvolutionDescriptor(convolution_desc);

  // Filter
  cudaFree(filter);
  cudnnDestroyFilterDescriptor(filter_desc);

  // Input
  cudaFree(input);
  cudnnDestroyTensorDescriptor(input_desc);

  // cuDNN
  cudnnDestroy(nn);
  return EXIT_SUCCESS;
}