#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include <hip/hip_runtime.h>
#include <miopen/miopen.h>

#include "hip.h"

#define cudaMallocManaged hipMallocManaged

#define cudnnHandle_t miopenHandle_t
#define cudnnCreate miopenCreate
#define cudnnDataType_t miopenDataType_t

#define cudnnTensorFormat_t miopenTensorFormat_t
#define cudnnTensorDescriptor_t miopenTensorDescriptor_t
#define cudnnCreateTensorDescriptor miopenCreateTensorDescriptor
#define cudnnSetTensor4dDescriptor miopenSet4dTensorDescriptor

#define cudnnFilterDescriptor_t miopenTensorDescriptor_t
#define cudnnCreateFilterDescriptor miopenCreateTensorDescriptor
#define cudnnSetFilter4dDescriptor miopenSet4dTensorDescriptor

#define cudnnConvolutionDescriptor_t miopenConvolutionDescriptor_t
#define cudnnCreateConvolutionDescriptor miopenCreateConvolutionDescriptor
#define CUDNN_CONVOLUTION miopenConvolution
#define cudnnSetConvolution2dDescriptor miopenInitConvolutionDescriptor
#define cudnnGetConvolution2dForwardOutputDim                                  \
  miopenGetConvolutionForwardOutputDim

#define cudnnGetConvolutionForwardWorkspaceSize                                \
  miopenConvolutionForwardGetWorkSpaceSize

#define cudnnConvolutionForward miopenConvolutionForward

// #define CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM                               \
//   miopenConvolutionFwdAlgoImplicitGEMM
// miopenConvolutionFwdAlgoDirect

#define cudnnDestroyConvolutionDescriptor miopenDestroyConvolutionDescriptor
#define cudnnDestroy miopenDestroy
#define cudnnDestroyTensorDescriptor miopenDestroyTensorDescriptor

#define cudnnConvolutionFwdAlgo_t miopenConvFwdAlgorithm_t
#define cudnnDestroyFilterDescriptor miopenDestroyTensorDescriptor

#define CUDNN_DATA_FLOAT miopenFloat
#define CUDNN_TENSOR_NHWC miopenTensor

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

  cudnnDataType_t dtype = CUDNN_DATA_FLOAT;
  // cudnnTensorFormat_t format = CUDNN_TENSOR_NHWC; // NHWC not supported

  // Input
  const size_t fn = 1;
  const size_t fc = 1;
  const size_t fh = 4096;
  const size_t fw = 4096;
  cudnnTensorDescriptor_t input_desc;
  cudnnCreateTensorDescriptor(&input_desc);
  // cudnnSetTensor4dDescriptor(input_desc, format, dtype, fn, fc, fh, fw);
  cudnnSetTensor4dDescriptor(input_desc, dtype, fn, fc, fh, fw);

  float* input;
  cudaMallocManaged((void**)&input, fn * fc * fh * fw * sizeof(input[0]));

  // // Kernel
  const size_t gk = 1;
  const size_t gc = 1;
  const size_t gh = 3;
  const size_t gw = 3;
  cudnnFilterDescriptor_t filter_desc;
  cudnnCreateFilterDescriptor(&filter_desc);
  cudnnSetFilter4dDescriptor(filter_desc, dtype, gk, gc, gh, gw);

  float* filter;
  cudaMallocManaged((void**)&filter, gk * gc * gh * gw * sizeof(filter[0]));

  // Convolution
  const size_t pad_h = 1;
  const size_t pad_w = 1;
  const size_t str_h = 1;
  const size_t str_w = 1;
  const size_t dil_h = 1;
  const size_t dil_w = 1;
  cudnnConvolutionDescriptor_t convolution_desc;
  cudnnCreateConvolutionDescriptor(&convolution_desc);
  // cudnnSetConvolution2dDescriptor(convolution_desc, pad_h, pad_w, str_h,
  // str_w,
  //                                 dil_h, dil_w, CUDNN_CONVOLUTION, dtype);
  cudnnSetConvolution2dDescriptor(convolution_desc, CUDNN_CONVOLUTION, pad_h,
                                  pad_w, str_h, str_w, dil_h, dil_w);

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
  cudnnSetTensor4dDescriptor(output_desc, dtype, fn_out, fc_out, fh_out,
                             fw_out);
  float* output;
  cudaMallocManaged((void**)&output,
                    fn_out * fc_out * fh_out * fw_out * sizeof(output[0]));

  // Algorithm
  // const cudnnConvolutionFwdAlgo_t
  //     algorithm = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
  // const cudnnConvolutionFwdAlgo_t algorithm =
  // CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM; const
  // cudnnConvolutionFwdAlgo_t algorithm = CUDNN_CONVOLUTION_FWD_ALGO_FFT;
  // const cudnnConvolutionFwdAlgo_t algorithm =
  // CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;

  // Workspace
  size_t workspace_size;
  // cudnnGetConvolutionForwardWorkspaceSize(nn, input_desc, filter_desc,
  //                                         convolution_desc, output_desc,
  //                                         algorithm, &workspace_size);
  cudnnGetConvolutionForwardWorkspaceSize(nn, input_desc, filter_desc,
                                          convolution_desc, output_desc,
                                          &workspace_size);

  float* workspace;
  cudaMallocManaged((void**)&workspace, workspace_size);

  // // FindConvolution() is mandatory.
  // // Allocate workspace prior to running this API.
  // // A table with times and memory requirements
  // // for different algorithms is returned.
  // // Users can choose the top-most algorithm if
  // // they only care about the fastest algorithm.
  // miopenStatus_t
  // miopenFindConvolutionForwardAlgorithm(
  //     miopenHandle_t handle,
  //     const miopenTensorDescriptor_t xDesc,
  //     const void *x,
  //     const miopenTensorDescriptor_t wDesc,
  //     const void *w,
  //     const miopenConvolutionDescriptor_t convDesc,
  //     const miopenTensorDescriptor_t yDesc,
  //     void *y,
  //     const int requestAlgoCount,
  //     int *returnedAlgoCount,
  //     miopenConvAlgoPerf_t *perfResults,
  //     void *workSpace,
  //     size_t workSpaceSize,
  //     bool exhaustiveSearch)
  const int required_algorithms = 1;
  miopenConvAlgoPerf_t algorithms[required_algorithms];
  int returned_algorithms;
  miopenFindConvolutionForwardAlgorithm(
      nn, input_desc, input, filter_desc, filter, convolution_desc, output_desc,
      output, required_algorithms, &returned_algorithms, algorithms, workspace,
      workspace_size, true);
  assert(returned_algorithms == required_algorithms);

  printf("Convolution algorithm selected: ");
  switch (algorithms[0].fwd_algo) {
  case 0:
    printf("miOpenConvolutionAlgoGEMM\n");
    break;
  case 1:
    printf("miopenConvolutionAlgoDirect\n");
    break;
  case 2:
    printf("miopenConvolutionAlgoFFT\n");
    break;
  case 3:
    printf("miopenConvolutionAlgoWinograd\n");
    break;
  case 5: // Yes, skips 4
    printf("miopenConvolutionAlgoImplicitGEMM\n");
    break;
  }

  // Compute ---------------------------------------
  const float alpha = 1.0f;
  const float beta  = 0.0f;

  // Warmup
  for (size_t i = 0; i < 10; ++i)
    cudnnConvolutionForward(nn, &alpha, input_desc, input, filter_desc, filter,
                            convolution_desc, algorithms[0].fwd_algo, &beta,
                            output_desc, output, workspace, workspace_size);

  // cudnnConvolutionForward(nn, &alpha, input_desc, input, filter_desc, filter,
  //                         convolution_desc, workspace, workspace_size, &beta,
  //                         output_desc, output);

  /*
  miopenStatus_t
  miopenConvolutionForward(
      miopenHandle_t handle,
      const void *alpha,
      const miopenTensorDescriptor_t xDesc,
      const void *x,
      const miopenTensorDescriptor_t wDesc,
      const void *w,
      const miopenConvolutionDescriptor_t convDesc,
      miopenConvFwdAlgorithm_t algo,
      const void *beta,
      const miopenTensorDescriptor_t yDesc,´`
      void *y,
      void *workSpace,
      size_t workSpaceSize)
  */

  // Benchmark
  Timer t;
  cudaDeviceSynchronize();
  timer_reset(&t);
  for (size_t i = 0; i < 1; ++i)
    cudnnConvolutionForward(nn, &alpha, input_desc, input, filter_desc, filter,
                            convolution_desc, algorithms[0].fwd_algo, &beta,
                            output_desc, output, workspace, workspace_size);

  // cudnnConvolutionForward(nn, &alpha, input_desc, input, filter_desc, filter,
  //                         convolution_desc, workspace, workspace_size, &beta,
  //                         output_desc, output);

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