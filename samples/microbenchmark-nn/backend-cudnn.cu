#include <cudnn.h>

#include "array.h"

#if DOUBLE_PRECISION // defined in array.h
static const cudnnDataType_t dtype = CUDNN_DATA_DOUBLE;
#else
static const cudnnDataType_t dtype = CUDNN_DATA_FLOAT;
#endif

static cudnnHandle_t nn = NULL;
static cudnnTensorDescriptor_t input_desc;
static Array input;

static cudnnFilterDescriptor_t filter_desc;
static Array filter;

static cudnnTensorDescriptor_t output_desc;
static Array output;

static cudnnConvolutionDescriptor_t convolution_desc;
static Array workspace;

static const int required_algorithms = 1;
static cudnnConvolutionFwdAlgoPerf_t algorithms[required_algorithms];
static cudnnConvolutionFwdAlgo_t algorithm;

static cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW;
static auto convolution_type      = CUDNN_CROSS_CORRELATION;

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

#define CUDNN_ERRCHK(params)                                                                       \
    {                                                                                              \
        cudnn_errchk((params), __FILE__, __LINE__, true);                                          \
    }

Array
backendGetInputTensor(void)
{
    return input;
}

Array
backendGetOutputTensor(void)
{
    return output;
}

void
backendInit(const size_t domain_length, const size_t radius, const size_t stride)
{
    // It seems that f64 is not supported with all algorithms
    // and autotuning may not take this into account (f64 verification fails)
    ERRCHK_ALWAYS(sizeof(real) == sizeof(float));

    // cuDNN
    cudnnCreate(&nn);

    // Input
    const size_t fn = 1;
    const size_t fc = 1;
    const size_t fh = 1;
    const size_t fw = domain_length;
    CUDNN_ERRCHK(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_ERRCHK(cudnnSetTensor4dDescriptor(input_desc, format, dtype, fn, fc, fh, fw));
    input = arrayCreate(fn * fc * fh * fw, true);

    // Kernel
    const size_t gk = 1;
    const size_t gc = 1;
    const size_t gh = 1;
    const size_t gw = 2 * radius + 1;
    CUDNN_ERRCHK(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_ERRCHK(cudnnSetFilter4dDescriptor(filter_desc, dtype, format, gk, gc, gh, gw));
    filter = arrayCreate(gk * gc * gh * gw, true);

    Array host_filter = arrayCreate(filter.length, false);
    for (size_t i = 0; i < filter.length; ++i)
        host_filter.data[i] = 1;
    cudaMemcpy(filter.data, host_filter.data, filter.bytes, cudaMemcpyHostToDevice);
    arrayDestroy(&host_filter);

    // Convolution
    const size_t pad_h = (gh - 1) / 2;
    const size_t pad_w = (gw - 1) / 2;
    const size_t str_h = 1;
    const size_t str_w = 1;
    const size_t dil_h = 1;
    const size_t dil_w = 1;
    CUDNN_ERRCHK(cudnnCreateConvolutionDescriptor(&convolution_desc));
    CUDNN_ERRCHK(cudnnSetConvolution2dDescriptor(convolution_desc, pad_h, pad_w, str_h, str_w,
                                                 dil_h, dil_w, convolution_type, dtype));

    // Output
    int fn_out;
    int fc_out;
    int fh_out;
    int fw_out;
    CUDNN_ERRCHK(cudnnGetConvolution2dForwardOutputDim(convolution_desc, input_desc, filter_desc,
                                                       &fn_out, &fc_out, &fh_out, &fw_out));

    CUDNN_ERRCHK(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_ERRCHK(
        cudnnSetTensor4dDescriptor(output_desc, format, dtype, fn_out, fc_out, fh_out, fw_out));
    output = arrayCreate(fn_out * fc_out * fh_out * fw_out, true);

    // Algorithm
    int returned_algorithms;
    CUDNN_ERRCHK(cudnnFindConvolutionForwardAlgorithm(nn, input_desc, filter_desc, convolution_desc,
                                                      output_desc, required_algorithms,
                                                      &returned_algorithms, algorithms));
    ERRCHK_ALWAYS(returned_algorithms == required_algorithms);
    algorithm = algorithms[0].algo;

    // Workspace
    size_t workspace_bytes;
    CUDNN_ERRCHK(cudnnGetConvolutionForwardWorkspaceSize(nn, input_desc, filter_desc,
                                                         convolution_desc, output_desc, algorithm,
                                                         &workspace_bytes));
    workspace = arrayCreate(workspace_bytes / sizeof(output.data[0]), true);
    ERRCHK_ALWAYS(workspace.bytes == workspace_bytes);
}

void
backendConvolutionFwd(void)
{
    const real alpha = 1;
    const real beta  = 0;
    cudnnConvolutionForward(nn, &alpha,                      //
                            input_desc, input.data,          //
                            filter_desc, filter.data,        //
                            convolution_desc, algorithm,     //
                            workspace.data, workspace.bytes, //
                            &beta, output_desc, output.data);
}

void
backendQuit(void)
{
    cudaDeviceSynchronize();

    // Workspace
    arrayDestroy(&workspace);

    // Output
    arrayDestroy(&output);
    cudnnDestroyTensorDescriptor(output_desc);

    // Convolution
    arrayDestroy(&workspace);
    cudnnDestroyConvolutionDescriptor(convolution_desc);

    // Filter
    arrayDestroy(&filter);
    cudnnDestroyFilterDescriptor(filter_desc);

    // Input
    arrayDestroy(&input);
    cudnnDestroyTensorDescriptor(input_desc);

    // cuDNN
    cudnnDestroy(nn);
}