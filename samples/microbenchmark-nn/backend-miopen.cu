#include "backend.h"

#include <miopen/miopen.h>

#if USE_DOUBLE
static const miopenDataType_t dtype = miopenDouble;
#else
static const miopenDataType_t dtype = miopenFloat;
#endif

static miopenHandle_t nn = NULL;
static miopenTensorDescriptor_t input_desc;
static Array input;

static miopenTensorDescriptor_t filter_desc;
static Array filter;

static miopenTensorDescriptor_t output_desc;
static Array output;

static miopenConvolutionDescriptor_t convolution_desc;
static Array workspace;

static const int required_algorithms = 1;
static miopenConvAlgoPerf_t algorithms[required_algorithms];

static auto convolution_type = miopenConvolution;

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
    ERRCHK_ALWAYS(sizeof(real) == sizeof(float)); // 2023-01-09 f64 not supported with MIOpen
    miopenCreate(&nn);

    // Input
    const size_t fn = 1;
    const size_t fc = 1;
    const size_t fh = 1;
    const size_t fw = domain_length;
    miopenCreateTensorDescriptor(&input_desc);
    miopenSet4dTensorDescriptor(input_desc, dtype, fn, fc, fh, fw);
    input = arrayCreate(fn * fc * fh * fw, true);

    // Kernel
    const size_t gk = 1;
    const size_t gc = 1;
    const size_t gh = 1;
    const size_t gw = 2 * radius + 1;
    miopenCreateTensorDescriptor(&filter_desc);
    miopenSet4dTensorDescriptor(filter_desc, dtype, gk, gc, gh, gw);
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
    miopenCreateConvolutionDescriptor(&convolution_desc);
    miopenInitConvolutionDescriptor(convolution_desc, convolution_type, pad_h, pad_w, str_h, str_w,
                                    dil_h, dil_w);

    // Output
    int fn_out;
    int fc_out;
    int fh_out;
    int fw_out;
    miopenGetConvolutionForwardOutputDim(convolution_desc, input_desc, filter_desc, &fn_out,
                                         &fc_out, &fh_out, &fw_out);

    miopenCreateTensorDescriptor(&output_desc);
    miopenSet4dTensorDescriptor(output_desc, dtype, fn_out, fc_out, fh_out, fw_out);
    output = arrayCreate(fn_out * fc_out * fh_out * fw_out, true);

    // Workspace
    size_t workspace_bytes;
    miopenConvolutionForwardGetWorkSpaceSize(nn, input_desc, filter_desc, convolution_desc,
                                             output_desc, &workspace_bytes);
    workspace = arrayCreate(workspace_bytes / sizeof(output.data[0]), true);
    ERRCHK_ALWAYS(workspace.bytes == workspace_bytes);

    int returned_algorithms;
    miopenFindConvolutionForwardAlgorithm(nn, input_desc, input.data, filter_desc, filter.data,
                                          convolution_desc, output_desc, output.data,
                                          required_algorithms, &returned_algorithms, algorithms,
                                          workspace.data, workspace.bytes, true);
    ERRCHK_ALWAYS(returned_algorithms == required_algorithms);

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
}

void
backendConvolutionFwd(void)
{
    const real alpha = 1;
    const real beta  = 0;
    miopenConvolutionForward(nn, &alpha, input_desc, input.data, filter_desc, filter.data,
                             convolution_desc, algorithms[0].fwd_algo, &beta, output_desc,
                             output.data, workspace.data, workspace.bytes);
}

void
backendQuit(void)
{
    cudaDeviceSynchronize();

    // Workspace
    arrayDestroy(&workspace);

    // Output
    arrayDestroy(&output);
    miopenDestroyTensorDescriptor(output_desc);

    // Convolution
    arrayDestroy(&workspace);
    miopenDestroyConvolutionDescriptor(convolution_desc);

    // Filter
    arrayDestroy(&filter);
    miopenDestroyTensorDescriptor(filter_desc);

    // Input
    arrayDestroy(&input);
    miopenDestroyTensorDescriptor(input_desc);

    // cuDNN
    miopenDestroy(nn);

    nn = NULL;
}