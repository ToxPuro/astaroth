#include <cudnn.h>

static Array input;  // TODO
static Array output; // TODO

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
    fprintf(stderr, "backendInit not implemented\n");
    return;
}

void
backendConvolutionFwd(void)
{
    fprintf(stderr, "backendConvolutionFwd not implemented\n");
    return;
}

void
backendQuit(void)
{
    fprintf(stderr, "backendQuit not implemented\n");
    return;
}