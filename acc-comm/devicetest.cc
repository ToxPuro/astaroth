#include "devicetest.h"

#include <iostream>

template <typename T>
void
call_device(const cudaStream_t stream, const size_t count, const T* in, T* out)
{
    std::cerr << "not implemented" << std::endl;
}

template void call_device<double>(const cudaStream_t, const size_t, const double*, double*);
