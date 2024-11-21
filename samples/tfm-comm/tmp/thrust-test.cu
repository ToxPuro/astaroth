#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>

#include <numeric> // iota

#if defined(DEVICE_ENABLED)
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
namespace ac {
template <typename T> using host_vector   = thrust::host_vector<T>;
template <typename T> using device_vector = thrust::device_vector<T>;
using thrust::copy;
using thrust::multiplies;
using thrust::reduce;
} // namespace ac
#else
#include <vector>
namespace ac {
template <typename T> using host_vector   = std::vector<T>;
template <typename T> using device_vector = std::vector<T>;
using std::copy;
using std::multiplies;
using std::reduce;
} // namespace ac
#endif

#include "datatypes.h"

template <typename T>
static void
print(const std::string& label, const Buffer<T, HostMemoryResource>& vec)
{
    std::cout << label << ": { ";
    for (const auto& elem : vec)
        std::cout << elem << " ";
    std::cout << "}" << std::endl;
}

template <typename T>
static void
ndarray_print_recursive(const size_t ndims, const uint64_t* dims, const T* array)
{
    if (ndims == 1) {
        for (size_t i{0}; i < dims[0]; ++i)
            std::cout << std::setw(4) << array[i];
        std::cout << std::endl;
    }
    else {
        // const uint64_t offset{prod(ndims - 1, dims)};
        const uint64_t offset = ac::reduce(dims, dims + ndims - 1, static_cast<uint64_t>(1),
                                           ac::multiplies<uint64_t>());
        for (size_t i{0}; i < dims[ndims - 1]; ++i) {
            if (ndims > 4)
                printf("%zu. %zu-dimensional hypercube:\n", i, ndims - 1);
            if (ndims == 4)
                printf("Cube %zu:\n", i);
            if (ndims == 3)
                printf("Layer %zu:\n", i);
            if (ndims == 2)
                printf("Row %zu: ", i);
            ndarray_print_recursive<T>(ndims - 1, dims, &array[i * offset]);
        }
        printf("\n");
    }
}

template <typename T>
__global__ void
pack(const uint64_t blocksize, const uint64_t offset, const T* in, T* out)
{
    const uint64_t i{static_cast<uint64_t>(threadIdx.x) + blockIdx.x * blockDim.x};
    if (i < blocksize)
        out[i] = in[offset + i];
}

int
main()
{
    std::cout << "hello" << std::endl;
    Shape<2> mm{4, 4};
    const size_t count{prod(mm)};

    Buffer<double, HostMemoryResource> hin(count);
    std::iota(hin.begin(), hin.end(), 1);
    print("hin", hin);

    ndarray_print_recursive(mm.size(), mm.data(), hin.data());

    // std::copy_if(hin.begin(), hin.end(), hout.begin(), [](const double& val) { return val > 6;
    // });
    Buffer<double, DeviceMemoryResource> din(count);
    ac::copy(hin.begin(), hin.end(), din.begin());

    Buffer<double, DeviceMemoryResource> dout(count - 2);
    Buffer<double, HostMemoryResource> hout(count - 2);
    pack<<<1, 256>>>(count - 2, 1, thrust::raw_pointer_cast(din.data()),
                     thrust::raw_pointer_cast(dout.data()));
    ac::copy(dout.begin(), dout.end(), hout.begin());
    // ERRCHK_CUDA_API(cudaDeviceSynchronize());
    print("hout", hout);

    return EXIT_SUCCESS;
}
