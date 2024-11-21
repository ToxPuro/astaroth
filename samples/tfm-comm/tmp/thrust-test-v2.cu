#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>

#include <numeric> // iota

#if defined(DEVICE_ENABLED)
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cuda/std/array>
namespace ac {
template <typename T> using host_vector   = thrust::host_vector<T>;
template <typename T> using device_vector = thrust::device_vector<T>;
template <typename T, size_t N> using array = cuda::std::array<T, N>;
using thrust::copy;
using thrust::multiplies;
using thrust::reduce;
} // namespace ac
#else
#include <vector>
namespace ac {
template <typename T> using host_vector   = std::vector<T>;
template <typename T> using device_vector = std::vector<T>;
template <typename T, size_t N> using array = std::array<T, N>;
using std::copy;
using std::multiplies;
using std::reduce;
} // namespace ac
#endif


template <typename T>
static void
print(const std::string& label, const Buffer<T, HostMemoryResource>& vec)
{
    std::cout << label << ": { ";
    for (const auto& elem : vec)
        std::cout << elem << " ";
    std::cout << "}" << std::endl;
}

int main() {
    constexpr size_t count{10};
    Buffer<int, HostMemoryResource> hin(count);
    Buffer<int, HostMemoryResource> hout(count);

    Buffer<int, DeviceMemoryResource> din(count);
    Buffer<int, DeviceMemoryResource> dout(count);

    std::cout << "Complete" << std::endl;
    return EXIT_SUCCESS;
}
