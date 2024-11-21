#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>

#include <numeric> // iota

#if defined(DEVICE_ENABLED)
#include <cuda/std/array>
#include <thrust/device_buffer.h>
#include <thrust/host_buffer.h>
namespace ac {
template <typename T> using host_vector     = thrust::host_vector<T>;
template <typename T> using device_vector   = thrust::device_vector<T>;
template <typename T, size_t N> using array = cuda::std::array<T, N>;
using thrust::copy;
using thrust::multiplies;
using thrust::reduce;
} // namespace ac
#else
#include <vector>
namespace ac {
template <typename T> using host_vector     = std::vector<T>;
template <typename T> using device_vector   = std::vector<T>;
template <typename T, size_t N> using array = std::array<T, N>;
using std::copy;
using std::multiplies;
using std::reduce;
} // namespace ac
#endif

template <typename T>
static void
print(const std::string& label, const Buffer<T, ac::mr::host_memory_resource>& vec)
{
    std::cout << label << ": { ";
    for (const auto& elem : vec)
        std::cout << elem << " ";
    std::cout << "}" << std::endl;
}

int
main()
{
    constexpr size_t count{10};
    Buffer<int, ac::mr::host_memory_resource> hin(count);
    Buffer<int, ac::mr::host_memory_resource> hout(count);

    Buffer<int, ac::mr::device_memory_resource> din(count);
    Buffer<int, ac::mr::device_memory_resource> dout(count);

    std::cout << "Complete" << std::endl;
    return EXIT_SUCCESS;
}
