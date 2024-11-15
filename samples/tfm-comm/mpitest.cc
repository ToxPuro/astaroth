#include <cstdlib>
#include <iostream>

#include <mpi.h>

#include "mpi_utils.h"

#include "devicetest.h"

#include "errchk_cuda.h"

#if defined(DEVICE_ENABLED)
namespace device {
template <typename T>
T*
nalloc(const size_t count)
{
    T* ptr{nullptr};
    ERRCHK_CUDA_API(cudaMalloc(&ptr, count * sizeof(T)));
    return ptr;
}
template <typename T>
void
ndealloc(T* ptr)
{
    WARNCHK(ptr);
    WARNCHK_CUDA_API(cudaFree(ptr));
}

template <typename T>
auto
make_unique(const size_t count)
{
    return std::unique_ptr<double, decltype(&device::ndealloc<T>)>{device::nalloc<T>(count),
                                                                   device::ndealloc<T>};
}
} // namespace device

template <typename T>
static void
print(const std::string& label, const size_t count, const T* data)
{
    std::cout << label << ": { ";
    for (size_t i = 0; i < count; ++i)
        std::cout << data[i] << " ";
    std::cout << "}" << std::endl;
}

template <typename T>
void
migrate(const size_t count, const std::unique_ptr<T[]>& in,
        std::unique_ptr<T, decltype(&device::ndealloc<T>)>& out)
{
    ERRCHK_CUDA_API(cudaMemcpy(out.get(), in.get(), count * sizeof(T), cudaMemcpyHostToDevice));
}
template <typename T>
void
migrate(const size_t count, const std::unique_ptr<T, decltype(&device::ndealloc<T>)>& in,
        std::unique_ptr<T[]>& out)
{
    ERRCHK_CUDA_API(cudaMemcpy(out.get(), in.get(), count * sizeof(T), cudaMemcpyDeviceToHost));
}

int
main(void)
{
    ERRCHK_MPI_API(MPI_Init(NULL, NULL));
    try {
        int rank;
        ERRCHK_MPI_API(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
        std::cout << "Hello from " << rank << std::endl;

        const size_t count = 10;
        auto hin{std::make_unique<double[]>(count)};
        auto din{device::make_unique<double>(count)};
        auto dout{device::make_unique<double>(count)};
        auto hout{std::make_unique<double[]>(count)};

        for (size_t i = 0; i < count; ++i)
            hin[i] = static_cast<double>(i);
        print("hin", count, hin.get());

        migrate(count, hin, din);
        BENCHMARK(call_device(0, count, din.get(), dout.get()));
        BENCHMARK(call_device(0, count, din.get(), dout.get()));
        BENCHMARK(call_device(0, count, din.get(), dout.get()));
        BENCHMARK(call_device(0, count, din.get(), dout.get()));
        BENCHMARK(call_device(0, count, din.get(), dout.get()));
        migrate(count, dout, hout);

        print("hout", count, hout.get());

        ERRCHK_MPI_API(MPI_Finalize());
    }
    catch (std::exception& e) {
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    return EXIT_SUCCESS;
}
#else
int
main(void)
{
    std::cerr << "not implemented for host" << std::endl;
    return EXIT_FAILURE;
}
#endif
