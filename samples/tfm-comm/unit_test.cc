#include "buffer.h"
#include "decomp.h"
#include "math_utils.h"
#include "ndarray.h"
#include "partition.h"
#include "static_array.h"
#include "type_conversion.h"

#include <stdlib.h>

#include <cxxabi.h>

// #include "buffer_draft.h"

#include "buf.h"
// #include "mem.h"
#include <memory.h>

template <typename T>
void
print_demangled(const T& obj)
{
    int status;
    std::unique_ptr<char, void (*)(void*)> res{abi::__cxa_demangle(typeid(obj).name(), nullptr,
                                                                   nullptr, &status),
                                               std::free};
    std::cout << "Type: " << (status == 0 ? res.get() : typeid(obj).name()) << std::endl;
}

// using memory_t = std::unique_ptr<void, void (*)(void*)>;

// template <typename T> using mem_t = std::unique_ptr<T, void (*)(void*)>;
// template <typename T, typename Deleter = void (*)(T*)> using memm_t = std::unique_ptr<T,
// Deleter>;

// auto dealloc = [](void* ptr) {
//     std::cout << "freeing" << std::endl;
//     free(ptr);
// };

int
main(void)
{
    test_type_conversion();
    test_static_array();
    test_ndarray();
    test_partition();
    test_decomp();
    test_buffer();
    test_math_utils();

    // Buffaa<double> a(10);
    // Buffaa<double> b(10);
    // a = std::move(b);

    // auto c = device::make_unique<double>(10);
    // auto a = host::make_unique<double>(10);
    // auto b = host::pinned::make_unique<double>(10);
    // auto c = device::make_unique<double>(10);

    // print_demangled(a);
    // print_demangled(b);
    // print_demangled(c);

    // Buffaa<int> buf(10);
    // auto a       = cuda::device::Buffer<double>(10);
    // auto b       = host::Buffer<double>(20);
    // BaseBuffer c = b;
    // migrate(a, b);
    // migrate(b, a);
    // migrate(a, b);
    // migrate(b, b);

    // auto a = host::Buffer<double>(10);
    // auto b = cuda::pinned::Buffer<double>(10);
    // auto c = cuda::pinned::wc::Buffer<double>(10);
    // auto d = cuda::device::Buffer<double>(10);
    // print_demangled(a);
    // print_demangled(b);
    // print_demangled(c);
    // print_demangled(d);
    // migrate(a, b);
    // migrate(a, c);
    // migrate(a, d);
    // migrate(b, b);
    // migrate(b, c);
    // migrate(b, d);
    // migrate(c, b);
    // migrate(c, c);
    // migrate(c, d);
    // migrate(d, b);
    // migrate(d, c);
    // migrate(d, d);

    // memory_t hmem = memory_t(malloc(1), [](void* ptr) {
    //     std::cout << "freeing" << std::endl;
    //     free(ptr);
    // });
    // memory_t dmem = memory_t(malloc(1), dealloc);
    // hmem          = std::move(dmem);

    // auto tmem = mem_t<int>((int*)malloc(sizeof(int)), dealloc);
    // print_demangled(tmem);
    // std::cout << tmem.get()[0] << std::endl;
    // auto tmem_b = mem_t<int>((int*)malloc(sizeof(int)), dealloc);
    // tmem        = std::move(tmem_b);

    // // auto tmem2 = mem_t<double>((double*)malloc(sizeof(double)));
    // // auto tmem2 = std::unique_ptr<double>(new double[10]);
    // auto tmem2 = memm_t<double>(new double[10], [](double* ptr) { delete ptr; });
    // print_demangled(tmem2);

    // mem_t<double> a = mem_t<double>((double*)nalloc_host<double>(10), dealloc_host);
    // auto a = std::unique_ptr<double, void (*)(double*)>(
    //     [](const size_t count) {
    //         std::cout << "alloc" << std::endl;
    //         return (double*)malloc(count * sizeof(double));
    //     },
    //     [](double* ptr) {
    //         std::cout << "free" << std::endl;
    //         free(ptr);
    //     });
    // auto alloc = [](const size_t count) {
    //     std::cout << "alloc" << std::endl;
    //     return (double*)malloc(count * sizeof(double));
    // };
    // auto dealloc = [](double* ptr) {
    //     std::cout << "free" << std::endl;
    //     free(ptr);
    // };
    // auto a = std::unique_ptr<double, decltype(dealloc)>(alloc(10), dealloc);
    // auto a = std::unique_ptr<double, void (*)(double*)>(alloc(10), dealloc);
    // mem_t<double> a = mem_t<double>(nalloc_host<double>(10), dealloc_host<double>);
    // auto a = host::make_unique<double>(10);
    // auto b = device::make_unique<double>(10);
    // auto c = host::pinned::make_unique<double>(10);
    // auto d = host::pinned::wc::make_unique<double>(10);

    // buf copy
    auto a = HostBufferDefault<double>(10);
    auto b = HostBufferPinned<double>(10);
    auto c = HostBufferPinnedWriteCombined<double>(10);
    auto d = DeviceBufferDefault<double>(10);

    migrate(a, a);
    migrate(a, b);
    migrate(a, c);
    migrate(a, d);

    migrate(b, a);
    migrate(b, b);
    migrate(b, c);
    migrate(b, d);

    migrate(c, a);
    migrate(c, b);
    migrate(c, c);
    migrate(c, d);

    migrate(d, a);
    migrate(d, b);
    migrate(d, c);
    migrate(d, d);

    // HostBuffer<double> e = HostBufferPinnedWriteCombined<double>(10);
    // migrate(e, c);

    // types
    // regular htoh can also be done with cuda if available
    //

    // auto a = GenericBuffer<double, host::make_unique<double>>(10);

    // auto a = HostBufferDefault<double>(10);
    // auto b = DeviceBufferDefault<double>(10);
    // fill(a);
    // auto a = HostBufferDefault<double>(10);
    // auto b = HostBufferPinned<double>(10);
    // fill(a);
    // print(a);

    return EXIT_SUCCESS;
}
