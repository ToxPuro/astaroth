#include "buffer.h"
#include "decomp.h"
#include "math_utils.h"
#include "ndarray.h"
#include "partition.h"
#include "static_array.h"
#include "type_conversion.h"

#include <stdlib.h>

#include <cxxabi.h>

#include "buffer_draft.h"

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

    auto a = host::Buffer<double>(10);
    auto b = cuda::pinned::Buffer<double>(10);
    auto c = cuda::pinned::wc::Buffer<double>(10);
    auto d = cuda::device::Buffer<double>(10);
    print_demangled(a);
    print_demangled(b);
    print_demangled(c);
    print_demangled(d);
    migrate(a, b);
    migrate(a, c);
    migrate(a, d);
    migrate(b, b);
    migrate(b, c);
    migrate(b, d);
    migrate(c, b);
    migrate(c, c);
    migrate(c, d);
    migrate(d, b);
    migrate(d, c);
    migrate(d, d);

    return EXIT_SUCCESS;
}
