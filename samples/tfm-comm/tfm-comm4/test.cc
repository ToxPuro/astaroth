#include "ntuple.h"
#include "type_conversion.h"

#include "print.h"

#include <algorithm>
#include <array>

#include "static_array.h"

int
main(void)
{
    test_type_conversion();

    // std::array<size_t, ndims> arr = {1, 2, 3};
    // PRINTD(arr.size());

    // const auto arr2 = arr;
    // arr[0]++;

    // for (const auto& e : arr2)
    //     std::cout << e << " ";
    // std::cout << std::endl;

    const size_t ndims = 3;

    StaticArray<size_t, ndims> nn = {128, 128, 128};
    StaticArray<size_t, ndims> rr = {1, 1, 1};
    StaticArray<size_t, 2> a      = {1, 2};
    StaticArray<size_t, 2> b      = {3, 4};
    PRINTD(as<size_t>(2) * rr + nn);
    PRINTD(prod(nn));
    PRINTD(a.dot(b));

    PRINTD(a);
    PRINTD(b);
    a = b;
    PRINTD(a);
    PRINTD(b);
    ++b[0];
    PRINTD(a);
    PRINTD(b);

    return EXIT_SUCCESS;
}
