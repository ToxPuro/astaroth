#include "ndarray.h"

#include <iostream>

#include "print_debug.h"

uint64_t
to_linear(const Index& coords, const Shape& shape)
{
    uint64_t result = 0;
    for (size_t j = 0; j < shape.count; ++j) {
        uint64_t factor = 1;
        for (size_t i = 0; i < j; ++i)
            factor *= shape[i];
        result += coords[j] * factor;
    }
    return result;
}

Index
to_spatial(const uint64_t index, const Shape& shape)
{
    Index coords(shape.count);
    for (size_t j = 0; j < shape.count; ++j) {
        uint64_t divisor = 1;
        for (size_t i = 0; i < j; ++i)
            divisor *= shape[i];
        coords[j] = (index / divisor) % shape[j];
    }
    return coords;
}

uint64_t
prod(const size_t count, const uint64_t* arr)
{
    uint64_t res = 1;
    for (size_t i = 0; i < count; ++i)
        res *= arr[i];
    return res;
}

void
test_ndarray(void)
{
    // std::cout << "hello" << std::endl;
    // NdArray<double> a(Shape{2, 2, 2});
    // NdArray<double> b(Shape{3, 3, 3}, 50);
    // // b.buffer = ones(20);
    // a = b;
    // Buffer<double> a(10);
    // auto a = ones<double>(10);
    // auto a = zeros<double>(10);
    // auto a = arange<double>(5, 20);
    // Buffer<double> a(10);
    // a.fill(1, dims, subdims, offset);
    // PRINT_DEBUG(a);
    // std::cout << a.buffer << std::endl;

    // NdArray<double> a(Shape{4, 4}, Buffer<double>(10, 50));

    // NdArray<double> mesh(Shape{8});
    // Shape dims(2);
    // mesh.fill(1, 7, 1);
    // std::cout << mesh << std::endl;

    // Shape dims    = {4, 4, 4};
    // Shape subdims = {2, 2, 2};
    // Index offset  = {1, 1, 1};
    // NdArray<double> mesh(dims);
    // mesh.fill(1, subdims, offset);
    // mesh.print();
    // std::cout << mesh << std::endl;
}
