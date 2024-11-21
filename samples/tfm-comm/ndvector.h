#pragma once
#include <iomanip>

#include "array.h"
#include "vector.h"

#include "math_utils.h"

template <typename T>
static void
ndvector_print_recursive(const size_t ndims, const uint64_t* dims, const T* array)
{
    if (ndims == 1) {
        for (size_t i{0}; i < dims[0]; ++i)
            std::cout << std::setw(4) << array[i];
        std::cout << std::endl;
    }
    else {
        const uint64_t offset{prod(ndims - 1, dims)};
        for (size_t i{0}; i < dims[ndims - 1]; ++i) {
            if (ndims > 4)
                printf("%zu. %zu-dimensional hypercube:\n", i, ndims - 1);
            if (ndims == 4)
                printf("Cube %zu:\n", i);
            if (ndims == 3)
                printf("Layer %zu:\n", i);
            if (ndims == 2)
                printf("Row %zu: ", i);
            ndvector_print_recursive<T>(ndims - 1, dims, &array[i * offset]);
        }
        printf("\n");
    }
}

template <typename T>
void
ndvector_print(const char* label, const size_t ndims, const uint64_t* dims, const T* array)
{
    ERRCHK(array != NULL);
    printf("%s:\n", label);
    ndvector_print_recursive<T>(ndims, dims, array);
}

namespace ac {
template <typename T, size_t N, typename MemoryResource> class ndvector {
  private:
    ac::shape<N> shape{};
    ac::vector<T, MemoryResource> resource{};

  public:
    explicit ndvector(const ac::shape<N>& in_shape)
        : shape{in_shape}, resource(prod(in_shape))
    {
    }

    explicit ndvector(const ac::shape<N>& in_shape, const T& fill_value)
        : shape{in_shape}, resource(prod(in_shape), fill_value)
    {
    }

    ac::vector<T, MemoryResource>& vector() { return resource; }
    const ac::vector<T, MemoryResource>& vector() const { return resource; }
    T* data() { return resource.data(); }
    const T* data() const { return resource.data(); }
    size_t size() const { return resource.size(); }
    auto dims() const { return shape; }

    T* begin() { return data(); }
    const T* begin() const { return data(); }
    T* end() { return data() + size(); }
    const T* end() const { return data() + size(); }

    void display() { ndvector_print_recursive(shape.size(), shape.data(), resource.data()); }
};
} // namespace ac

template <typename T>
void
ndvector_fill(const T& value, const size_t ndims, const uint64_t* dims, const uint64_t* subdims,
              const uint64_t* start, T* arr)
{
    if (ndims == 0) {
        *arr = value;
    }
    else {
        ERRCHK(start[ndims - 1] + subdims[ndims - 1] <= dims[ndims - 1]); // OOB
        ERRCHK(dims[ndims - 1] > 0);                                      // Invalid dims
        ERRCHK(subdims[ndims - 1] > 0);                                   // Invalid subdims

        const uint64_t offset{prod(ndims - 1, dims)};
        for (size_t i{start[ndims - 1]}; i < start[ndims - 1] + subdims[ndims - 1]; ++i)
            ndvector_fill<T>(value, ndims - 1, dims, subdims, start, &arr[i * offset]);
    }
}

template <typename T, size_t N>
void
fill(const T& fill_value, const ac::shape<N>& subdims, const ac::shape<N>& offset,
     ac::ndvector<T, N, HostMemoryResource>& ndvec)
{
    ERRCHK(offset + subdims <= ndvec.dims());
    ndvector_fill<T>(fill_value, ndvec.dims().size(), ndvec.dims().data(), subdims.data(),
                     offset.data(), ndvec.data());
}

void test_ndvector(void);
