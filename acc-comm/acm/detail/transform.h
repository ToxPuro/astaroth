#pragma once

#include "pointer.h"

#include "errchk.h"
#include "ntuple.h"
#include "print_debug.h"

#include <numeric>
#include <vector>

namespace ac {

template <typename T>
void
transform(const ac::ntuple<uint64_t>& dims, const ac::ntuple<uint64_t>& subdims,
          const ac::ntuple<uint64_t>& offset, const T* in, T* out)
{
    for (uint64_t out_idx{0}; out_idx < prod(subdims); ++out_idx) {
        const ac::ntuple<uint64_t> out_coords{to_spatial(out_idx, subdims)};
        ERRCHK(out_coords < subdims);

        const ac::ntuple<uint64_t> in_coords{offset + out_coords};
        ERRCHK(in_coords < dims);

        const uint64_t in_idx{to_linear(in_coords, dims)};
        out[out_idx] = in[in_idx];
    }
}

// template <typename T> print_recursive(const size_t depth, const ac::ntuple<uint64_t>& ntuple) {
// ERRCHK() }

template <typename T>
void
print_recursive(const ac::ntuple<uint64_t>& shape, const T* data)
{
    if (shape.size() == 1) {
        for (size_t i{0}; i < shape[0]; ++i)
            std::cout << std::setw(4) << data[i];
        std::cout << std::endl;
    }
    else {
        const auto curr_extent{shape[shape.size() - 1]};
        const auto new_shape{slice(shape, 0, shape.size() - 1)};
        const auto offset{prod(new_shape)};
        for (size_t i{0}; i < curr_extent; ++i) {
            if (shape.size() > 4)
                printf("%zu. %zu-dimensional hypercube:\n", i, shape.size() - 1);
            if (shape.size() == 4)
                printf("Cube %zu:\n", i);
            if (shape.size() == 3)
                printf("Layer %zu:\n", i);
            if (shape.size() == 2)
                printf("Row %zu: ", i);
            print_recursive(new_shape, &data[i * offset]);
        }
        printf("\n");
    }
}

template <typename T>
void
print(const std::string& label, const ac::ntuple<uint64_t>& shape, const T* data)
{
    std::cout << label << ":" << std::endl;
    print_recursive(shape, data);
}

} // namespace ac

template <typename T> class ttuple {
  private:
  public:
    std::vector<T> m_m_resource;
    ttuple(const std::initializer_list<T>& init_list)
        : m_m_resource{init_list}
    {
    }
};

template <typename T>
auto
make_ttuple(const size_t count, const T& fill_value = 0)
{
    return ttuple{std::vector<T>(count, fill_value)};
}

template <typename T = int> class tttuple {
  private:
  public:
    std::vector<T> m_m_resource;
    tttuple(const std::initializer_list<T>& init_list)
        : m_m_resource{init_list}
    {
    }
};

template <typename T = int>
auto
make_tttuple(const size_t count, const int& fill_value = 0)
{
    return ttuple{std::vector<T>(count, fill_value)};
}

#include <iostream>
void
test_transform()
{
    std::cout << "Transform begin" << std::endl;
    const ac::ntuple<uint64_t> dims{3, 3, 3, 3};
    const ac::ntuple<uint64_t> subdims{1, 2, 1, 1};
    const ac::ntuple<uint64_t> offset{1, 1, 1, 1};
    auto in{std::make_unique<int[]>(prod(dims))};
    auto out{std::make_unique<int[]>(prod(subdims))};
    std::iota(in.get(), in.get() + prod(dims), 1);
    ac::transform(dims, subdims, offset, in.get(), out.get());
    ac::print("reference", dims, in.get());
    ac::print("candidate", subdims, out.get());
    std::cout << "Transform end" << std::endl;

    ttuple t{1, 2, 3};
    ttuple tt{make_ttuple<int>(2)};
    ttuple ttt{static_cast<size_t>(10)};
    PRINT_DEBUG_VECTOR(tt.m_m_resource);
    PRINT_DEBUG_VECTOR(ttt.m_m_resource);
    std::vector<int> something(10, 1);
    ttuple tttt{something};
    PRINT_DEBUG_VECTOR(tttt.m_m_resource);
    // ttuple ttttt(10); // Not allowed

    tttuple a{make_tttuple(10)};

    ac::ntuple abc{1, 2, 3};
}
