#pragma once

#include "pointer.h"

#include "errchk.h"
#include "print_debug.h"

#include <numeric>
#include <vector>

namespace ac {

class ntuple {
  private:
    std::vector<uint64_t> resource;

  public:
    ntuple(const size_t order, const uint64_t fill_value = 0)
        : resource(order, fill_value)
    {
    }

    ntuple(const std::vector<uint64_t>& vec)
        : resource{vec}
    {
    }

    auto order() const { return resource.size(); }
    auto begin() { return resource.begin(); }
    auto begin() const { return resource.begin(); }
    auto end() { return resource.end(); }
    auto end() const { return resource.end(); }
    auto data() { return resource.data(); }
    auto data() const { return resource.data(); }

    auto& operator[](const size_t i)
    {
        ERRCHK(i < order());
        return resource[i];
    }
    const auto& operator[](const size_t i) const
    {
        ERRCHK(i < order());
        return resource[i];
    }
    friend std::ostream& operator<<(std::ostream& os, const ac::ntuple& obj)
    {
        os << "{ ";
        for (const auto& elem : obj)
            os << elem << " ";
        os << "}";
        return os;
    }

    auto back() const { return resource[order() - 1]; }
};

auto
operator+(const ac::ntuple& a, const ac::ntuple& b)
{
    ERRCHK(a.order() == b.order());

    ac::ntuple c{a.order()};
    for (size_t i{0}; i < a.order(); ++i)
        c[i] = a[i] + b[i];
    return c;
}

bool
operator<(const ac::ntuple& a, const ac::ntuple& b)
{
    ERRCHK(a.order() == b.order());

    for (size_t i{0}; i < a.order(); ++i)
        if (a[i] >= b[i])
            return false;
    return true;
}

static auto
slice(const ac::ntuple& ntuple, const size_t lb, const size_t ub)
{
    ac::ntuple out{ub - lb};
    for (size_t i{lb}; i < ub; ++i)
        out[i - lb] = ntuple[i];
    return out;
}

static auto
prod(const ac::ntuple& in)
{
    uint64_t out{1};
    for (size_t i{0}; i < in.order(); ++i)
        out *= in[i];
    return out;
}

static auto
to_linear(const ac::ntuple& coords, const ac::ntuple& shape)
{
    uint64_t result{0};
    for (size_t j{0}; j < shape.order(); ++j) {
        uint64_t factor{1};
        for (size_t i{0}; i < j; ++i)
            factor *= shape[i];
        result += coords[j] * factor;
    }
    return result;
}

static auto
to_spatial(const uint64_t index, const ac::ntuple& shape)
{
    ac::ntuple coords(shape.order());
    for (size_t j{0}; j < shape.order(); ++j) {
        uint64_t divisor{1};
        for (size_t i{0}; i < j; ++i)
            divisor *= shape[i];
        coords[j] = (index / divisor) % shape[j];
    }
    return coords;
}

template <typename T>
void
transform(const ac::ntuple& dims, const ac::ntuple& subdims, const ac::ntuple& offset, const T* in,
          T* out)
{
    for (uint64_t out_idx{0}; out_idx < prod(subdims); ++out_idx) {
        const ac::ntuple out_coords{to_spatial(out_idx, subdims)};
        ERRCHK(out_coords < subdims);

        const ac::ntuple in_coords{offset + out_coords};
        ERRCHK(in_coords < dims);

        const uint64_t in_idx{to_linear(in_coords, dims)};
        out[out_idx] = in[in_idx];
    }
}

// template <typename T> print_recursive(const size_t depth, const ac::ntuple& ntuple) { ERRCHK() }

template <typename T>
void
print_recursive(const ac::ntuple& shape, const T* data)
{
    if (shape.order() == 1) {
        for (size_t i{0}; i < shape[0]; ++i)
            std::cout << std::setw(4) << data[i];
        std::cout << std::endl;
    }
    else {
        const auto curr_extent{shape[shape.order() - 1]};
        const auto new_shape{slice(shape, 0, shape.order() - 1)};
        const auto offset{prod(new_shape)};
        for (size_t i{0}; i < curr_extent; ++i) {
            if (shape.order() > 4)
                printf("%zu. %zu-dimensional hypercube:\n", i, shape.order() - 1);
            if (shape.order() == 4)
                printf("Cube %zu:\n", i);
            if (shape.order() == 3)
                printf("Layer %zu:\n", i);
            if (shape.order() == 2)
                printf("Row %zu: ", i);
            print_recursive(new_shape, &data[i * offset]);
        }
        printf("\n");
    }
}

template <typename T>
void
print(const std::string& label, const ac::ntuple& shape, const T* data)
{
    std::cout << label << ":" << std::endl;
    print_recursive(shape, data);
}

} // namespace ac

template <typename T> class ttuple {
  private:
  public:
    std::vector<T> m_resource;
    ttuple(const std::initializer_list<T>& init_list)
        : m_resource{init_list}
    {
    }
};

template <typename T>
auto
make_ttuple(const size_t count)
{
    std::vector<T> vec(count);
    return ttuple{vec};
}

#include <iostream>
void
test_transform()
{
    std::cout << "Transform begin" << std::endl;
    const ac::ntuple dims{{3, 3, 3, 3}};
    const ac::ntuple subdims{{1, 2, 1, 1}};
    const ac::ntuple offset{{1, 1, 1, 1}};
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
    PRINT_DEBUG_VECTOR(tt.m_resource);
    PRINT_DEBUG_VECTOR(ttt.m_resource);
    std::vector<int> something(10, 1);
    ttuple tttt{something};
    PRINT_DEBUG_VECTOR(tttt.m_resource);
    // ttuple ttttt(10); // Not allowed
}
