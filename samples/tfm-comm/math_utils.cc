#include "math_utils.h"

#include "errchk.h"

template <size_t N>
auto
to_linear(const ac::array<uint64_t, N>& coords, const ac::array<uint64_t, N>& shape)
{
    uint64_t result = 0;
    for (size_t j = 0; j < shape.size(); ++j) {
        uint64_t factor = 1;
        for (size_t i = 0; i < j; ++i)
            factor *= shape[i];
        result += coords[j] * factor;
    }
    return result;
}

static void
test_to_linear(void)
{
    {
        const ac::array<uint64_t, 3> coords{0, 0, 0};
        const ac::array<uint64_t, 3> shape{1, 1, 1};

        ERRCHK(to_linear(coords, shape) == 0);
    }
    {
        const ac::array<uint64_t, 2> coords{1, 0};
        const ac::array<uint64_t, 2> shape{32, 32};

        ERRCHK(to_linear(coords, shape) == 1);
    }
    {
        const ac::array<uint64_t, 2> coords{31, 0};
        const ac::array<uint64_t, 2> shape{32, 32};

        ERRCHK(to_linear(coords, shape) == 31);
    }
    {
        const ac::array<uint64_t, 2> coords{0, 31};
        const ac::array<uint64_t, 2> shape{32, 32};

        ERRCHK(to_linear(coords, shape) == 31 * 32);
    }
    {
        const ac::array<uint64_t, 4> coords{1, 2, 3, 4};
        const ac::array<uint64_t, 4> shape{10, 9, 8, 7};

        ERRCHK(to_linear(coords, shape) == 1 + 2 * 10 + 3 * 10 * 9 + 4 * 10 * 9 * 8);
    }
}

template <size_t N>
auto
to_spatial(const uint64_t index, const ac::array<uint64_t, N>& shape)
{
    ac::array<uint64_t, N> coords;
    for (size_t j = 0; j < shape.size(); ++j) {
        uint64_t divisor = 1;
        for (size_t i = 0; i < j; ++i)
            divisor *= shape[i];
        coords[j] = (index / divisor) % shape[j];
    }
    return coords;
}

static void
test_to_spatial(void)
{
    {
        const uint64_t i{0};
        const Shape shape{4, 5, 6};
        const Shape coords{0, 0, 0};
        ERRCHK(to_spatial(i, shape) == coords);
    }
    {
        const uint64_t i{4};
        const Shape shape{4, 5, 6};
        const Shape coords{0, 1, 0};
        ERRCHK(to_spatial(i, shape) == coords);
    }
    {
        const uint64_t i{4 * 5};
        const Shape shape{4, 5, 6};
        const Shape coords{0, 0, 1};
        ERRCHK(to_spatial(i, shape) == coords);
    }
    {
        const uint64_t i{4 * 5 * 6 - 1};
        const Shape shape{4, 5, 6};
        const Shape coords{3, 4, 5};
        ERRCHK(to_spatial(i, shape) == coords);
    }
}

uint64_t
prod(const size_t count, const uint64_t* arr)
{
    uint64_t res = 1;
    for (size_t i = 0; i < count; ++i)
        res *= arr[i];
    return res;
}

static void
test_within_box(void)
{
    {
        const Index box_offset{0, 0, 0};
        const Shape box_dims{10, 10, 10};
        const Index coords{0, 0, 0};
        ERRCHK(within_box(coords, box_dims, box_offset) == true);
    }
    {
        const Index box_offset{0, 0, 0};
        const Shape box_dims{10, 10, 10};
        const Index coords{0, 10, 0};
        ERRCHK(within_box(coords, box_dims, box_offset) == false);
    }
    {
        const Index box_offset{0, 0, 0};
        const Shape box_dims{10, 10, 10};
        const Index coords{11, 11, 11};
        ERRCHK(within_box(coords, box_dims, box_offset) == false);
    }
    {
        const ac::array<uint64_t, 10> box_offset{0, 0, 0, 0, 0, 0, 0};
        const ac::array<uint64_t, 10> box_dims{1, 2, 3, 4, 5, 6, 7};
        const ac::array<uint64_t, 10> coords{0, 1, 2, 3, 4, 5, 6};
        ERRCHK(within_box(coords, box_dims, box_offset) == true);
    }
}

void
test_math_utils(void)
{
    test_to_linear();
    test_to_spatial();
    test_within_box();
}
