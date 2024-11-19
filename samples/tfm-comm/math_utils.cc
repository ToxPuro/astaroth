#include "math_utils.h"

#include "errchk.h"

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

static void
test_to_spatial(void)
{
    {
        const uint64_t i{0};
        const Shape<3> shape{4, 5, 6};
        const Shape<3> coords{0, 0, 0};
        ERRCHK(to_spatial(i, shape) == coords);
    }
    {
        const uint64_t i{4};
        const Shape<3> shape{4, 5, 6};
        const Shape<3> coords{0, 1, 0};
        ERRCHK(to_spatial(i, shape) == coords);
    }
    {
        const uint64_t i{4 * 5};
        const Shape<3> shape{4, 5, 6};
        const Shape<3> coords{0, 0, 1};
        ERRCHK(to_spatial(i, shape) == coords);
    }
    {
        const uint64_t i{4 * 5 * 6 - 1};
        const Shape<3> shape{4, 5, 6};
        const Shape<3> coords{3, 4, 5};
        ERRCHK(to_spatial(i, shape) == coords);
    }
}

uint64_t
prod(const size_t count, const uint64_t* arr)
{
    uint64_t res{1};
    for (size_t i{0}; i < count; ++i)
        res *= arr[i];
    return res;
}

static void
test_within_box(void)
{
    {
        const Index<3> box_offset{0, 0, 0};
        const Shape<3> box_dims{10, 10, 10};
        const Index<3> coords{0, 0, 0};
        ERRCHK(within_box(coords, box_dims, box_offset) == true);
    }
    {
        const Index<3> box_offset{0, 0, 0};
        const Shape<3> box_dims{10, 10, 10};
        const Index<3> coords{0, 10, 0};
        ERRCHK(within_box(coords, box_dims, box_offset) == false);
    }
    {
        const Index<3> box_offset{0, 0, 0};
        const Shape<3> box_dims{10, 10, 10};
        const Index<3> coords{11, 11, 11};
        ERRCHK(within_box(coords, box_dims, box_offset) == false);
    }
    {
        const ac::array<uint64_t, 7> box_offset{0, 0, 0, 0, 0, 0, 0};
        const ac::array<uint64_t, 7> box_dims{1, 2, 3, 4, 5, 6, 7};
        const ac::array<uint64_t, 7> coords{0, 1, 2, 3, 4, 5, 6};
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
