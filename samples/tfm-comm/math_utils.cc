#include "math_utils.h"

#include "errchk.h"

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

bool
within_box(const Index& coords, const Shape& box_dims, const Index& box_offset)
{
    for (size_t i = 0; i < coords.count; ++i)
        if (coords[i] < box_offset[i] || coords[i] >= box_offset[i] + box_dims[i])
            return false;
    return true;
}

void
test_math_utils(void)
{
    {
        const Index coords{0, 0, 0};
        const Shape shape{1, 1, 1};

        ERRCHK(to_linear(coords, shape) == 0);
    }
    {
        const Index coords{1, 0};
        const Shape shape{32, 32};

        ERRCHK(to_linear(coords, shape) == 1);
    }
    {
        const Index coords{31, 0};
        const Shape shape{32, 32};

        ERRCHK(to_linear(coords, shape) == 31);
    }
    {
        const Index coords{0, 31};
        const Shape shape{32, 32};

        ERRCHK(to_linear(coords, shape) == 31 * 32);
    }
    {
        const Index coords{1, 2, 3, 4};
        const Shape shape{10, 9, 8, 7};

        ERRCHK(to_linear(coords, shape) == 1 + 2 * 10 + 3 * 10 * 9 + 4 * 10 * 9 * 8);
    }
}
