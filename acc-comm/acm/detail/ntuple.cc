#include "ntuple.h"

#include "errchk.h"
#include "type_conversion.h"

namespace ac {

Index
make_index(const size_t count, const uint64_t& fill_value)
{
    return ac::make_ntuple(count, fill_value);
}

Shape
make_shape(const size_t count, const uint64_t& fill_value)
{
    return ac::make_ntuple(count, fill_value);
}

Direction
make_direction(const size_t count, const int64_t& fill_value)
{
    return ac::make_ntuple(count, fill_value);
}

} // namespace ac

static void
test_fn(const ac::ntuple<uint64_t>& arr)
{
    ERRCHK((arr == ac::ntuple<uint64_t>{1, 2, 3}));
}

void
test_static_ntuple()
{
    PRINT_LOG_WARNING("Not implemented");
}

template <typename Container>
static void
test_to_linear(void)
{
    {
        const Container coords{0, 0, 0};
        const Container shape{1, 1, 1};

        ERRCHK(to_linear(coords, shape) == 0);
    }
    {
        const Container coords{1, 0};
        const Container shape{32, 32};

        ERRCHK(to_linear(coords, shape) == 1);
    }
    {
        const Container coords{31, 0};
        const Container shape{32, 32};

        ERRCHK(to_linear(coords, shape) == 31);
    }
    {
        const Container coords{0, 31};
        const Container shape{32, 32};

        ERRCHK(to_linear(coords, shape) == 31 * 32);
    }
    {
        const Container coords{1, 2, 3, 4};
        const Container shape{10, 9, 8, 7};

        ERRCHK(to_linear(coords, shape) == 1 + 2 * 10 + 3 * 10 * 9 + 4 * 10 * 9 * 8);
    }
}

template <typename Container>
static void
test_to_spatial(void)
{
    {
        const uint64_t  i{0};
        const Container shape{4, 5, 6};
        const Container coords{0, 0, 0};
        ERRCHK(ac::to_spatial(i, shape) == coords);
    }
    {
        const uint64_t  i{4};
        const Container shape{4, 5, 6};
        const Container coords{0, 1, 0};
        ERRCHK(to_spatial(i, shape) == coords);
    }
    {
        const uint64_t  i{4 * 5};
        const Container shape{4, 5, 6};
        const Container coords{0, 0, 1};
        ERRCHK(to_spatial(i, shape) == coords);
    }
    {
        const uint64_t  i{4 * 5 * 6 - 1};
        const Container shape{4, 5, 6};
        const Container coords{3, 4, 5};
        ERRCHK(to_spatial(i, shape) == coords);
    }
}

template <typename Container>
static void
test_within_box(void)
{
    {
        const Container box_offset{0, 0, 0};
        const Container box_dims{10, 10, 10};
        const Container coords{0, 0, 0};
        ERRCHK(within_box(coords, box_dims, box_offset) == true);
    }
    {
        const Container box_offset{0, 0, 0};
        const Container box_dims{10, 10, 10};
        const Container coords{0, 10, 0};
        ERRCHK(within_box(coords, box_dims, box_offset) == false);
    }
    {
        const Container box_offset{0, 0, 0};
        const Container box_dims{10, 10, 10};
        const Container coords{11, 11, 11};
        ERRCHK(within_box(coords, box_dims, box_offset) == false);
    }
    {
        const Container box_offset{0, 0, 0, 0, 0, 0, 0};
        const Container box_dims{1, 2, 3, 4, 5, 6, 7};
        const Container coords{0, 1, 2, 3, 4, 5, 6};
        ERRCHK(within_box(coords, box_dims, box_offset) == true);
    }
}

void
test_ntuple(void)
{

    {
        ac::ntuple<uint64_t> a{3};
        for (size_t i{0}; i < a.size(); ++i)
            ERRCHK(a[i] == 3);
        // PRINT_DEBUG(a);
    }
    {
        ac::ntuple<uint64_t> a{2, 5};
        ERRCHK(a[0] == 2);
        ERRCHK(a[1] == 5);
        // PRINT_DEBUG(a);
    }
    {
        ac::ntuple<uint64_t> a{3, 4, 5};
        ERRCHK((a == ac::ntuple<uint64_t>{3, 4, 5}));
    }
    {
        ac::ntuple<uint64_t> a{1, 2, 3};
        ac::ntuple<uint64_t> b{4, 5, 6};
        ac::ntuple<uint64_t> c{5, 7, 9};
        ERRCHK(a + b == c);
        // PRINT_DEBUG(a + b);
    }
    {
        ac::ntuple<uint64_t> a{1, 2, 3};
        ac::ntuple<uint64_t> b{4, 5, 6};
        ac::ntuple<uint64_t> c{2, 1, 9};
        ac::ntuple<uint64_t> d{2 * 1 + 4 / 2, 2 * 2 + 5 / 1, 2 * 3 + 6 / 9};
        ERRCHK(as<uint64_t>(2) * a + b / c == d);
    }
    {
        ac::ntuple<uint64_t> a{8, 9, 10, 11};
        ac::ntuple<uint64_t> b{7, 6, 5, 4};
        ac::ntuple<uint64_t> c{1, 3, 0, 3};
        ERRCHK(a % b == c);
    }
    {
        test_fn(ac::ntuple<uint64_t>{1, 2, 3});
    }
    {
        const size_t    count       = 10;
        int             data[count] = {1};
        ac::ntuple<int> a{ac::make_ntuple_from_ptr(count, data)};
        ERRCHK(a[0] == 1);
        ERRCHK(a[9] == 0);
    }
    {
        ac::ntuple<int> a{ac::make_ntuple(100, 1)};
        ERRCHK(a[0] == 1);
        ERRCHK(a[99] == 1);
    }
    {
        test_static_ntuple();
    }
    {
        test_to_linear<ac::ntuple<uint64_t>>();
        test_to_spatial<ac::ntuple<uint64_t>>();
        test_within_box<ac::ntuple<uint64_t>>();
        test_to_linear<ac::static_ntuple<uint64_t, 10>>();
        test_to_spatial<ac::static_ntuple<uint64_t, 10>>();
        test_within_box<ac::static_ntuple<uint64_t, 10>>();
    }
    PRINT_LOG_INFO("OK");
}
