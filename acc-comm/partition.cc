#include "partition.h"

#include "print_debug.h"

void
test_partition(void)
{
    {
        const ac::array<uint64_t, 1> mm{8};
        const ac::array<uint64_t, 1> nn{6};
        const ac::array<uint64_t, 1> nn_offset{1};
        ERRCHK(partition(mm, nn, nn_offset).size() == 3);
    }
    {
        const ac::array<uint64_t, 2> mm{8, 8};
        const ac::array<uint64_t, 2> nn{6, 6};
        const ac::array<uint64_t, 2> nn_offset{1, 1};
        ERRCHK(partition(mm, nn, nn_offset).size() == 9);
    }
    {
        const ac::array<uint64_t, 3> mm{8, 8, 8};
        const ac::array<uint64_t, 3> nn{6, 6, 6};
        const ac::array<uint64_t, 3> nn_offset{1, 1, 1};
        ERRCHK(partition(mm, nn, nn_offset).size() == 27);
    }
    {
        const ac::array<uint64_t, 4> mm{5, 6, 7, 8};
        const ac::array<uint64_t, 4> nn{3, 4, 5, 6};
        const ac::array<uint64_t, 4> nn_offset{1, 1, 1, 1};
        ERRCHK(partition(mm, nn, nn_offset).size() == 3 * 3 * 3 * 3);
    }
    {
        const ac::array<uint64_t, 3> mm{4, 4, 4};
        const ac::array<uint64_t, 3> nn{4, 4, 4};
        const ac::array<uint64_t, 3> nn_offset{0, 0, 0};
        ERRCHK(partition(mm, nn, nn_offset).size() == 1);
    }
    {
        const ac::array<uint64_t, 2> mm{4, 4};
        const ac::array<uint64_t, 2> nn{3, 3};
        const ac::array<uint64_t, 2> nn_offset{1, 1};
        ERRCHK(partition(mm, nn, nn_offset).size() == 4);
    }
}
