#include <cstdlib>

#include "acm/detail/partition.h"

int
main()
{
    {
        const ac::ntuple<uint64_t> mm{8};
        const ac::ntuple<uint64_t> nn{6};
        const ac::ntuple<uint64_t> nn_offset{1};
        ERRCHK(partition(mm, nn, nn_offset).size() == 3);
    }
    {
        const ac::ntuple<uint64_t> mm{8, 8};
        const ac::ntuple<uint64_t> nn{6, 6};
        const ac::ntuple<uint64_t> nn_offset{1, 1};
        ERRCHK(partition(mm, nn, nn_offset).size() == 9);
    }
    {
        const ac::ntuple<uint64_t> mm{8, 8, 8};
        const ac::ntuple<uint64_t> nn{6, 6, 6};
        const ac::ntuple<uint64_t> nn_offset{1, 1, 1};
        ERRCHK(partition(mm, nn, nn_offset).size() == 27);
    }
    {
        const ac::ntuple<uint64_t> mm{5, 6, 7, 8};
        const ac::ntuple<uint64_t> nn{3, 4, 5, 6};
        const ac::ntuple<uint64_t> nn_offset{1, 1, 1, 1};
        ERRCHK(partition(mm, nn, nn_offset).size() == 3 * 3 * 3 * 3);
    }
    {
        const ac::ntuple<uint64_t> mm{4, 4, 4};
        const ac::ntuple<uint64_t> nn{4, 4, 4};
        const ac::ntuple<uint64_t> nn_offset{0, 0, 0};
        ERRCHK(partition(mm, nn, nn_offset).size() == 1);
    }
    {
        const ac::ntuple<uint64_t> mm{4, 4};
        const ac::ntuple<uint64_t> nn{3, 3};
        const ac::ntuple<uint64_t> nn_offset{1, 1};
        ERRCHK(partition(mm, nn, nn_offset).size() == 4);
    }
    PRINT_LOG_INFO("OK");

    return EXIT_SUCCESS;
}
