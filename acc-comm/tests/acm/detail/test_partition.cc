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
    {
        const ac::shape mm{4, 4};
        const ac::shape nn{2, 2};
        const ac::index rr{1, 1};

        std::cout << "Segments" << std::endl;
        const auto segments{partition(mm, nn, rr)};
        for (const auto& segment : segments)
            std::cout << segment << std::endl;

        std::cout << "Pruned segments" << std::endl;
        const auto pruned_segments{prune(segments, nn, rr)};
        for (const auto& segment : pruned_segments)
            std::cout << segment << std::endl;

        std::cout << "Pruned segments (inverted selection)" << std::endl;
        const auto inv_pruned_segments{prune(segments, nn, rr, true)};
        for (const auto& segment : inv_pruned_segments)
            std::cout << segment << std::endl;

        ERRCHK(prune(segments, nn, rr).size() == segments.size() - 1);
        ERRCHK(prune(segments, nn, rr, true).size() == 1);
    }
    PRINT_LOG_INFO("OK");

    return EXIT_SUCCESS;
}
