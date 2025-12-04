#include <cstdlib>
#include <memory>
#include <numeric>

#include "acm/detail/decomp.h"

static uint64_t
vecprod(const std::vector<uint64_t>& vec)
{
    return std::reduce(vec.begin(),
                       vec.end(),
                       static_cast<uint64_t>(1),
                       std::multiplies<uint64_t>());
}

// REMOVE
#include "acm/detail/print_debug.h"
static std::vector<uint64_t>
get_nprocs_per_layer(const uint64_t& nprocs, const std::vector<uint64_t>& max_per_layer)
{
    uint64_t              curr_nprocs{nprocs};
    std::vector<uint64_t> nprocs_per_layer;
    for (const auto& elem : max_per_layer) {
        nprocs_per_layer.push_back(std::min(curr_nprocs, elem));
        curr_nprocs /= nprocs_per_layer.back();
    }
    nprocs_per_layer.push_back(curr_nprocs); // Push remainder
    ERRCHK(vecprod(nprocs_per_layer) == nprocs);
    return nprocs_per_layer;
}

int
main()
{
    {
        const uint64_t  nprocs{32};
        const ac::shape nn{512, 128, 128};
        const auto      decomp{decompose(nn, nprocs)};
        const auto      local_nn{nn / decomp};
        ERRCHK(nn == decomp * local_nn);
    }
    {
        const uint64_t  nprocs{7};
        const ac::shape nn{7 * 20, 128, 128};
        const auto      decomp{decompose(nn, nprocs)};
        const auto      local_nn{nn / decomp};
        ERRCHK(nn == decomp * local_nn);
    }
    {
        const uint64_t  nprocs{11};
        const ac::shape nn{9, 10, 11};
        const auto      decomp{decompose(nn, nprocs)};
        const auto      local_nn{nn / decomp};
        ERRCHK(nn == decomp * local_nn);
    }
    {
        const uint64_t  nprocs{50};
        const ac::shape nn{20, 30, 40};
        const ac::shape decomp{decompose(nn, nprocs)};
        const ac::shape local_nn = nn / decomp;
        ERRCHK(nn == decomp * local_nn);
    }
    {
        ac::shape             nn{256, 128, 128};
        std::vector<uint64_t> nprocs_per_layer{16, 4, 2};
        const auto            decompositions{decompose_hierarchical(nn, nprocs_per_layer)};

        std::vector<ac::index> offsets;
        for (const auto& decomp : decompositions) {
            nn = nn / decomp;
            offsets.push_back(nn);
        }
        // ac::index is then
        // offset[0] * decompositions[0] + offset[1] * decompositions[1] + ...
        ERRCHK(prod(decompositions[0]) == 4 * 2 * 2);
        ERRCHK(prod(decompositions[1]) == 1 * 2 * 2);
        ERRCHK(prod(decompositions[2]) == 2 * 1 * 1);
    }
    {
        ac::shape             nn{32, 32, 32};
        std::vector<uint64_t> nprocs_per_layer{16, 8, 4};
        const auto            decompositions{decompose_hierarchical(nn, nprocs_per_layer)};

        std::vector<ac::index> offsets;
        for (const auto& decomp : decompositions) {
            nn = nn / decomp;
            offsets.push_back(nn);
            // PRINDIMST_DEBUG(decomp);
        }

        for (size_t i{0}; i < vecprod(nprocs_per_layer); ++i) {
            // Forward
            ac::index coords{0, 0, 0};
            ac::index scale{1, 1, 1};
            for (size_t j{decompositions.size() - 1}; j < decompositions.size(); --j) {
                coords = coords + scale * to_spatial(i / prod(scale), decompositions[j]);
                scale  = scale * decompositions[j];
            }
            // PRINDIMST_DEBUG(i);
            // PRINDIMST_DEBUG(coords);

            // Backward
            scale = ac::index{1, 1, 1};
            size_t index{0};
            for (size_t j{decompositions.size() - 1}; j < decompositions.size(); --j) {
                index = index + prod(scale) * to_linear((coords / scale) % decompositions[j],
                                                        decompositions[j]);
                scale = scale * decompositions[j];
            }
            ERRCHK(i == index);
        }
    }
    {
        ac::shape             nn{32, 16};
        std::vector<uint64_t> nprocs_per_layer{16, 4};
        const auto            decompositions{decompose_hierarchical(nn, nprocs_per_layer)};
        for (size_t i{0}; i < vecprod(nprocs_per_layer); ++i) {
            ERRCHK(i == hierarchical_to_linear(hierarchical_to_spatial(i, decompositions),
                                               decompositions));
        }
    }
    {
        ac::shape             nn{130, 111, 64, 250 * 7};
        std::vector<uint64_t> nprocs_per_layer{7, 5};
        const auto            decompositions{decompose_hierarchical(nn, nprocs_per_layer)};
        for (size_t i{0}; i < vecprod(nprocs_per_layer); ++i) {
            ERRCHK(i == hierarchical_to_linear(hierarchical_to_spatial(i, decompositions),
                                               decompositions));
        }
    }
    {
        ac::shape             nn{32, 8, 128, 64};
        std::vector<uint64_t> nprocs_per_layer{32, 64, 2, 8};
        const auto            decompositions{decompose_hierarchical(nn, nprocs_per_layer)};
        for (size_t i{0}; i < vecprod(nprocs_per_layer); ++i) {
            ERRCHK(i == hierarchical_to_linear(hierarchical_to_spatial(i, decompositions),
                                               decompositions));
        }
    }
    {
        std::vector<ac::shape> decompositions{ac::shape{2, 2}, ac::shape{4, 1}, ac::shape{1, 4}};
        const ac::shape        gdecomp{8, 8};
        const size_t           count{prod(gdecomp)};

        auto buf{std::make_unique<uint64_t[]>(count)};

        for (size_t i{0}; i < count; ++i) {
            ac::index coords                        = hierarchical_to_spatial(i, decompositions);
            buf[coords[0] + coords[1] * gdecomp[0]] = i;
        }
        // ndarray_print("buf", gdecomp.size(), gdecomp.data, buf.get());
    }
    {
        std::vector<ac::shape> decompositions{ac::shape{2, 2}, ac::shape{4, 1}, ac::shape{1, 4}};
        ERRCHK((hierarchical_decomposition_to_global(decompositions) == ac::shape{8, 8}));
    }
    {
        const std::vector<uint64_t> max_nprocs_per_layer{2, 2, 2};
        const size_t                nprocs{512};
        const auto nprocs_per_layer{get_nprocs_per_layer(nprocs, max_nprocs_per_layer)};
        // PRINT_DEBUG_VECTOR(nprocs_per_layer);

        const ac::shape nn{2048, 1024, 512};
        const auto      decompositions{decompose_hierarchical(nn, nprocs_per_layer)};
        // PRINT_DEBUG_VECTOR(decompositions);
        ERRCHK((decompositions[0] == ac::shape{2, 1, 1}));
        ERRCHK((decompositions[1] == ac::shape{1, 2, 1}));
        ERRCHK((decompositions[2] == ac::shape{1, 1, 2}));
        ERRCHK((decompositions[3] == ac::shape{8, 4, 2}));
    }
    {
        const std::vector<uint64_t> max_nprocs_per_layer{2, 8, 4};
        const size_t                nprocs{1024};
        const auto nprocs_per_layer{get_nprocs_per_layer(nprocs, max_nprocs_per_layer)};
        PRINT_DEBUG_VECTOR(nprocs_per_layer);
        ERRCHK(nprocs_per_layer.at(3) == 16);

        const ac::shape nn{1024, 1024, 1024};
        const auto      decompositions{decompose_hierarchical(nn, nprocs_per_layer)};
        PRINT_DEBUG_VECTOR(decompositions);
        ERRCHK((decompositions[0] == ac::shape{1, 1, 2}));
        ERRCHK((decompositions[1] == ac::shape{2, 2, 2}));
        ERRCHK((decompositions[2] == ac::shape{2, 2, 1}));
        ERRCHK((decompositions[3] == ac::shape{2, 2, 4}));
    }
    PRINT_LOG_INFO("OK");
    return EXIT_SUCCESS;
}
