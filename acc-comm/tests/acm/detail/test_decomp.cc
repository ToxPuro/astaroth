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
        ERRCHK(prod(decompositions[0]) == 2);
        ERRCHK(prod(decompositions[1]) == 4);
        ERRCHK(prod(decompositions[2]) == 16);
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
        ac::shape  nn{256, 128, 128};
        uint64_t   nprocs{16 * 4 * 2};
        const auto decompositions{decompose_hierarchical_alt(nn, nprocs)};

        std::vector<ac::index> offsets;
        for (const auto& decomp : decompositions) {
            nn = nn / decomp;
            offsets.push_back(nn);
        }
        // ac::index is then
        // offset[0] * decompositions[0] + offset[1] * decompositions[1] + ...
        ERRCHK(prod(decompositions[0]) == 1);
        ERRCHK(prod(decompositions[1]) == 2);
        ERRCHK(prod(decompositions[2]) == 2);
        ERRCHK(prod(hierarchical_decomposition_to_global(decompositions)) == nprocs);
    }
    {
        ac::shape  nn{32, 32, 32};
        uint64_t   nprocs{16 * 8 * 4};
        const auto decompositions{decompose_hierarchical_alt(nn, nprocs)};

        std::vector<ac::index> offsets;
        for (const auto& decomp : decompositions) {
            nn = nn / decomp;
            offsets.push_back(nn);
            // PRINDIMST_DEBUG(decomp);
        }

        for (size_t i{0}; i < nprocs; ++i) {
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
        ac::shape  nn{32, 16};
        uint64_t   nprocs{16 * 4};
        const auto decompositions{decompose_hierarchical_alt(nn, nprocs)};
        for (size_t i{0}; i < nprocs; ++i) {
            ERRCHK(i == hierarchical_to_linear(hierarchical_to_spatial(i, decompositions),
                                               decompositions));
        }
    }
    {
        ac::shape  nn{64, 128, 32, 16};
        uint64_t   nprocs{32};
        const auto decompositions{decompose_hierarchical_alt(nn, nprocs)};
        for (size_t i{0}; i < nprocs; ++i) {
            ERRCHK(i == hierarchical_to_linear(hierarchical_to_spatial(i, decompositions),
                                               decompositions));
        }
    }
    {
        ac::shape  nn{32, 8, 128, 64};
        uint64_t   nprocs{32 * 64 * 2 * 8};
        const auto decompositions{decompose_hierarchical_alt(nn, nprocs)};
        for (size_t i{0}; i < nprocs; ++i) {
            ERRCHK(i == hierarchical_to_linear(hierarchical_to_spatial(i, decompositions),
                                               decompositions));
        }
    }
    PRINT_LOG_INFO("OK");
    return EXIT_SUCCESS;
}
