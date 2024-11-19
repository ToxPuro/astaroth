#include "decomp.h"

#include "errchk.h"
#include "print_debug.h"

#include <algorithm>
#include <iostream>
#include <vector>
#include <numeric>

#include "math_utils.h"




static uint64_t
vecprod(const std::vector<uint64_t>& vec)
{
    return std::reduce(vec.begin(), vec.end(), static_cast<uint64_t>(1),
                       std::multiplies<uint64_t>());
}

#include "ndarray.h"
#include <memory.h>

void
test_decomp(void)
{
    {
        const uint64_t nprocs{32};
        const Shape<3> nn{512, 128, 128};
        const auto decomp{decompose(nn, nprocs)};
        const auto local_nn{nn / decomp};
        ERRCHK(nn == decomp * local_nn);
    }
    {
        const uint64_t nprocs{7};
        const Shape<3> nn{7 * 20, 128, 128};
        const auto decomp{decompose(nn, nprocs)};
        const auto local_nn{nn / decomp};
        ERRCHK(nn == decomp * local_nn);
    }
    {
        const uint64_t nprocs{11};
        const Shape<3> nn{9, 10, 11};
        const auto decomp{decompose(nn, nprocs)};
        const auto local_nn{nn / decomp};
        ERRCHK(nn == decomp * local_nn);
    }
    {
        const uint64_t nprocs{50};
        const Shape<3> nn{20, 30, 40};
        const Shape<3> decomp{decompose(nn, nprocs)};
        const Shape<3> local_nn = nn / decomp;
        ERRCHK(nn == decomp * local_nn);
    }
    {
        Shape<3> nn{256, 128, 128};
        std::vector<uint64_t> nprocs_per_layer{16, 4, 2};
        const auto decompositions{decompose_hierarchical(nn, nprocs_per_layer)};

        std::vector<Index<3>> offsets;
        for (const auto& decomp : decompositions) {
            nn = nn / decomp;
            offsets.push_back(nn);
        }
        // Index<3> is then
        // offset[0] * decompositions[0] + offset[1] * decompositions[1] + ...
        ERRCHK(prod(decompositions[0]) == 2);
        ERRCHK(prod(decompositions[1]) == 4);
        ERRCHK(prod(decompositions[2]) == 16);
    }
    {
        Shape<3> nn{32, 32, 32};
        std::vector<uint64_t> nprocs_per_layer{16, 8, 4};
        const auto decompositions{decompose_hierarchical(nn, nprocs_per_layer)};

        std::vector<Index<3>> offsets;
        for (const auto& decomp : decompositions) {
            nn = nn / decomp;
            offsets.push_back(nn);
            // PRINDIMST_DEBUG(decomp);
        }

        for (size_t i{0}; i < vecprod(nprocs_per_layer); ++i) {
            // Forward
            Index<3> coords = {0, 0, 0};
            Index<3> scale  = {1, 1, 1};
            for (size_t j{decompositions.size() - 1}; j < decompositions.size(); --j) {
                coords = coords + scale * to_spatial(i / prod(scale), decompositions[j]);
                scale  = scale * decompositions[j];
            }
            // PRINDIMST_DEBUG(i);
            // PRINDIMST_DEBUG(coords);

            // Backward
            scale        = {1, 1, 1};
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
        Shape<2> nn{32, 16};
        std::vector<uint64_t> nprocs_per_layer{16, 4};
        const auto decompositions{decompose_hierarchical(nn, nprocs_per_layer)};
        for (size_t i{0}; i < vecprod(nprocs_per_layer); ++i) {
            ERRCHK(i == hierarchical_to_linear(hierarchical_to_spatial(i, decompositions),
                                               decompositions));
        }
    }
    {
        Shape<4> nn{130, 111, 64, 250 * 7};
        std::vector<uint64_t> nprocs_per_layer{7, 5};
        const auto decompositions{decompose_hierarchical(nn, nprocs_per_layer)};
        for (size_t i{0}; i < vecprod(nprocs_per_layer); ++i) {
            ERRCHK(i == hierarchical_to_linear(hierarchical_to_spatial(i, decompositions),
                                               decompositions));
        }
    }
    {
        Shape<4> nn{32, 8, 128, 64};
        std::vector<uint64_t> nprocs_per_layer{32, 64, 2, 8};
        const auto decompositions{decompose_hierarchical(nn, nprocs_per_layer)};
        for (size_t i{0}; i < vecprod(nprocs_per_layer); ++i) {
            ERRCHK(i == hierarchical_to_linear(hierarchical_to_spatial(i, decompositions),
                                               decompositions));
        }
    }
    {
        std::vector<Shape<2>> decompositions{Shape<2>{2, 2}, Shape<2>{4, 1}, Shape<2>{1, 4}};
        const Shape<2> gdecomp{8, 8};
        const size_t count{prod(gdecomp)};

        auto buf{std::make_unique<uint64_t[]>(count)};

        for (size_t i{0}; i < count; ++i) {
            Index<2> coords                         = hierarchical_to_spatial(i, decompositions);
            buf[coords[0] + coords[1] * gdecomp[0]] = i;
        }
        // ndarray_print("buf", gdecomp.size(), gdecomp.data, buf.get());
    }
}
