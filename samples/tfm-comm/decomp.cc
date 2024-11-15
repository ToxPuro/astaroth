#include "decomp.h"

#include "errchk.h"
#include "print_debug.h"

#include <algorithm>
#include <vector>

#include "math_utils.h"

// Returns the non-trivial unique factors of n in increasing order
static std::vector<uint64_t>
factorize(uint64_t n)
{
    std::vector<uint64_t> factors;
    for (uint64_t d = 2; d * d <= n; ++d) {
        while (n % d == 0) {
            factors.push_back(d);
            n /= d;
        }
    }
    if (n > 1)
        factors.push_back(n); // Push back the remainder

    auto last = std::unique(factors.begin(), factors.end());
    factors.erase(last, factors.end());
    return factors;
}

/** Calculate the n-1 manifold extent to n manifold extent ratio :)
 * I.e. surface to volume.
 *
 * Lower is better
 *
 * Also does not currently calculate the actual surface to volume ratio,
 * but a related number (halo to computational domain ratio)
 */
static double
surface_area_to_volume(const Shape& nn)
{
    const Shape rr(nn.count, 1);
    return static_cast<double>((prod(as<uint64_t>(2) * rr + nn))) / static_cast<double>(prod(nn));
}

Shape
decompose(const Shape& nn, uint64_t nprocs)
{
    Shape local_nn(nn);
    Shape decomp(nn.count, 1);

    // More flexible dims (inspired by W.D. Gropp https://doi.org/10.1145/3236367.3236377)
    // Adapted to try out all factors to work with a wider range of dims
    // Or maybe this is what they meant all along, but the description was just unclear
    // UPDATE: Actually this is now whole different concept: now searched for decomp that maximizes
    // surface area to volume ratio at each slice.
    // Greedy algorithm: choose the current best factor at each iteration
    while (prod(nn) != prod(nprocs * local_nn * decomp)) {
        size_t best_axis      = SIZE_MAX;
        uint64_t best_factor  = 0;
        double best_sa_to_vol = std::numeric_limits<double>::max();

        auto factors = factorize(nprocs);
        for (const auto& factor : factors) {
            for (size_t axis = nn.count - 1; axis < nn.count; --axis) {
                if ((local_nn[axis] % factor) == 0) {
                    auto test_nn(local_nn);
                    test_nn[axis] /= factor;
                    double sa_to_vol = surface_area_to_volume(test_nn);
                    if (sa_to_vol < best_sa_to_vol) {
                        best_axis      = axis;
                        best_factor    = factor;
                        best_sa_to_vol = sa_to_vol;
                    }
                }
            }
        }
        ERRCHK(best_factor > 0);
        nprocs /= best_factor;
        local_nn[best_axis] /= best_factor;
        decomp[best_axis] *= best_factor;
    }

    ERRCHK(prod(nn) == prod(nprocs * local_nn * decomp));
    return decomp;
}

std::vector<Shape>
decompose_hierarchical(const Shape& nn, const std::vector<uint64_t>& nprocs_per_layer)
{
    std::vector<Shape> decompositions;

    Shape curr_nn(nn);
    for (const auto& nprocs : nprocs_per_layer) {
        const Shape decomp = decompose(curr_nn, nprocs);
        decompositions.push_back(decomp);
        curr_nn = curr_nn / decomp;
    }
    return decompositions;
}

static Index
to_spatial(const uint64_t in_index, const std::vector<Shape>& in_decompositions)
{
    const size_t count = in_decompositions[0].count;
    Index coords(count, static_cast<uint64_t>(0));
    Index scale(count, static_cast<uint64_t>(1));
    ERRCHK(coords[0] == 0);
    ERRCHK(scale[0] == 1);
    for (size_t j = in_decompositions.size() - 1; j < in_decompositions.size(); --j) {
        coords = coords + scale * to_spatial(in_index / prod(scale), in_decompositions[j]);
        scale  = scale * in_decompositions[j];
    }
    return coords;
}

static uint64_t
to_linear(const Index& in_coords, const std::vector<Shape>& in_decompositions)
{
    const size_t count = in_decompositions[0].count;
    Index scale(count, static_cast<uint64_t>(1));
    ERRCHK(scale[0] == 1);
    uint64_t index = 0;
    for (size_t j = in_decompositions.size() - 1; j < in_decompositions.size(); --j) {
        index = index + prod(scale) * to_linear((in_coords / scale) % in_decompositions[j],
                                                in_decompositions[j]);
        scale = scale * in_decompositions[j];
    }
    return index;
}

static uint64_t
vecprod(const std::vector<uint64_t>& vec)
{
    return std::reduce(vec.begin(), vec.end(), static_cast<uint64_t>(1),
                       std::multiplies<uint64_t>());
}

void
test_decomp(void)
{
    {
        const uint64_t nprocs = 32;
        const Shape nn{512, 128, 128};
        const Shape decomp   = decompose(nn, nprocs);
        const Shape local_nn = nn / decomp;
        ERRCHK(nn == decomp * local_nn);
    }
    {
        const uint64_t nprocs = 7;
        const Shape nn{7 * 20, 128, 128};
        const Shape decomp   = decompose(nn, nprocs);
        const Shape local_nn = nn / decomp;
        ERRCHK(nn == decomp * local_nn);
    }
    {
        const uint64_t nprocs = 11;
        const Shape nn{9, 10, 11};
        const Shape decomp   = decompose(nn, nprocs);
        const Shape local_nn = nn / decomp;
        ERRCHK(nn == decomp * local_nn);
    }
    {
        const uint64_t nprocs = 50;
        const Shape nn{20, 30, 40};
        const Shape decomp   = decompose(nn, nprocs);
        const Shape local_nn = nn / decomp;
        ERRCHK(nn == decomp * local_nn);
    }
    {
        Shape nn{256, 128, 128};
        std::vector<uint64_t> nprocs_per_layer{16, 4, 2};
        const auto decompositions = decompose_hierarchical(nn, nprocs_per_layer);

        std::vector<Index> offsets;
        for (const auto& decomp : decompositions) {
            nn = nn / decomp;
            offsets.push_back(nn);
        }
        // Index is then
        // offset[0] * decompositions[0] + offset[1] * decompositions[1] + ...
        ERRCHK(prod(decompositions[0]) == 16);
        ERRCHK(prod(decompositions[1]) == 4);
        ERRCHK(prod(decompositions[2]) == 2);
    }
    {
        Shape nn{32, 32, 32};
        std::vector<uint64_t> nprocs_per_layer{16, 8, 4};
        const auto decompositions = decompose_hierarchical(nn, nprocs_per_layer);

        std::vector<Index> offsets;
        for (const auto& decomp : decompositions) {
            nn = nn / decomp;
            offsets.push_back(nn);
            // PRINT_DEBUG(decomp);
        }

        for (size_t i = 0; i < vecprod(nprocs_per_layer); ++i) {
            // Forward
            Index coords = {0, 0, 0};
            Index scale  = {1, 1, 1};
            for (size_t j = decompositions.size() - 1; j < decompositions.size(); --j) {
                coords = coords + scale * to_spatial(i / prod(scale), decompositions[j]);
                scale  = scale * decompositions[j];
            }
            // PRINT_DEBUG(i);
            // PRINT_DEBUG(coords);

            // Backward
            scale        = {1, 1, 1};
            size_t index = 0;
            for (size_t j = decompositions.size() - 1; j < decompositions.size(); --j) {
                index = index + prod(scale) * to_linear((coords / scale) % decompositions[j],
                                                        decompositions[j]);
                scale = scale * decompositions[j];
            }
            ERRCHK(i == index);
        }
    }
    {
        Shape nn{32, 16};
        std::vector<uint64_t> nprocs_per_layer{16, 4};
        const auto decompositions = decompose_hierarchical(nn, nprocs_per_layer);
        for (size_t i = 0; i < vecprod(nprocs_per_layer); ++i) {
            ERRCHK(i == to_linear(to_spatial(i, decompositions), decompositions));
        }
    }
    {
        Shape nn{130, 111, 64, 250 * 7};
        std::vector<uint64_t> nprocs_per_layer{7, 5};
        const auto decompositions = decompose_hierarchical(nn, nprocs_per_layer);
        for (size_t i = 0; i < vecprod(nprocs_per_layer); ++i) {
            ERRCHK(i == to_linear(to_spatial(i, decompositions), decompositions));
        }
    }
    {
        Shape nn{32, 8, 128, 64};
        std::vector<uint64_t> nprocs_per_layer{32, 64, 2, 8};
        const auto decompositions = decompose_hierarchical(nn, nprocs_per_layer);
        for (size_t i = 0; i < vecprod(nprocs_per_layer); ++i) {
            ERRCHK(i == to_linear(to_spatial(i, decompositions), decompositions));
        }
    }
}
