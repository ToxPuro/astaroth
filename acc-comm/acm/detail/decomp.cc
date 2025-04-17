#include "decomp.h"

#include <algorithm>
#include <vector>

#include "errchk.h"
#include "type_conversion.h"

/** Return the non-trivial unique factors of n in increasing order */
static std::vector<uint64_t>
factorize(uint64_t n)
{
    std::vector<uint64_t> factors;
    for (uint64_t d{2}; d * d <= n; ++d) {
        while (n % d == 0) {
            factors.push_back(d);
            n /= d;
        }
    }
    if (n > 1)
        factors.push_back(n); // Push back the remainder

    auto last{std::unique(factors.begin(), factors.end())};
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
surface_area_to_volume(const ac::shape& nn)
{
    const ac::index rr{ac::make_index(nn.size(), 1)};
    return static_cast<double>((prod(as<uint64_t>(2) * rr + nn))) / static_cast<double>(prod(nn));
}

ac::shape
decompose(const ac::shape& nn, uint64_t nprocs)
{
    ac::shape local_nn{nn};
    ac::shape decomp{ac::make_shape(nn.size(), 1)};

    // More flexible dims (inspired by W.D. Gropp https://doi.org/10.1145/3236367.3236377)
    // Adapted to try out all factors to work with a wider range of dims
    // Or maybe this is what they meant all along, but the description was just unclear
    // UPDATE: Actually this is now whole different concept: now searched for decomp that maximizes
    // surface area to volume ratio at each slice.
    // Greedy algorithm: choose the current best factor at each iteration
    while (prod(nn) != prod(nprocs * local_nn * decomp)) {
        size_t   best_axis{SIZE_MAX};
        uint64_t best_factor{0};
        double   best_sa_to_vol{std::numeric_limits<double>::max()};

        auto factors{factorize(nprocs)};
        for (const auto& factor : factors) {
            for (size_t axis{nn.size() - 1}; axis < nn.size(); --axis) {
                if ((local_nn[axis] % factor) == 0) {
                    auto test_nn{local_nn};
                    test_nn[axis] /= factor;
                    double sa_to_vol{surface_area_to_volume(test_nn)};
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

// Initial version. Issues: does not produce a good mapping
std::vector<ac::shape>
decompose_hierarchical(const ac::shape& nn, const std::vector<uint64_t>& nprocs_per_layer)
{
    std::vector<ac::shape> decompositions;

    ac::shape curr_nn{nn};
    for (const auto& nprocs : nprocs_per_layer) {
        const ac::shape decomp{decompose(curr_nn, nprocs)};
        decompositions.push_back(decomp);
        curr_nn = curr_nn / decomp;
    }
    std::reverse(decompositions.begin(), decompositions.end());
    return decompositions;
}

// Better (under testing)
// Note: does not handle decompositions not divisible by two: should
// incorporate factoring in the same way as with decompose
std::vector<ac::shape>
decompose_hierarchical_alt(const ac::shape& global_nn, const size_t nprocs)
{
    auto decomposition(decompose(global_nn, nprocs));
    auto local_nn{global_nn / decomposition};

    std::vector<ac::shape> decompositions{ac::make_shape(global_nn.size(), 1)};
    while (!(local_nn == global_nn)) {
        uint64_t best_axis{0};
        uint64_t best_count{0};
        for (size_t axis{0}; axis < global_nn.size(); ++axis) {
            if (local_nn[axis] == global_nn[axis])
                continue;

            const auto count{prod(local_nn) / local_nn[axis]};
            // PRINT_DEBUG(axis);
            // PRINT_DEBUG(count);
            if (count >= best_count) {
                best_axis  = axis;
                best_count = count;
            }
        }
        ERRCHK(best_count > 0);
        // PRINT_DEBUG(local_nn);
        // PRINT_DEBUG(best_axis);
        // PRINT_DEBUG(best_count);
        auto decomp{ac::make_index(local_nn.size(), 1)};
        decomp[best_axis] *= 2;
        local_nn[best_axis] *= 2;
        // PRINT_DEBUG(decomp);
        // PRINT_DEBUG(local_nn);
        decompositions.push_back(decomp);
    }
    return decompositions;
}

ac::index
hierarchical_to_spatial(const uint64_t index, const std::vector<ac::shape>& decompositions)
{
    ERRCHK(decompositions.size() > 0);
    const size_t ndims = decompositions[0].size();
    ac::index    coords{ac::make_index(ndims, 0)};
    ac::index    scale{ac::make_index(ndims, 1)};
    ERRCHK(coords[0] == 0);
    ERRCHK(scale[0] == 1);
    for (const auto& dims : decompositions) {
        coords = coords + scale * to_spatial(index / prod(scale), dims);
        scale  = scale * dims;
    }
    return coords;
}

uint64_t
hierarchical_to_linear(const ac::index& coords, const std::vector<ac::shape>& decompositions)
{
    ac::index scale{ac::make_index(coords.size(), 1)};
    ERRCHK(scale[0] == 1);
    uint64_t index{0};
    for (const auto& dims : decompositions) {
        index = index + prod(scale) * to_linear((coords / scale) % dims, dims);
        scale = scale * dims;
    }
    return index;
}

ac::shape
hierarchical_decomposition_to_global(const std::vector<ac::shape>& decomposition)
{
    ERRCHK(decomposition.size() > 0);
    ac::shape global_decomp{ac::make_shape(decomposition[0].size(), 1)};
    for (const auto& vec : decomposition)
        global_decomp = mul(global_decomp, vec);
    return global_decomp;
}
