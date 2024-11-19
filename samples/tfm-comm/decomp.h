#pragma once

#include <vector>
#include <algorithm>

#include "datatypes.h"
#include "math_utils.h"

/** Return the non-trivial unique factors of n in increasing order */
inline std::vector<uint64_t>
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
template <size_t N>
inline double
surface_area_to_volume(const Shape<N>& nn)
{
    const auto rr{ones<uint64_t, N>()};
    return static_cast<double>((prod(as<uint64_t>(2) * rr + nn))) / static_cast<double>(prod(nn));
}

/**
 * Perform a simple decomposition of domain nn to nprocs partitions.
 * Uses a greedy algorithm to maximize the surface area to volume
 * ratio at each cut.
 */
template <size_t N>
Shape<N>
decompose(const Shape<N>& nn, uint64_t nprocs)
{
    Shape<N> local_nn{nn};
    Shape<N> decomp{ones<uint64_t, N>()};

    // More flexible dims (inspired by W.D. Gropp https://doi.org/10.1145/3236367.3236377)
    // Adapted to try out all factors to work with a wider range of dims
    // Or maybe this is what they meant all along, but the description was just unclear
    // UPDATE: Actually this is now whole different concept: now searched for decomp that maximizes
    // surface area to volume ratio at each slice.
    // Greedy algorithm: choose the current best factor at each iteration
    while (prod(nn) != prod(nprocs * local_nn * decomp)) {
        size_t best_axis{SIZE_MAX};
        uint64_t best_factor{0};
        double best_sa_to_vol{std::numeric_limits<double>::max()};

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

/**
 * Perform a layered decomposition.
 * Returns a vector of decompositions ordered from higher
 * granularity (i.e. core-level) to lower (i.e. node-level).
 * The input vector nprocs_per_layer is likewise ordered from
 * high to low granularity and indicates the number of partitions
 * on each level of the decomposition hierarchy.
 *
 * For example, a process running on 8 nodes, consisting of
 * 4 GPUs, where each GPU is a multi-chip module of 2 devices,
 * the decomposition can be calculated by
 * decompose_hierarchical(nn, std::vector<uint64_t>{2, 4, 8});
 */
template <size_t N>
std::vector<Shape<N>>
decompose_hierarchical(const Shape<N>& nn, const std::vector<uint64_t>& nprocs_per_layer)
{
    std::vector<Shape<N>> decompositions;

    Shape<N> curr_nn{nn};
    for (const auto& nprocs : nprocs_per_layer) {
        const Shape<N> decomp{decompose(curr_nn, nprocs)};
        decompositions.push_back(decomp);
        curr_nn = curr_nn / decomp;
    }
    std::reverse(decompositions.begin(), decompositions.end());
    return decompositions;
}

template <size_t N>
Index<N>
hierarchical_to_spatial(const uint64_t in_index, const std::vector<Shape<N>>& in_decompositions)
{
    Index<N> coords{};
    Index<N> scale{ones<uint64_t, N>()};
    ERRCHK(coords[0] == 0);
    ERRCHK(scale[0] == 1);
    for (const auto& dims : in_decompositions) {
        coords = coords + scale * to_spatial(in_index / prod(scale), dims);
        scale  = scale * dims;
    }
    return coords;
}

template <size_t N>
uint64_t
hierarchical_to_linear(const Index<N>& in_coords, const std::vector<Shape<N>>& in_decompositions)
{
    Index<N> scale{ones<uint64_t, N>()};
    ERRCHK(scale[0] == 1);
    uint64_t index{0};
    for (const auto& dims : in_decompositions) {
        index = index + prod(scale) * to_linear((in_coords / scale) % dims, dims);
        scale = scale * dims;
    }
    return index;
}

void test_decomp(void);
