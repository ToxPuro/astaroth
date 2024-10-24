#include "decomp.h"

#include "errchk.h"
#include "print_debug.h"

#include <algorithm>
#include <vector>

// Returns the non-trivial factors of n in increasing order
std::vector<uint64_t>
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
double
surface_area_to_volume(const Shape& nn)
{
    const Shape rr(nn.count, 1);
    return static_cast<double>((prod(as<uint64_t>(2) * rr + nn))) / prod(nn);
}

Shape
decompose(const Shape& nn, uint64_t nprocs)
{
    Shape local_nn(nn);
    Shape decomp(nn.count, 1);

    // More flexible dims (inspired by W.D. Gropp https://doi.org/10.1145/3236367.3236377)
    // Adapted to try out all factors to work with a wider range of dims
    // Or maybe this is what they meant all along, but the description was just unclear
    // Actually this is now whole different concept: now searched for decomp that maximizes
    // surf area to volume ratio at each slice.
    while (prod(nn) != prod(nprocs * local_nn * decomp)) {
        size_t best_axis      = SIZE_MAX;
        uint64_t best_factor  = 0;
        double best_sa_to_vol = std::numeric_limits<double>::max();

        auto factors = factorize(nprocs);
        for (const auto& factor : factors) {
            for (size_t axis = 0; axis < nn.count; ++axis) {
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
        nprocs /= best_factor;
        local_nn[best_axis] /= best_factor;
        decomp[best_axis] *= best_factor;
    }

    ERRCHK(prod(nn) == prod(nprocs * local_nn * decomp));
    return decomp;
}

void
test_decomp(void)
{
    std::cout << "hello from decomp" << std::endl;

    const uint64_t nprocs = 32;
    const Shape nn        = {512, 128, 128};
    const Shape decomp    = decompose(nn, nprocs);
    const Shape local_nn  = nn / decomp;
    PRINT_DEBUG(nn);
    PRINT_DEBUG(decomp);
    PRINT_DEBUG(local_nn);
}
