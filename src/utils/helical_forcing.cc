#include "astaroth_forcing.h"

#include <array>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <numbers>
#include <random>

#include "errchk.h"
#include "astaroth_random.h"

using vec3 = std::array<double, 3>;

template <typename T>
static void print(const std::string &label, const T &expr) {
  std::cout << label << ": " << expr << std::endl;
}

static void print(const std::string &label, const vec3 &vec) {
  std::cout << label << ": ";
  for (const auto &elem : vec)
    std::cout << elem << " ";
  std::cout << std::endl;
}

template <typename T>
static void print(const std::string &label, const std::complex<T> &c) {
  print(label + " (Re)", c.real());
  print(label + " (Im)", c.imag());
}

#define PRINT_DEBUG(x) (print(#x, (x)))

static auto dot(const vec3 &a, const vec3 &b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

static auto cross(const vec3 &a, const vec3 &b) {
  return vec3{
      a[1] * b[2] - a[2] * b[1],
      a[2] * b[0] - a[0] * b[2],
      a[0] * b[1] - a[1] * b[0],
  };
}

static auto operator+(const vec3 &a, const vec3 &b) {
  return vec3{
      a[0] + b[0],
      a[1] + b[1],
      a[2] + b[2],
  };
}

static auto operator-(const vec3 &a, const vec3 &b) {
  return vec3{
      a[0] - b[0],
      a[1] - b[1],
      a[2] - b[2],
  };
}

static auto operator*(const double fac, const vec3 &vec) {
  return vec3{
      fac * vec[0],
      fac * vec[1],
      fac * vec[2],
  };
}

static auto operator*(const vec3 &vec, const double fac) {
  return vec3{
      vec[0] * fac,
      vec[1] * fac,
      vec[2] * fac,
  };
}

static auto operator/(const vec3 &vec, const double denom) {
  return vec3{
      vec[0] / denom,
      vec[1] / denom,
      vec[2] / denom,
  };
}

static auto operator-(const vec3 &vec) {
  return vec3{
      -vec[0],
      -vec[1],
      -vec[2],
  };
}

static auto norm(const vec3 &a) { return std::sqrt(dot(a, a)); }

static auto normalized(const vec3 &vec) { return vec / norm(vec); }

static auto angle(const vec3 &a, const vec3 &b) {
  return std::acos(dot(a, b) / (norm(a) * norm(b)));
}

static auto allclose(const double a, const double b, const double rtol = 1e-5,
                     const double atol = 1e-8) {
  return std::abs(a - b) <= (atol + rtol * std::abs(b));
}

static auto allclose(const vec3 &a, const vec3 &b, const double rtol = 1e-5,
                     const double atol = 1e-8) {
  return allclose(a[0], b[0]) && allclose(a[1], b[1]) && allclose(a[2], b[2]);
}

/**
 * Generates a random uniform real in the half-open interval [low, high)
 *
 * NOTE: not thread safe.
 */
static double gen_random_uniform_real(const double low, const double high) {
  // static std::random_device rd;
  // static std::mt19937 gen{rd()};
  // static std::mt19937 gen{981598153};
  std::uniform_real_distribution dis{low, high};
  return dis(get_rng());
}

/**
 * Generates a random uniform integer in the half-open interval [low, high)
 *
 * NOTE: not thread safe.
 */
static int gen_random_uniform_int(const int low, const int high) {
  // static std::random_device rd;
  // static std::mt19937 gen{rd()};
  // static std::mt19937 gen{3657238679};

  // Note: std::uniform_int_distribution returns a *closed interval*
  // instead of open.
  // Therefore we subtract high here by 1 to get the half-open interval for
  // consistency with std::uniform_real_distribution.
  std::uniform_int_distribution dis{low, high - 1};
  return dis(get_rng());
}

/**
 * Generates a uniformly sampled point on the unit sphere with components
 * in the half-open interval [-1, 1).
 * Uses rejection sampling to ensure uniform distribution.
 *
 * NOTE: not thread safe.
 */
static vec3 gen_random_uniform_unit_vector() {
  auto gen_vec = [] {
    return vec3{
        gen_random_uniform_real(-1, 1),
        gen_random_uniform_real(-1, 1),
        gen_random_uniform_real(-1, 1),
    };
  };

  // Rejection sampling: discard vectors not on the unit ball
  auto vec{gen_vec()};
  while (dot(vec, vec) > 1)
    vec = gen_vec();

  vec = normalized(vec);
  ERRCHK_ALWAYS(allclose(norm(vec), 1));
  return vec;
}

namespace ac::helical_forcing {

/** Generates a k vector comprising integer elements of length in the closed
 * interval [kmin, kmax] */
static auto gen_k(const double kmin, const double kmax) {

  static std::vector<vec3> vectors;
  static auto kmin_stored{kmin};
  static auto kmax_stored{kmax};

  // Require that always called with the same kmin, kmax
  // Otherwise there's a potential performance pitfall due to
  // regenerating vectors at each call.
  // Uncomment to enable regeneration for different kmin, kmax
  ERRCHK_ALWAYS(kmin_stored == kmin);
  ERRCHK_ALWAYS(kmax_stored == kmax);

  // Discard precomputed vectors if calling with different kmin, kmax than
  // previously
  if (kmin_stored != kmin || kmax_stored != kmax)
    vectors.clear();

  // Regenerate vectors if they have not been generated for the given kmin, kmax
  if (vectors.size() == 0) {
    const auto max_count{static_cast<int64_t>(std::ceil(kmax * kmax * kmax))};
    // PRINT_DEBUG(max_count);
    std::cout << "Regenerating vectors" << std::endl;

    for (int64_t k{-max_count}; k <= max_count; ++k) {
      for (int64_t j{-max_count}; j <= max_count; ++j) {
        for (int64_t i{-max_count}; i <= max_count; ++i) {
          const vec3 vec{
              static_cast<double>(i),
              static_cast<double>(j),
              static_cast<double>(k),
          };
          if (norm(vec) >= kmin && norm(vec) <= kmax)
            vectors.push_back(vec);
        }
      }
    }
  }
  //   for (const auto &vec : vectors)
  //     PRINT_DEBUG(vec);
  // PRINT_DEBUG(vectors.size());
  ERRCHK_ALWAYS(vectors.size() == 350);

  return vectors.at(gen_random_uniform_int(0, vectors.size()));
}

/** Generates a random unit vector e in the half-open interval [-1, 1) not
 * parallel to k */
static auto gen_e(const vec3 &k) {
  vec3 e{gen_random_uniform_unit_vector()};

  // Regenerate e if parallel with k
  while (allclose(angle(k, e), 0) || allclose(angle(k, e), M_PI))
    e = gen_random_uniform_unit_vector();

  ERRCHK_ALWAYS(!allclose(norm(cross(k, e)), 0));
  ERRCHK_ALWAYS(allclose(norm(e), 1));
  return e;
}

/** Generate the real and imaginary parts of the forcing vector f_k */
static auto gen_f_k(const vec3 &k, const vec3 &e, const double relhel) {
  const auto f_k_denom{std::sqrt(1 + relhel * relhel) //
                       * dot(k, k)                    //
                       * std::sqrt(1 - (dot(k, e) * dot(k, e)) / dot(k, k))};
  const auto f_k_re{norm(k) * cross(k, e) / f_k_denom};
  const auto f_k_im{relhel * cross(k, cross(k, e)) / f_k_denom};
  const auto f_k_modulus{dot(f_k_re, f_k_re) + dot(f_k_im, f_k_im)};

  // PRINT_DEBUG(f_k_re);
  // PRINT_DEBUG(f_k_im);
  // PRINT_DEBUG(f_k_denom);
  // PRINT_DEBUG(f_k_modulus);

  ERRCHK_ALWAYS(allclose(f_k_modulus, 1));
  if (allclose(relhel, 1)) {
    ERRCHK_ALWAYS(allclose(-cross(k, f_k_im), norm(k) * f_k_re));
    ERRCHK_ALWAYS(allclose(cross(k, f_k_re), norm(k) * f_k_im));
  }

  return std::complex{f_k_re, f_k_im};
}

/** Generates a uniformly random phase within the half-open interval [-M_PI,
 * M_PI) */
static auto gen_phi() { return gen_random_uniform_real(-M_PI, M_PI); }

} // namespace ac::helical_forcing

static AcReal3 make_acreal3(const vec3& vec)
{
    return AcReal3{vec[0], vec[1], vec[2]};
}

ForcingParams generateHelicalForcingParams(const AcReal relhel, const AcReal magnitude, const AcReal kmin,
                                    const AcReal kmax)
{
    const auto k{ac::helical_forcing::gen_k(kmin, kmax)};
    const auto e{ac::helical_forcing::gen_e(k)};
    const auto f_k{ac::helical_forcing::gen_f_k(k, e, relhel)};
    const auto phi{ac::helical_forcing::gen_phi()};
    
    return ForcingParams{
        .magnitude = magnitude,
        .k_force = make_acreal3(k),
        .ff_hel_re = make_acreal3(f_k.real()),
        .ff_hel_im = make_acreal3(f_k.imag()),
        .phase = phi,
        .kaver = (kmax - kmin) / 2 + kmin,
    };
}