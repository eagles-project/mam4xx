// Copyright (c) 2021, National Technology & Engineering Solutions of Sandia,
// LLC (NTESS). Copyright (c) 2022, Battelle Memorial Institute
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MAM4XX_FLOATING_POINT_UTILS_HPP
#define MAM4XX_FLOATING_POINT_UTILS_HPP

#include "mam4_config.hpp"

#include <Kokkos_Core.hpp>
#include <ekat_kernel_assert.hpp>

namespace mam4 {

using std::abs;

/**
  struct for help with common floating point operations

*/
template <typename T = Real> struct FloatingPoint {
  static_assert(std::is_floating_point<T>::value,
                "floating point type required.");

  /// Default tolerance for floating point comparisons
  static constexpr T zero_tol = std::numeric_limits<Real>::epsilon();

  /// Define floating point zero by @f$\lvert x \rvert < \epsilon_{tol}@f$
  KOKKOS_INLINE_FUNCTION
  static bool zero(const T x, const T tol = zero_tol) {
    EKAT_KERNEL_ASSERT(tol > 0);
    return std::abs(x) < tol;
  }

  /// Define floating point equivalence by @f$\lvert x_0 - x_1 \rvert <
  /// \epsilon_{tol}@f$
  KOKKOS_INLINE_FUNCTION
  static bool equiv(const T x0, const T x1, const T tol = zero_tol) {
    EKAT_KERNEL_ASSERT(tol > 0);
    return std::abs(x0 - x1) < tol;
  }

  /// Define floating point equivalence by
  /// @f$\frac{\lvert x_0 - x_1 \rvert}{max(\lvert x_0 \rvert, \lvert x_1
  /// \rvert)} < \epsilon_{tol}@f$
  KOKKOS_INLINE_FUNCTION
  static bool rel(const T x0, const T x1, const T tol = zero_tol) {
    EKAT_KERNEL_ASSERT(tol > 0);
    const T max = std::abs(x0) < std::abs(x1) ? std::abs(x1) : std::abs(x0);
    return max ? std::abs(x0 - x1) / max < tol : true;
  }

  /** Define floating point in bounds as @f$ l - \epsilon_{tol} < x < u +
    \epsilon_{tol}@f$

    @param [in] x
    @param [in] lower lower bound @f$l@f$
    @param [in] upper upper bound @f$u@f$
    @param [in] tol tolerance @f$\epsilon_{tol}@f$.
  */
  KOKKOS_INLINE_FUNCTION
  static bool in_bounds(const T x, const T lower, const T upper,
                        const T tol = zero_tol) {
    EKAT_KERNEL_ASSERT(tol > 0);
    return (x >= (lower - tol) && x <= (upper + tol));
  }

  /** multiplier for safe division by x, @f$ \frac{1}{x} \approx \frac{x}{x^2 +
    \epsilon_{tol}^2}@f$

    For use with removable singularities.

    @warning the return value is a multiplication factor, not a divisor

    @param [in] x
    @param [in] tol
    @return @f$\frac{1}{x} \approx \frac{x}{x^2 + \epsilon_{tol}^2@f$
  */
  KOKKOS_INLINE_FUNCTION
  static T safe_denominator(const T x, const T tol = zero_tol) {
    EKAT_KERNEL_ASSERT(tol > 0);
    return x / (x * x + tol * tol);
  }
};

} // namespace mam4
#endif
