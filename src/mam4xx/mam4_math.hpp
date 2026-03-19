#ifndef MAM4XX_MATH_HPP
#define MAM4XX_MATH_HPP

#include "mam4_config.hpp"

#include <Kokkos_MathematicalFunctions.hpp>
#include <Kokkos_MinMax.hpp>
#include <Kokkos_NumericTraits.hpp>

namespace mam4 {

using Kokkos::atan;
using Kokkos::cbrt;
using Kokkos::cos;
using Kokkos::erf;
using Kokkos::erfc;
using Kokkos::exp;
using Kokkos::isnan;
using Kokkos::nan;
using Kokkos::log;
using Kokkos::log10;
using Kokkos::max;
using Kokkos::min;
using Kokkos::pow;
using Kokkos::round;
using Kokkos::sin;
using Kokkos::sqrt;

KOKKOS_INLINE_FUNCTION Real max() { return Kokkos::Experimental::finite_max_v<Real>; }
KOKKOS_INLINE_FUNCTION Real max(Real x, Real y) { return Kokkos::max(x, y); }
KOKKOS_INLINE_FUNCTION Real min(Real x, Real y) { return Kokkos::min(x, y); }

KOKKOS_INLINE_FUNCTION Real square(const Real x) { return x * x; }
KOKKOS_INLINE_FUNCTION Real cube(const Real x) { return x * x * x; }

KOKKOS_INLINE_FUNCTION constexpr Real epsilon() {
  return Kokkos::Experimental::epsilon_v<Real>;
}

} // namespace mam4

#endif
