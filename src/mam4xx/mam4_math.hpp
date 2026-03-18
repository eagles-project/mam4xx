#ifndef MAM4XX_MATH_HPP
#define MAM4XX_MATH_HPP

#include "mam4_config.hpp"

#include <Kokkos_MathematicalFunctions.hpp>
#include <Kokkos_MinMax.hpp>
#include <Kokkos_NumericTraits.hpp>

namespace mam4 {

using namespace Kokkos; // bring in Kokkos math functions

inline Real max() { return Kokkos::Experimental::finite_max_v<Real>; }
inline Real min(const Real x, const Real y) { return (x < y) ? x : y; }
inline Real max(const Real x, const Real y) { return (x > y) ? x : y; }

inline Real square(const Real x) { return x * x; }
inline Real cube(const Real x) { return x * x * x; }

inline constexpr Real epsilon() {
  return Kokkos::Experimental::epsilon_v<Real>;
}

} // namespace mam4

#endif
