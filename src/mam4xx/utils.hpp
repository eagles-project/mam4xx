#ifndef MAM4XX_UTILS_HPP
#define MAM4XX_UTILS_HPP

#include <haero/math.hpp>

// This file contains utility-type functions that are available for use by
// various processes, tests, etc.

namespace mam4::utils {

using Real = haero::Real;
using haero::max;
using haero::min;

// this function considers 'num' and returns either 'num' (already in bounds) or
// 'high'/'low' if num is outside the bounds
KOKKOS_INLINE_FUNCTION
Real min_max_bound(const Real &low, const Real &high, const Real &num) {
  return max(low, min(high, num));
}

} // end namespace mam4::utils

#endif
