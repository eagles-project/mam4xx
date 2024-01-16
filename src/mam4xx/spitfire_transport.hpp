// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MAM4XX_SPITFIRE_HPP
#define MAM4XX_SPITFIRE_HPP

#include <haero/math.hpp>
#include <mam4xx/mam4_types.hpp>

namespace mam4::spitfire {

//##############################################################################
// The minmod function
//##############################################################################
KOKKOS_INLINE_FUNCTION
Real minmod(const Real aa, const Real bb) {
  return 0.5 * (Kokkos::copysign(1.0, aa) + Kokkos::copysign(1.0, bb)) *
         haero::min(haero::abs(aa), haero::abs(bb));
}

//##############################################################################
// The medan function
//##############################################################################
KOKKOS_INLINE_FUNCTION
Real median(const Real aa, const Real bb, const Real cc) {
  return aa + minmod(bb - aa, cc - aa);
}

KOKKOS_INLINE_FUNCTION
void get_flux(const haero::ThreadTeamPolicy &team_policy, ColumnView &xw,
              ColumnView &phi, ColumnView &vel, const Real deltat,
              ColumnView &flux) {}

} // namespace mam4::spitfire
#endif
