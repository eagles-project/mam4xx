// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PHYSICAL_LIMITS_HPP
#define PHYSICAL_LIMITS_HPP

namespace mam4 {
KOKKOS_INLINE_FUNCTION
constexpr Real interstitial_aerosol_number_min() { return 0; }
KOKKOS_INLINE_FUNCTION
constexpr Real interstitial_aerosol_number_max() { return 1e13; }
KOKKOS_INLINE_FUNCTION
void check_valid_interstitial_aerosol_number(const Real aero_num) {
  EKAT_KERNEL_REQUIRE_MSG(interstitial_aerosol_number_min() <= aero_num &&
                              aero_num < interstitial_aerosol_number_max(),
                          "Computed total interstitial aerosol number outside "
                          "of the resonable baounds of 0 to 1e13.");
}
} // namespace mam4

#endif
