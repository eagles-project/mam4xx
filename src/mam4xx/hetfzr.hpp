// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MAM4XX_HETFZR_HPP
#define MAM4XX_HETFZR_HPP

#include <haero/atmosphere.hpp>
#include <haero/math.hpp>

#include <mam4xx/aero_config.hpp>
#include <mam4xx/conversions.hpp>
#include <mam4xx/mam4_types.hpp>
#include <mam4xx/utils.hpp>

namespace mam4 {

namespace hetfzr {

KOKKOS_INLINE_FUNCTION
Real get_air_viscosity(const Real tc) {
  // tc temperature [deg C]
  // dynamic viscosity of air [kg/m/s]

  const Real coeff_a = -1.2e-5;
  const Real coeff_b = 0.0049;
  const Real coeff_c = 1.718;

  return (coeff_c + coeff_b * tc + coeff_a * tc * tc) * 1.e-5;
}

KOKKOS_INLINE_FUNCTION
Real get_latent_heat_vapor(const Real tc) {
  // tc temperature [deg C]
  // latent heat of vaporization [J/kg]

  const Real coeff_a = -0.0000614342;
  const Real coeff_b = 0.00158927;
  const Real coeff_c = -2.36418;
  const Real coeff_d = 2500.79;

  const Real latvap =
      1000 * (coeff_a * haero::cube(tc) + coeff_b * haero::square(tc) +
              coeff_c * tc + coeff_d);

  return latvap;
}

KOKKOS_INLINE_FUNCTION
Real get_reynolds_num(const Real r3lx, const Real rho_air,
                      const Real viscos_air) {

  const Real coeff_vlc_a = 8.8462e2;
  const Real coeff_vlc_b = 9.7593e7;
  const Real coeff_vlc_c = -3.4249e-11;
  const Real coeff_adj_a = 3.1250e-1;
  const Real coeff_adj_b = 1.0552e-3;
  const Real coeff_adj_c = -2.4023;

  // droplet terminal velocity after Chen & Liu, QJRMS 2004
  const Real vlc_drop_adjfunc =
      (haero::exp(coeff_adj_a + coeff_adj_b * haero::cube(haero::log(r3lx)) +
                  coeff_adj_c * haero::pow(rho_air, 1.5)));
  const Real vlc_drop =
      (coeff_vlc_a + (coeff_vlc_b + coeff_vlc_c * r3lx) * r3lx) * r3lx *
      vlc_drop_adjfunc;

  // Reynolds number
  const Real Re = 2 * vlc_drop * r3lx * rho_air / viscos_air;

  return Re;
}

KOKKOS_INLINE_FUNCTION
Real get_temperature_difference(const Real temperature, const Real pressure,
                                const Real eswtr, const Real latvap,
                                const Real Ktherm_air) {

   // water vapor diffusivity: Pruppacher & Klett 13-3
   const Real Dvap = 0.211e-4*(temperature/273.15)*(101325./pressure);

   // G-factor = rhoh2o*Xi in Rogers & Yau, p. 104
   const Real G_factor = rhoh2o/((latvap/(rh2o*temperature) -1)*latvap*rhoh2o/(Ktherm_air*temperature) &
            + rhoh2o*rh2o*temperature/(Dvap*eswtr));


  // calculate T-Tc as in Cotton et al.
   const Real Tdiff = -G_factor*(rhwincloud - 1.)*latvap/Ktherm_air;

   return Tdiff;
}

} // namespace hetfzr
} // namespace mam4

#endif