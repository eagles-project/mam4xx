// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MAM4XX_ATMOSPHERE_INIT_HPP
#define MAM4XX_ATMOSPHERE_INIT_HPP

#include "mam4xx/mam4_types.hpp"

#include <ekat/ekat_pack.hpp>
#include <ekat/ekat_pack_math.hpp>
#include <haero/atmosphere.hpp>
#include <haero/constants.hpp>
#include <haero/floating_point.hpp>
#include <haero/haero.hpp>

namespace mam4 {

/** @brief Specific humidity profile, @f$ q_v(z) @f$

  @param [in] z height [m]
  @param [in] qv0 mixing ratio at surface [kg H<sub>2</sub>O / kg moist air]
  @param [in] qv1 decay rate of water vapor [1/m]
  @return @f$ q_v @f$
*/
template <typename Scalar>
KOKKOS_INLINE_FUNCTION Scalar init_specific_humidity(const Scalar z,
                                                     const Real qv0,
                                                     const Real qv1) {
  Scalar result = qv0 * exp(-qv1 * z);

  return result;
}

/** @brief Linear virtual temperature profile, @f$T_v(z)@f$.

  @param [in] z height [m]
  @param [in] T0 reference virtual temperature [K]
  @param [in] Gamma virtual temperature lapse rate [K/m]
  @return virtual temperature
*/
template <typename Scalar>
KOKKOS_INLINE_FUNCTION Scalar init_virtual_temperature(const Scalar z,
                                                       const Real Tv0,
                                                       const Real Gammav) {
  return Tv0 - Gammav * z;
}

/** @brief Computes the hydrostatic pressure at a given height, based on a
  virtual temperature profile with constant lapse rate @f$ \Gamma =
  -\frac{\partial T_v}{\partial z} \ge 0 @f$.

  @param [in] z height [m]
  @param [in] p0 reference presssure [Pa]
  @param [in] T0 reference temperature [K]
  @param [in] Gamma temperature lapse rate [K/m]
  @return p [Pa]
*/
template <typename Scalar>
KOKKOS_INLINE_FUNCTION Scalar hydrostatic_pressure_at_height(const Scalar z,
                                                             const Real p0,
                                                             const Real T0,
                                                             const Real Gamma) {
  Scalar result = 0;
  if (FloatingPoint<Real>::zero(Gamma)) {
    result =
        p0 * exp(-Constants::gravity * z / (Constants::r_gas_dry_air * T0));
  } else {
    const Real pwr = Constants::gravity / (Constants::r_gas_dry_air * Gamma);
    result = p0 * pow(T0, -pwr) * pow(T0 - Gamma * z, pwr);
  }
  return result;
}

/** @brief Computes the height based on hydrostatic pressure.

    @param [in] p pressure [Pa]
    @param [in] p0 reference pressure [Pa]
    @param [in] T0 reference virtual temperature
    @param [in] Gamma virtual temperature lapse rate, @f$\Gamma =
   -\frac{\partial T_v}{\partial z} \ge 0 @f$
    @return height [m]
*/
template <typename Scalar>
KOKKOS_INLINE_FUNCTION Scalar height_at_hydrostatic_pressure(const Scalar p,
                                                             const Real p0,
                                                             const Real T0,
                                                             const Real Gamma) {
  Scalar result = 0;
  if (FloatingPoint<Real>::zero(Gamma)) {
    result = -Constants::r_gas_dry_air * T0 * log(p / p0) / Constants::gravity;
  } else {
    const Real pwr = Constants::r_gas_dry_air * Gamma / Constants::gravity;
    result = (T0 / Gamma) * (1 - pow(p / p0, pwr));
  }
  return result;
}

/** @brief Initializes a hydrostatic atmosphere column with linear
  virtual temperature profile.

  Levels are spaced uniformly in height, water vapor decays exponentially
  with height, and virtual temperature decays linearly with height.

  @param [in/out] atm Atmosphere column whose data need initialization
  @param [in] Tv0 Reference virtual temperature (surface) [K]
  @param [in] Gammav virtual temperature lapse rate [K/m]
  @param [in] qv0 mixing ratio at surface [kg H<sub>2</sub>O / kg air]
  @param [in] qv1 decay rate of water vapor [1/m]
*/
void init_atm_const_tv_lapse_rate(const Atmosphere &atm, const Real Tv0 = 300,
                                  const Real Gammav = 0.01,
                                  const Real qv0 = 0.005,
                                  const Real qv1 = 1e-5);

} // namespace mam4
#endif
