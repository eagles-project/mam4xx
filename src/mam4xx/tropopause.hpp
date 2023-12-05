#ifndef MAM4XX_TROPOPAUSE_HPP
#define MAM4XX_TROPOPAUSE_HPP

#include <haero/math.hpp>
#include <mam4xx/aero_config.hpp>

#include <fstream>
#include <iostream>

namespace mam4 {

namespace tropopause {

using ConstColumnView = haero::ConstColumnView;
// From radconstants //FIXME: BAD CONSTANT
constexpr int nswbands = 14;
constexpr int nlwbands = 16;
using View2D = DeviceType::view_2d<Real>;
constexpr Real km_inv_to_m_inv = 0.001; // 1/km to 1/m

constexpr Real shr_const_rgas =
    haero::Constants::r_gas * 1e3; // Universal gas constant ~ J/K/kmole
constexpr Real shr_const_mwdair = haero::Constants::molec_weight_dry_air *
                                  1e3; // molecular weight dry air ~ kg/kmole
constexpr Real shr_const_cpdair =
    haero::Constants::cp_dry_air; // specific heat of dry air [J/K/kmole]

constexpr Real cnst_kap =
    (shr_const_rgas / shr_const_mwdair) / shr_const_cpdair; //  ! R/Cp

constexpr Real cnst_faktor =
    -haero::Constants::gravity /
    haero::Constants::r_gas_dry_air; // acceleration of gravity ~ m/s^2/Dry air
                                     // gas constant     ~ J/K/kg
constexpr Real cnst_ka1 = cnst_kap - 1.0;

KOKKOS_INLINE_FUNCTION
void get_dtdz(const Real pm, const Real pmk, const Real pmid1d_up,
              const Real pmid1d_down, const Real temp1d_up,
              const Real temp1d_down, Real &dtdz, Real &tm) {

  // pm      ! mean pressure [Pa]
  // pmk
  // pmid1d_up     !  midpoint pressure in column at upper level [Pa]
  // pmid1d_down   !  midpoint pressure in column at lower level [Pa]
  // temp1d_up     !  temperature in column at upper level [K]
  // temp1d_down   !  temperature in column at lower level [K]
  // dtdz     ! temperature lapse rate vs. height [K/m]
  // tm       ! mean temperature [K] -- needed to find pressure at trop + 2 km

  const Real a1 =
      (temp1d_up - temp1d_down) /
      (haero::pow(pmid1d_up, cnst_kap) - haero::pow(pmid1d_down, cnst_kap));
  const Real b1 = temp1d_down - a1 * haero::pow(pmid1d_down, cnst_kap);
  tm = a1 * pmk + b1;
  const Real dtdp = a1 * cnst_kap * (haero::pow(pm, cnst_ka1));
  dtdz = cnst_faktor * dtdp * pm / tm;

} // get_dtdz

} // namespace tropopause
} // end namespace mam4

#endif