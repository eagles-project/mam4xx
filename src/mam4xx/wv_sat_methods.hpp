#ifndef MAM4XX_WV_SAT_METHODS_HPP
#define MAM4XX_WV_SAT_METHODS_HPP

namespace mam4 {

namespace wv_sat_methods {

KOKKOS_INLINE_FUNCTION
Real GoffGratch_svp_water(const Real temperature) {
  // Goff & Gratch (1946)
  // temperature in Kelvin

  // FROM wv_saturation.F90
  // Boiling point of water at 1 atm (K)
  // This value is slightly high, but it seems to be the value for the
  // steam point of water originally (and most frequently) used in the
  // Goff & Gratch scheme.
  const Real tboil = haero::Constants::boil_pt_h2o;

  const Real ten = 10;
  const Real one = 1;
  // got this from wikipedia article for Goff-Gratch eqn. and pulled it out of
  // the calculation for visibility's sake
  // https://en.wikipedia.org/wiki/Goff-Gratch_equation
  const Real svp_at_steam_pt_pressure = 1013.246; // BAD_CONSTANT!

  // uncertain below -70 C (NOTE: from mam4)
  return haero::pow(ten,
                    -Real(7.90298) * (tboil / temperature - one) +
                        Real(5.02808) * haero::log10(tboil / temperature) -
                        Real(1.3816e-7) *
                            (haero::pow(ten, Real(11.344) *
                                                 (one - temperature / tboil)) -
                             one) +
                        Real(8.1328e-3) *
                            (haero::pow(ten, -Real(3.49149) *
                                                 (tboil / temperature - one)) -
                             one) +
                        haero::log10(svp_at_steam_pt_pressure)) *
         ten * ten;

} // GoffGratch_svp_water

KOKKOS_INLINE_FUNCTION
Real GoffGratch_svp_ice(const Real temperature) {
  // temperature in Kelvin

  // good down to -100 C
  const Real h2otrip = haero::Constants::triple_pt_h2o;
  const Real ten = 10;
  const Real one = 1;
  // got this from wikipedia article for Goff-Gratch eqn. and pulled it out of
  // the calculation for visibility's sake
  // https://en.wikipedia.org/wiki/Goff-Gratch_equation
  const Real svp_at_ice_pt_pressure = 6.1071; // BAD_CONSTANT!

  return haero::pow(ten,
                    -Real(9.09718) * (h2otrip / temperature - one) -
                        Real(3.56654) * haero::log10(h2otrip / temperature) +
                        Real(0.876793) * (one - temperature / h2otrip) +
                        haero::log10(svp_at_ice_pt_pressure)) *
         ten * ten;

} // end GoffGratch_svp_ice

// FIXME
// Compute saturation vapor pressure over water
KOKKOS_INLINE_FUNCTION
Real svp_water(const Real temperature) {
  // FIXME
  // ask if we need to implement the other methods to compute svp_water
  // initial_default_idx = GoffGratch_idx
  return GoffGratch_svp_water(temperature);
}

/*---------------------------------------------------------------------
! UTILITIES
!---------------------------------------------------------------------*/

// Get saturation specific humidity given pressure and SVP.
// Specific humidity is limited to range 0-1.
KOKKOS_INLINE_FUNCTION
Real wv_sat_svp_to_qsat(const Real es, const Real p) {
  // es  SVP
  // p   Current pressure.
  // If pressure is less than SVP, set qs to maximum of 1.
  // epsilo  ! Ice-water transition range
  // omeps   ! 1._r8 - epsilo
  // epsilo       = shr_const_mwwv/shr_const_mwdair   ! ratio of h2o to dry air
  // molecular weights real(R8),parameter :: SHR_CONST_MWDAIR  = 28.966_R8 !
  // molecular weight dry air ~ kg/kmole real(R8),parameter :: SHR_CONST_MWWV
  // = 18.016_R8       ! molecular weight water vapor
  // const Real SHR_CONST_MWWV = 18.016;
  // const Real SHR_CONST_MWDAIR = 28.966;
  const Real epsilo = haero::Constants::weight_ratio_h2o_air;

  const Real zero = 0;
  const Real one = 1;
  const Real omeps = one - epsilo;
  Real qs = zero;
  if ((p - es) <= zero) {
    qs = one;
  } else {
    qs = epsilo * es / (p - omeps * es);
  }
  return qs;

} // wv_sat_svp_to_qsat

KOKKOS_INLINE_FUNCTION
void wv_sat_qsat_water(const Real t, const Real p, Real &es, Real &qs) {
  /*------------------------------------------------------------------!
  ! Purpose:                                                         !
  !   Calculate SVP over water at a given temperature, and then      !
  !   calculate and return saturation specific humidity.             !
  !------------------------------------------------------------------*/
  // Inputs
  // t    temperature
  // p    Pressure
  // Outputs
  // es  Saturation vapor pressure
  // qs  Saturation specific humidity

  es = svp_water(t);
  qs = wv_sat_svp_to_qsat(es, p);
  // Ensures returned es is consistent with limiters on qs.
  es = haero::min(es, p);

} // wv_sat_qsat_water

KOKKOS_INLINE_FUNCTION
Real svp_ice(const Real temperature) { return GoffGratch_svp_ice(temperature); }

} // namespace wv_sat_methods
} // namespace mam4

#endif
