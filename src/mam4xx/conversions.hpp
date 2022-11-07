#ifndef MAM4_CONVERSIONS_HPP
#define MAM4_CONVERSIONS_HPP

#include <ekat/ekat_pack.hpp>
#include <ekat/ekat_pack_math.hpp>
#include <haero/constants.hpp>
#include <haero/math.hpp>

/// This file contains functions for converting between various representations
/// of physical quantities in aerosol parameterizations.

namespace mam4::conversions {

using Real = haero::Real;
using Constants = haero::Constants;

/// Given a number concentration for a species or mixture [m-3], computes and
/// returns a mass mixing ratio [kg species/kg dry air] based on its molecular
/// weight and on the density of dry air in the vicinity.
/// @param [in] number_conc The number concentration of the species/mixture
/// [m-3]
/// @param [in] molecular_wt The molecular weight of the species/mixture
/// [kg/kmol]
/// @param [in] dry_air_density The mass density of dry air [kg/m3]
template <typename Scalar>
KOKKOS_INLINE_FUNCTION Scalar
mmr_from_number_conc(const Scalar &number_conc, Real molecular_wt,
                     const Scalar &dry_air_density) {
  const auto Na = Constants::avogadro;
  return number_conc * molecular_wt / (dry_air_density * Na);
}

/// Given a mass mixing ratio (mmr) for a species or mixture [kg species/kg
/// dry air], computes and returns a number density [m-3] based on its molecular
/// weight and on the density of dry air in the vicinity.
/// @param [in] mmr The number concentration of the species/mixture [m-3]
/// @param [in] molecular_wt The molecular weight of the species/mixture
/// [kg/kmol]
/// @param [in] dry_air_density The mass density of dry air [kg/m3]
template <typename Scalar>
KOKKOS_INLINE_FUNCTION Scalar number_conc_from_mmr(
    const Scalar &mmr, Real molecular_wt, const Scalar &dry_air_density) {
  const auto Na = Constants::avogadro;
  return mmr * (dry_air_density * Na) / molecular_wt;
}

/// Given a molar mixing ratio (vmr) for a species or mixture
/// [kmol species/kmol dry air], computes and returns a mass mixing ratio
/// [kg species/kg dry air] based on its molecular weight.
/// @param [in] vmr The molar mixing ratio of the species/mixture [kmol/kmol
/// air]
/// @param [in] molecular_wt The molecular weight of the species/mixture
/// [kg/kmol]
template <typename Scalar>
KOKKOS_INLINE_FUNCTION Scalar mmr_from_vmr(const Scalar &vmr,
                                           Real molecular_wt) {
  const auto mw_dry_air = Constants::molec_weight_dry_air;
  return vmr * molecular_wt / mw_dry_air;
}

/// Given a mass mixing ratio (mmr) for a species or mixture [kg species/kg
/// dry air], computes and returns a molar mixing ratio [kmol species/k dry air]
/// based on its molecular weight.
/// @param [in] mmr The molar mixing ratio of the species/mixture [kmol/kmol
/// dry air]
/// @param [in] molecular_wt The molecular weight of the species/mixture
/// [kg/kmol]
template <typename Scalar>
KOKKOS_INLINE_FUNCTION Scalar vmr_from_mmr(const Scalar &mmr,
                                           Real molecular_wt) {
  const auto mw_dry_air = Constants::molec_weight_dry_air;
  return mmr * mw_dry_air / molecular_wt;
}

/// Computes the virtual temperature [K] from the temperature [K] and a water
/// vapor mass mixing ratio [kg vapor/kg dry air]. See Equation (3.1.15) from
/// Gill, 1982, Atmosphere-Ocean Dynamics, Academic Press, San Diego CA.
/// @param [in] Tv virtual temperature [K]
/// @param [in] q1 specific humidity [-]
template <typename Scalar>
KOKKOS_INLINE_FUNCTION Scalar
virtual_temperature_from_temperature(const Scalar &T, const Scalar &q1) {
  return T * (1.0 + 0.6078 * q1);
}

/// Computes the temperature [K] from the virtual temperature [K] and a water
/// vapor mass mixing ratio [kg vapor/kg dry air].
/// @param [in] Tv virtual temperature [K]
/// @param [in] q1 specific humidity [-]
template <typename Scalar>
KOKKOS_INLINE_FUNCTION Scalar
temperature_from_virtual_temperature(const Scalar &Tv, const Scalar &q1) {
  return Tv / (1.0 + 0.6078 * q1);
}

/// Computes the dry mass density from the total mass density using the specific
/// humidity.
/// @param [in] rho total mass density [kg/m3]
/// @param [in] q1 specific humidity [-]
template <typename Scalar>
KOKKOS_INLINE_FUNCTION Scalar dry_from_total_mass_density(const Scalar &rho,
                                                          const Scalar &q1) {
  return rho * (1 - q1);
}

/// Computes the mass density of water vapor from the total mass density using
/// the specific humidity.
/// @param [in] rho total mass density [kg/m3]
/// @param [in] q1 specific humidity [-]
template <typename Scalar>
KOKKOS_INLINE_FUNCTION Scalar vapor_from_total_mass_density(const Scalar &rho,
                                                            const Scalar &q1) {
  return rho * q1;
}

/// Computes the water vapor mixing ratio from the specific humidity. This
/// calculation diverges at q1 = 1.
/// @param [in] q1 specific humidity [-]
template <typename Scalar>
KOKKOS_INLINE_FUNCTION Scalar
vapor_mixing_ratio_from_specific_humidity(const Scalar &q1) {
  return q1 / (1 - q1);
}

/// Computes the specific humidity from the water vapor mixing ratio. This
/// calculation diverges at q1 = 1.
/// @param [in] qv water vapor mixing ratio [kg vapor / kg dry air]
template <typename Scalar>
KOKKOS_INLINE_FUNCTION Scalar
specific_humidity_from_vapor_mixing_ratio(const Scalar &qv) {
  return qv / (qv + 1);
}

/// Computes the saturation vapor pressure of water as a function
/// of temperature.
///
/// Note that this formula computes saturation vapor pressure
/// over a planar surface of pure water; it does not account
/// for surrounding air and does not take environmental pressure into account.
///
/// The formula is the improved Magnus formula from
///
///  O. A. Alduchov and R. E. Eskridge, 1996, Improved Magnus form approximation
///  of saturation vapor pressure, Journal of Applied Meteorology 35:601--609.
///
///  See eqs. (21) and (22) from that paper, which improves the original
///  formula's accuracy over the temperature range [-40, 50] C expected of
///  atmospheric conditions throughout an entire vertical column.
///  The paper uses temperature in Celsius and returns
///  pressure in hPa. We have changed to SI units in this implementation.
///
///  See also Lamb & Verlinde section 3.3.
///
///  @param [in] T temperature [K]
///  @param [in] P pressure [Pa]
///  @return es(T) saturation vapor pressure of water vapor [Pa]
template <typename Scalar>
KOKKOS_INLINE_FUNCTION Scalar
vapor_saturation_pressure_magnus_ew(const Scalar &T) {
  static constexpr Real e0 = 610.94;      // Pa
  static constexpr Real exp_num = 17.625; // nondimensional
  static constexpr Real exp_den = 234.04; // deg C
  const auto Tf = Constants::freezing_pt_h2o;
  const auto celsius_temp = T - Tf;
  return e0 * exp(exp_num * celsius_temp / (exp_den + celsius_temp));
}

/// Computes the saturation vapor pressure of water as a function
/// of temperature and pressure for moist air
/// above a planar surface of water.
///
/// The formula is the improved Magnus formula from
///
///  O. A. Alduchov and R. E. Eskridge, 1996, Improved Magnus form approximation
///  of saturation vapor pressure, Journal of Applied Meteorology 35:601--609.
///
///  See eq. (22) from that paper, which improves the original formula's
///  accuracy over the temperature range [-40, 50] C expected of
///  atmospheric conditions throughout an entire vertical column.
///  The paper uses temperature in Celsius and returns
///  pressure in hPa. We have changed to SI units in this implementation.
///
///  See also Lamb & Verlinde section 3.3.
///
///  @param [in] T temperature [K]
///  @param [in] P pressure [Pa]
///  @return es(T) saturation vapor pressure of water vapor [Pa]
template <typename Scalar>
KOKKOS_INLINE_FUNCTION Scalar
vapor_saturation_pressure_magnus(const Scalar &T, const Scalar &P) {
  const auto ew = vapor_saturation_pressure_magnus_ew(T);
  return 1.00071 * exp(4.5e-8 * P) * ew;
}

///  Saturation vapor pressure, Hardy formula
///
///  @param [in] T [K]
///  @return es [Pa]
///
///  This formula is used by NCAR for atmospheric sounding data, and
///  is included in their
///  [ASPEN](https://ncar.github.io/aspendocs/form_wsat.html#formula) data
///  processing code.
///
///    B. Hardy, 1998, ITS-90 Formulations for Vapor Pressure, Frostpoint
///   Temperature, Dewpoint Temperature, and Enhancement Factors in the Range
///   –100 to +100 C, Proceedings of the Third International Symposium on
///   Humidity & Moisture, Teddington, London, England, April 1998.
///
template <typename Scalar>
KOKKOS_INLINE_FUNCTION Scalar vapor_saturation_pressure_hardy(const Scalar &T) {
  const Real g[8] = {-2.8365744e3,    -6.028076559e3, 1.954263612e1,
                     -2.737830188e-2, 1.6261698e-5,   7.0229056e-10,
                     -1.8680009e-13,  2.7150305};
  Scalar log_es(0);
  for (int i = 0; i < 7; ++i) {
    log_es += g[i] * pow(T, i - 2);
  }
  log_es += g[7] * log(T);
  return exp(log_es);
}

///  Saturation mixing ratio of water vapor
///
///
///  // FIXME: these humidity functions need to be reconciled with EAMXX
///
///  See NCAR's software for atmospheric sounding data,
///  [ASPEN](https://ncar.github.io/aspendocs/form_wsat.html#formula)
///
///  @param [in] T temperature of moist air [K]
///  @param [in] P pressure of moist air [Pa]
///  @return saturation mixing ratio of water vapor [kg h2o / kg dry air]
///
template <typename Scalar>
KOKKOS_INLINE_FUNCTION Scalar saturation_mixing_ratio_hardy(const Scalar &T,
                                                            const Scalar &P) {
  const Real eps_h2o =
      Constants::molec_weight_h2o / Constants::molec_weight_dry_air;
  const auto es = vapor_saturation_pressure_hardy(T);
  return eps_h2o * es / (P - es);
}

/// Computes the relative humidity from the water vapor mixing ratio and the
/// pressure and temperature, given the relationship between temperature and
/// the water vapor saturation pressure.
///
/// Use this formula with parameterizations that are defined with respect to
/// air air (and note that mixing ratio is defined with respect to dry air).
///
/// @param [in] w water vapor mixing ratio [kg vapor / kg dry air]
/// @param [in] p total pressure [Pa]
/// @param [in] T temperature [K]
/// @param [in] wsat A function that computes the saturation mixing ratio from
///                 the temperature. If not supplied,
///                 @ref saturation_mixing_ratio_hardy is used.
/// @return relative humidity [1]
template <typename Scalar>
KOKKOS_INLINE_FUNCTION Scalar relative_humidity_from_vapor_mixing_ratio(
    const Scalar &w, const Scalar &T, const Scalar &p,
    Scalar (*wsat)(const Scalar &,
                   const Scalar &) = saturation_mixing_ratio_hardy<Scalar>) {
  const auto ws = wsat(T, p);
  return w / ws;
}

///  Computes the relative humidity from specific humidity,
/// temperature, and pressure.
///
/// Use this formula with parameterizations defined with respect to moist air,
/// since EAM's dynamical core uses moist air, this formula should be the
/// default.
///
/// @param [in] q specific humidity [density vapor / density moist air]
/// @param [in] T temperature [K]
/// @param [in] P total pressure [Pa]
/// @return relative humidity [1]
template <typename Scalar>
KOKKOS_INLINE_FUNCTION Scalar relative_humidity_from_specific_humidity(
    const Scalar &q, const Scalar &T, const Scalar &P) {
  const auto w = vapor_mixing_ratio_from_specific_humidity(q);
  const auto ws = saturation_mixing_ratio_hardy(T, P);
  return w / ws;
}

/// Computes the water vapor mixing ratio from the relative humidity and the
/// pressure and temperature, given the relationship between temperature and
/// the water vapor saturation pressure.
/// @param [in] rel_hum relative humidity [-]
/// @param [in] p total pressure [Pa]
/// @param [in] T temperature [K]
/// @param [in] ws A function that computes the saturation mixing ratio from
///                 the temperature and pressure. If not supplied,
///                 @ref saturation_mixing_ratio_hardy is used.
/// @return saturation mixing ratio [kg h2o / kg dry air]
template <typename Scalar>
KOKKOS_INLINE_FUNCTION Scalar vapor_mixing_ratio_from_relative_humidity(
    const Scalar &rel_hum, const Scalar &p, const Scalar &T,
    Scalar (*wsat)(const Scalar &,
                   const Scalar &) = saturation_mixing_ratio_hardy<Scalar>) {
  const auto ws = wsat(T, p);
  return rel_hum * ws;
}

///   This function returns the modal geometric mean particle diameter,
/// given the mode's mean volume (~ to 3rd log-normal moment) and the modal
/// standard deviation.
///
/// @param mode_mean_particle_volume mean particle volume for mode [m^3 per
/// particle]
/// @return modal mean particle diameter [m per particle]
template <typename Scalar>
KOKKOS_INLINE_FUNCTION Scalar mean_particle_diameter_from_volume(
    const Scalar mode_mean_particle_volume, const Real mean_std_dev) {
  const double pio6 = Constants::pi_sixth;
  return cbrt(mode_mean_particle_volume / pio6) *
         exp(-1.5 * ekat::square<Scalar>(log(mean_std_dev)));
}

///   This function is the inverse of
///   modal_mean_particle_diameter_from_volume; given the modal mean geometric
///   diamaeter, it returns the corresponding volume.
///
///   @param [in] geom_diam geometric mean diameter [m per particle]
///   @return mean volume [m^3 per particle]
template <typename Scalar>
KOKKOS_INLINE_FUNCTION Scalar mean_particle_volume_from_diameter(
    const Scalar geom_diam, const Real mean_std_dev) {
  const double pio6 = Constants::pi_sixth;
  return cube(geom_diam) * exp(4.5 * ekat::square<Scalar>(log(mean_std_dev))) *
         pio6;
}

} // namespace mam4::conversions

#endif
