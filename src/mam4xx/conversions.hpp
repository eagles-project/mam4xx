// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MAM4_CONVERSIONS_HPP
#define MAM4_CONVERSIONS_HPP

#include <haero/constants.hpp>
#include <haero/math.hpp>

/// This file contains functions for converting between various representations
/// of physical quantities in aerosol parameterizations.

namespace mam4::conversions {

using Real = haero::Real;
using Constants = haero::Constants;
using haero::cube;
using haero::square;

/// Given a number concentration for a species or mixture [m-3], computes and
/// returns a mass mixing ratio [kg species/kg dry air] based on its molecular
/// weight and on the density of dry air in the vicinity.
/// @param [in] number_conc The number concentration of the species/mixture
/// [m-3]
/// @param [in] molecular_wt The molecular weight of the species/mixture
/// [kg/kmol]
/// @param [in] dry_air_density The mass density of dry air [kg/m3]
KOKKOS_INLINE_FUNCTION Real mmr_from_number_conc(Real number_conc,
                                                 Real molecular_wt,
                                                 Real dry_air_density) {
  const auto Na = Constants::avogadro;
  return number_conc * molecular_wt / (dry_air_density * Na);
}

/// Given a mass mixing ratio (mmr) for a species or mixture [kg species/kg
/// dry air], computes and returns a number density [m-3] based on its molecular
/// weight and on the density of dry air in the vicinity.
/// @param [in] mmr The mass mixing ratio  of the species/mixture [kg/kg dry
/// air]
/// @param [in] molecular_wt The molecular weight of the species/mixture
/// [kg/kmol]
/// @param [in] dry_air_density The mass density of dry air [kg/m3]
KOKKOS_INLINE_FUNCTION Real number_conc_from_mmr(Real mmr, Real molecular_wt,
                                                 Real dry_air_density) {
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
KOKKOS_INLINE_FUNCTION Real mmr_from_vmr(Real vmr, Real molecular_wt) {
  const auto mw_dry_air = Constants::molec_weight_dry_air;
  return vmr * molecular_wt / mw_dry_air;
}

/// Given a mass mixing ratio (mmr) for a species or mixture [kg species/kg
/// dry air], computes and returns a molar mixing ratio [kmol species/k dry air]
/// based on its molecular weight.
/// @param [in] mmr The mass mixing ratio of the species/mixture [kg/kg dry air]
/// @param [in] molecular_wt The molecular weight of the species/mixture
/// [kg/kmol]
KOKKOS_INLINE_FUNCTION Real vmr_from_mmr(Real mmr, Real molecular_wt) {
  const auto mw_dry_air = Constants::molec_weight_dry_air;
  return mmr * mw_dry_air / molecular_wt;
}

/// Computes the virtual temperature [K] from the temperature [K] and a water
/// vapor mass mixing ratio [kg vapor/kg dry air]. See Equation (3.1.15) from
/// Gill, 1982, Atmosphere-Ocean Dynamics, Academic Press, San Diego CA.
/// @param [in] Tv virtual temperature [K]
/// @param [in] q1 specific humidity [-]
KOKKOS_INLINE_FUNCTION Real virtual_temperature_from_temperature(Real T,
                                                                 Real q1) {
  return T * (1.0 + 0.6078 * q1);
}

/// Computes the temperature [K] from the virtual temperature [K] and a water
/// vapor mass mixing ratio [kg vapor/kg dry air].
/// @param [in] Tv virtual temperature [K]
/// @param [in] q1 specific humidity [-]
KOKKOS_INLINE_FUNCTION Real temperature_from_virtual_temperature(Real Tv,
                                                                 Real q1) {
  return Tv / (1.0 + 0.6078 * q1);
}

/// Computes the dry air density from the total air density using the specific
/// humidity.
/// @param [in] rho total air density [kg/m3]
/// @param [in] q1 specific humidity [-]
KOKKOS_INLINE_FUNCTION Real dry_air_density_from_total_air_density(Real rho,
                                                                   Real q1) {
  return rho * (1 - q1);
}

/// Computes the mass density of water vapor from the total mass density using
/// the specific humidity.
/// @param [in] rho total mass density [kg/m3]
/// @param [in] q1 specific humidity [-]
KOKKOS_INLINE_FUNCTION Real vapor_from_total_mass_density(Real rho, Real q1) {
  return rho * q1;
}

/// Computes the water vapor mixing ratio from the specific humidity. This
/// calculation diverges at q1 = 1.
/// @param [in] q1 specific humidity [-]
KOKKOS_INLINE_FUNCTION Real vapor_mixing_ratio_from_specific_humidity(Real q1) {
  return q1 / (1 - q1);
}

/// Computes the specific humidity from the water vapor mixing ratio. This
/// calculation diverges at q1 = 1.
/// @param [in] qv water vapor mixing ratio [kg vapor / kg dry air]
KOKKOS_INLINE_FUNCTION Real specific_humidity_from_vapor_mixing_ratio(Real qv) {
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
///  @return es(T) saturation vapor pressure of water vapor [Pa]
KOKKOS_INLINE_FUNCTION Real vapor_saturation_pressure_magnus_ew(Real T) {
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
KOKKOS_INLINE_FUNCTION Real vapor_saturation_pressure_magnus(Real T, Real P) {
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
///   -100 to +100 C, Proceedings of the Third International Symposium on
///   Humidity & Moisture, Teddington, London, England, April 1998.
///
KOKKOS_INLINE_FUNCTION Real vapor_saturation_pressure_hardy(Real T) {
  const Real g[8] = {-2.8365744e3,    -6.028076559e3, 1.954263612e1,
                     -2.737830188e-2, 1.6261698e-5,   7.0229056e-10,
                     -1.8680009e-13,  2.7150305};
  Real log_es = 0.0;
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
KOKKOS_INLINE_FUNCTION Real saturation_mixing_ratio_hardy(Real T, Real P) {
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
KOKKOS_INLINE_FUNCTION Real relative_humidity_from_vapor_mixing_ratio(
    Real w, Real T, Real p,
    Real (*wsat)(Real, Real) = saturation_mixing_ratio_hardy) {
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
KOKKOS_INLINE_FUNCTION Real relative_humidity_from_specific_humidity(Real q,
                                                                     Real T,
                                                                     Real P) {
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
KOKKOS_INLINE_FUNCTION Real vapor_mixing_ratio_from_relative_humidity(
    Real rel_hum, Real p, Real T,
    Real (*wsat)(Real, Real) = saturation_mixing_ratio_hardy) {
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
KOKKOS_INLINE_FUNCTION Real mean_particle_diameter_from_volume(
    const Real mode_mean_particle_volume, const Real mean_std_dev) {
  const double pio6 = Constants::pi_sixth;
  return cbrt(mode_mean_particle_volume / pio6) *
         exp(-1.5 * square(log(mean_std_dev)));
}

///   This function is the inverse of
///   modal_mean_particle_diameter_from_volume; given the modal mean geometric
///   diameter, it returns the corresponding volume.
///
///   @param [in] geom_diam geometric mean diameter [m per particle]
///   @return mean volume [m^3 per particle]
KOKKOS_INLINE_FUNCTION Real
mean_particle_volume_from_diameter(Real geom_diam, Real mean_std_dev) {
  const double pio6 = Constants::pi_sixth;
  return cube(geom_diam) * exp(4.5 * square(log(mean_std_dev))) * pio6;
}

/// Compute the density of an ideal gas given its temperature and pressure.
///
/// Example usage (default assumes gas is air):
///    atm = haero::Atmosphere;
///    k = level idx;
///
///    rho = density_of_ideal_gas(atm.temperature(k), atm.pressure(k));
///
///    To compute total (moist) air, use virtual temperature instead of
///    temperature.
///
///    To compute density of a different gas than air, supply the
///    appropriate gas constant as the third argument.
///
/// @param [in] T temperature [K]
/// @param [in] P pressure [Pa]
/// @param [in] R gas constant [J/K/kg]
/// @return density [kg/m3]
KOKKOS_INLINE_FUNCTION Real density_of_ideal_gas(
    const Real T, const Real P, const Real R = Constants::r_gas_dry_air) {
  return P / (R * T);
}

} // namespace mam4::conversions

#endif
