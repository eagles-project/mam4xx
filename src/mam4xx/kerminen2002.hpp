// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MAM4XX_KERMINEN2002_HPP
#define MAM4XX_KERMINEN2002_HPP

#include <haero/haero.hpp>
#include <haero/math.hpp>

namespace mam4::kerminen2002 {

using Real = haero::Real;
using haero::cube;
using haero::exp;
using haero::log;
using haero::square;

/// The functions in this file implement parameterizations described in
/// Kerminen and Kulmala, Analytical formulae connecting the "real" and the
/// "apparent" nucleation rate and the nuclei number concentration for
/// atmospheric nucleation events, Aerosol Science 33 (2002).
///
/// The parameterization has been adapted for systems whose aerosol particles
/// are nucleated from H2SO4 and NH3 gases.

/// This function computes the growth rate @f$GR@f$ [nm/h] for particles with
/// the given number concentration, mass density, and molecular weight at the
/// given temperature, using KK2002 eq 21.
/// @param [in] c The number concentration density of nucleated particles
/// [kg/m3]
/// @param [in] rho The mass density of nucleated particles [kg/m3]
/// @param [in] mw The molecular weight of the nucleated particle species
/// [kg/mol]
/// @param [in] temp The atmospheric temperature [K]
KOKKOS_INLINE_FUNCTION
Real growth_rate(const Real c, const Real rho, const Real mw, const Real temp) {
  Real speed = 14.7 * sqrt(temp); // molecular speed [m/s]
  return 3.0e-9 * speed * mw * c / rho;
}

/// This function computes the condensation sink @f$CS'@f$ parameter [m-2] used
/// to compute the @f$\eta@f$ parameter for the nucleated particle growth
/// parameterization.
/// @param [in] rho_air The mass density of dry air [kg/m3]
/// @param [in] d_wet_grown The wet diameter of grown particles [nm]
/// @param [in] c_tot The total number concentration of aerosol particles [#/cc]
KOKKOS_INLINE_FUNCTION
Real condensation_sink(const Real rho_air, const Real d_wet_grown,
                       const Real c_tot) {
  // For the purposes of this calculation, we use alpha == 1 and we use the mean
  // free path of air as computed from the air density in the calculation of the
  // Knudsen number for the nucleation mode.
  // NOTE: this differs from the MAM4 calculation, which uses an H2SO4
  // NOTE: uptake rate that assumes a process ordering, which we're no
  // NOTE: longer allowed to do.
  const Real alpha = 1; // accommodation coefficient

  // The Knudsen number for the nucleated particles is Kn = 2 * lambda / d,
  // where lambda is the mean free path of air, and d is the grown particle
  // diameter. The mean free path is 1/(n * sigma), where n = rho_air/mw_air
  // is the number density of air, and sigma = pi*d^2 is the cross section
  // of a grown particle. Putting everything togther, we have
  //      2 * mw_air
  // Kn = --------------- 3
  //      pi * rho_air * d
  // TODO: should we attempt to estimate the wet number density?
  static const Real mw_air = Constants::molec_weight_dry_air;
  static const Real pi = Constants::pi;
  const Real Kn = 2 * mw_air / (pi * rho_air * cube(d_wet_grown));

  // Compute the transitional correction for the condensational mass flux
  // (Fuchs and Sutugin, 1971, or KK2002 eq 4).
  const auto beta =
      (1.0 + Kn) / (1.0 + 0.377 * Kn + 1.33 * Kn * (1 + Kn) / alpha);

  // Compute the condensation sink from KK2002 eq 3.
  return 0.5 * d_wet_grown * beta * c_tot;
}

/// This function computes the growth parameter @f$\eta@f$ [nm] in terms of a
/// given condensation growth rate GR and a given condensity sink CS'.
/// @param [in] temp The atmospheric temperature [K]
/// @param [in] d_dry_crit The dry diameter of particles in a CC [nm]
/// @param [in] d_wet_crit The wet diameter of particles in a CC [nm]
/// @param [in] d_dry_grown The dry diameter of grown particles [nm]
/// @param [in] d_wet_grown The wet diameter of grown particles [nm]
/// @param [in] rho_grown The mass density of grown particles [kg/m3]
/// @param [in] cond_growth_rate The condensation growth rate GR [m/s]
/// @param [in] cond_sink The condensation sink CS' [1/m2]
KOKKOS_INLINE_FUNCTION
Real growth_parameter(const Real temp, const Real d_dry_crit,
                      const Real d_wet_crit, const Real d_dry_grown,
                      const Real d_wet_grown, const Real rho_grown,
                      const Real cond_growth_rate, const Real cond_sink) {
  // Compute gamma from KK2002 eq 22 [nm2/m2/h], neglecting the
  // (d_mean/150)^0.048 factor.
  Real gamma = 0.23 * pow(d_wet_crit, 0.2) * pow(d_wet_grown / 3.0, 0.075) *
               pow(1e-3 * rho_grown, -0.33) * pow(temp / 293.0, -0.75);

  // Compute eta [nm] using KK2002 eq 11.
  return gamma * cond_sink / cond_growth_rate;
}

/// This function computes the growth parameter @f$\eta@f$ [nm] used in the
/// conversion from the "real" (base) nucleation rate to the "apparent"
/// nucleation rate:
/// @f$J_{app} = J_{real} \exp\left[\frac{\eta}{d_f} -
///                                 \frac{\eta}{d_i}\right],@f$
/// where @f$d_i@f$ and @f$d_f@f$ are the initial (nucleated) and final (grown)
/// wet diameters of the particles in question.
/// @param [in] c_so4 The number concentration of SO4 aerosol [#/cc]
/// @param [in] c_nh4 The number concentration of NH4 aerosol [#/cc]
/// @param [in] nh4_to_so4_molar_ratio The molar ratio of NH4 to SO4 [-]
/// @param [in] temp The atmospheric temperature [K]
/// @param [in] rel_hum The atmospheric relative humidity [-]
/// @param [in] d_dry_crit The dry diameter of particles in a CC [nm]
/// @param [in] d_wet_crit The wet diameter of particles in a CC [nm]
/// @param [in] d_dry_grown The dry diameter of grown particles [nm]
/// @param [in] rho_grown The mass density of grown particles [kg/m3]
/// @param [in] rho_air The mass density of dry air [kg/m3]
/// @param [in] mw_h2so4 The molecular weight of H2SO4 gas [kg/mol]
KOKKOS_INLINE_FUNCTION
Real growth_parameter(const Real c_so4, const Real c_nh4,
                      const Real nh4_to_so4_molar_ratio, const Real temp,
                      const Real rel_hum, const Real d_dry_crit,
                      const Real d_wet_crit, const Real d_dry_grown,
                      const Real rho_grown, const Ral rho_air,
                      const Real mw_h2so4) {
  // Compute the wet/dry volume ratio using the simple Kohler approximation
  // for ammonium sulfate and bisulfate.
  const auto bounded_rel_hum = max(0.10, min(0.95, rel_hum));
  const auto wet_dry_vol_ratio = 1.0 - 0.56 / log(bounded_rel_hum);

  // Compute the growth rate [nm/h] of new particles.

  // Compute the fraction of the wet volume due to SO4 aerosol.
  Real V_frac_wet_so4 =
      1.0 / (wet_dry_vol_ratio * (1.0 + nh4_to_so4_molar_ratio * 17.0 / 98.0));

  // Compute the condensation growth rate gr [nm/h] of new particles from
  // KK2002 eq 21 for H2SO4 uptake and correct for NH3/H2O uptake.
  Real cond_growth_rate = growth_rate(c_so4, rho_grown, mw_h2so4, temp);
  cond_growth_rate /= V_frac_wet_so4;

  // Wet diameter [nm] of grown particles with dry diameter d_dry_grown.
  Real d_wet_grown = 1e9 * d_dry_grown * pow(wet_dry_vol_ratio, 1.0 / 3.0);

  // Compute the condensation sink CS' from KK2002 eqs 3-4.
  Real cond_sink = condensation_sink(rho_air, d_wet_grown, c_so4 + c_nh4);

  // Compute eta [nm] using KK2002 eq 11.
  return growth_parameter(temp, d_dry_crit, d_wet_crit, d_dry_grown,
                          d_wet_grown, rho_grown, cond_growth_rate, cond_sink);
}

/// Computes the conversion factor connecting the "real" (base) nucleation rate
/// @f$J@f$ to the "apparent" nucleation rate using the given growth parameter
/// @f$\eta@f$ and the initial and final wet particle diameters:
/// @f$J_{app} = J_{real} \exp\left[\frac{\eta}{d_f} -
///                                 \frac{\eta}{d_i}\right],@f$
/// where and @f$d_i@f$ and @f$d_f@f$ [nm] are the initial (nucleated) and final
/// (grown) wet diameters of the particles in question.
/// @param [in] eta The growth parameter @f$\eta@f$ [nm]
/// @param [in] d_wet_crit The wet diameter of particles in a CC [nm]
/// @param [in] d_wet_grown The wet diameter of grown particles [nm]
KOKKOS_INLINE_FUNCTION
Real apparent_nucleation_factor(const Real eta, const Real d_wet_crit,
                                const Real d_wet_grown) {
  return exp(eta / d_wet_grown - eta / d_wet_crit);
}

/// Computes a conversion factor that transforms the "real" (base) nucleation
/// rate @f$J@f$ to an "apparent" nucleation rate that accounts for the growth
/// of nucleated particles in critical clusters (CC) needed to place them into
/// an appropriate nucleation mode. The converstion from the "real" to
/// "apparent" nucleation rate is
/// @f$J_{app} = J_{real} \exp\left[\frac{\eta}{d_f} -
///                                 \frac{\eta}{d_i}\right],@f$
/// where @f$\eta@f$ [nm]is a growth parameter computed by the parameterization,
/// and @f$d_i@f$ and @f$d_f@f$ [nm] are the initial (nucleated) and final
/// (grown) wet diameters of the particles in question.
/// @param [in] c_so4 The number concentration of SO4 aerosol [#/cc]
/// @param [in] c_nh4 The number concentration of NH4 aerosol [#/cc]
/// @param [in] nh4_to_so4_molar_ratio The molar ratio of NH4 to SO4 [-]
/// @param [in] temp The atmospheric temperature [K]
/// @param [in] rel_hum The atmospheric relative humidity [-]
/// @param [in] d_dry_crit The dry diameter of particles in a CC [nm]
/// @param [in] d_wet_crit The wet diameter of particles in a CC [nm]
/// @param [in] d_dry_grown The dry diameter of grown particles [nm]
/// @param [in] rho_grown The mass density of grown particles [kg/m3]
/// @param [in] rho_air The mass density of dry air [kg/m3]
/// @param [in] mw_h2so4 The molecular weight of H2SO4 gas [kg/mol]
KOKKOS_INLINE_FUNCTION
Real apparent_nucleation_factor(const Real c_so4, const Real c_nh4,
                                const Real nh4_to_so4_molar_ratio,
                                const Real temp, const Real rel_hum,
                                const Real d_dry_crit, const Real d_wet_crit,
                                const Real d_dry_grown, const Real rho_grown,
                                const Real rho_air, const Real mw_h2so4) {
  // Compute the wet/dry volume ratio using the simple Kohler approximation
  // for ammonium sulfate and bisulfate.
  const auto bounded_rel_hum = max(0.10, min(0.95, rel_hum));
  const auto wet_dry_vol_ratio = 1.0 - 0.56 / log(bounded_rel_hum);

  // Wet diameter [nm] of grown particles with dry diameter d_dry_grown.
  Real d_wet_grown = 1e9 * d_dry_grown * pow(wet_dry_vol_ratio, 1.0 / 3.0);

  // Growth parameter eta.
  Real eta = growth_parameter(c_so4, c_nh4, nh4_to_so4_molar_ratio, temp,
                              rel_hum, d_dry_crit, d_wet_crit, d_dry_grown,
                              rho_grown, rho_air, mw_h2so4);

  return apparent_nucleation_factor(eta, d_wet_crit, d_wet_grown);
}

} // namespace mam4::kerminen2002

#endif
