// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MAM4XX_VEHKAMAKI2002_HPP
#define MAM4XX_VEHKAMAKI2002_HPP

#include <haero/haero.hpp>
#include <haero/math.hpp>

namespace mam4::vehkamaki2002 {

using Real = haero::Real;
using haero::cube;
using haero::exp;
using haero::log;
using haero::square;

/// The functions in this file implement parameterizations described in
/// Vehkamaki et al, An improved parameterization for sulfuric acid-water /
/// nucleation rates for tropospheric and stratospheric conditions,
/// Journal of Geophysical Research 107 (2002). Also included are corrections
/// described in Vehkamaki et al, Correction to "An improved...",
/// Journal of Geophysical Research: Atmospheres 118 (2013).

/// These parameterizations are valid for the following ranges:
/// temperature:                230.15 - 300.15 K
/// relative humidity:          0.01 - 1
/// H2SO4 number concentration: 1e4 - 1e11 cm-3

/// Returns the temperature range [K] for which the Vehkamaki el al (2002)
/// parameterizations are valid.
KOKKOS_INLINE_FUNCTION
Kokkos::pair<Real, Real> valid_temp_range() {
  return Kokkos::pair<Real, Real>({230.15, 300.15});
}

/// Returns the relative humidity range [-] for which the Vehkamaki el al (2002)
/// parameterizations are valid.
KOKKOS_INLINE_FUNCTION
Kokkos::pair<Real, Real> valid_rel_hum_range() {
  return Kokkos::pair<Real, Real>({0.01, 1.});
}

/// Returns the H2SO4 number concentration range [cm-3] for which the Merikanto
/// et al (2000) parameterizations are valid.
KOKKOS_INLINE_FUNCTION
Kokkos::pair<Real, Real> valid_c_h2so4_range() {
  return Kokkos::pair<Real, Real>({1e4, 1e11});
}

/// Computes the mole fraction of sulfuric acid in a critical cluster as
/// parameterized by Vehkmaki et al (2002), eq 11.
/// @param [in] c_h2so4 The number concentration of H2SO4 gas [cm-3]
/// @param [in] temp The atmospheric temperature [K]
/// @param [in] rel_hum The relative humidity [-]
KOKKOS_INLINE_FUNCTION
Real h2so4_critical_mole_fraction(Real c_h2so4, Real temp, Real rel_hum) {
  // Calculate the mole fraction using eq 11 of Vehkamaki et al (2002).
  auto N_a = c_h2so4;
  return 0.740997 - 0.00266379 * temp - 0.00349998 * log(N_a) +
         0.0000504022 * temp * log(N_a) + 0.00201048 * log(rel_hum) -
         0.000183289 * temp * log(rel_hum) + 0.00157407 * square(log(rel_hum)) -
         0.0000179059 * temp * square(log(rel_hum)) +
         0.000184403 * cube(log(rel_hum)) -
         1.50345e-6 * temp * cube(log(rel_hum));
}

/// Computes the binary nucleation rate [m-3 s-1] as parameterized by
/// Vehkmaki et al (2002), eq 12.
/// @param [in] c_h2so4 The number concentration of H2SO4 gas [cm-3]
/// @param [in] temp The atmospheric temperature [K]
/// @param [in] rel_hum The relative humidity [-]
/// @param [in] x_crit The mole fraction of H2SO4 in a critical cluster [-]
KOKKOS_INLINE_FUNCTION
Real nucleation_rate(Real c_h2so4, Real temp, Real rel_hum, Real x_crit) {
  // Calculate the coefficients in eq 12 of Vehkamaki et al (2002).
  Real a = 0.14309 + 2.21956 * temp - 0.0273911 * square(temp) +
           0.0000722811 * cube(temp) + 5.91822 / x_crit;

  Real b = 0.117489 + 0.462532 * temp - 0.0118059 * square(temp) +
           0.0000404196 * cube(temp) + 15.7963 / x_crit;

  Real c = -0.215554 - 0.0810269 * temp + 0.00143581 * square(temp) -
           4.7758e-6 * cube(temp) - 2.91297 / x_crit;

  Real d = -3.58856 + 0.049508 * temp - 0.00021382 * square(temp) +
           3.10801e-7 * cube(temp) - 0.0293333 / x_crit;

  Real e = 1.14598 - 0.600796 * temp + 0.00864245 * square(temp) -
           0.0000228947 * cube(temp) - 8.44985 / x_crit;

  Real f = 2.15855 + 0.0808121 * temp - 0.000407382 * square(temp) -
           4.01957e-7 * cube(temp) + 0.721326 / x_crit;

  Real g = 1.6241 - 0.0160106 * temp + 0.0000377124 * square(temp) +
           3.21794e-8 * cube(temp) - 0.0113255 / x_crit;

  Real h = 9.71682 - 0.115048 * temp + 0.000157098 * square(temp) +
           4.00914e-7 * cube(temp) + 0.71186 / x_crit;

  Real i = -1.05611 + 0.00903378 * temp - 0.0000198417 * square(temp) +
           2.46048e-8 * cube(temp) - 0.0579087 / x_crit;

  Real j = -0.148712 + 0.00283508 * temp - 9.24619e-6 * square(temp) +
           5.00427e-9 * cube(temp) - 0.0127081 / x_crit;

  // Compute the nucleation rate using eq 12.
  auto N_a = c_h2so4;
  return exp(a + b * log(rel_hum) + c * square(log(rel_hum)) +
             d * cube(log(rel_hum)) + e * log(N_a) +
             f * log(rel_hum) * log(N_a) +
             g * square(log(rel_hum)) * (log(N_a)) + h * square(log(N_a)) +
             i * log(rel_hum) * square(log(N_a)) + j * cube(log(N_a)));
}

/// Computes the total number of molecules in a critical cluster as
/// parameterized in Vehkamaki et al (2002), eq 13.
/// @param [in] c_h2so4 The number concentration of H2SO4 gas [cm-3]
/// @param [in] temp The atmospheric temperature [K]
/// @param [in] rel_hum The relative humidity [-]
/// @param [in] x_crit The mole fraction of H2SO4 in a critical cluster [-]
KOKKOS_INLINE_FUNCTION
Real num_critical_molecules(Real c_h2so4, Real temp, Real rel_hum,
                            Real x_crit) {
  // Calc the coefficients for the number of molecules in a critical
  // cluster (eq 13).
  Real A = -0.00295413 - 0.0976834 * temp + 0.00102485 * square(temp) -
           2.18646e-6 * cube(temp) - 0.101717 / x_crit;

  Real B = -0.00205064 - 0.00758504 * temp + 0.000192654 * square(temp) -
           6.7043e-7 * cube(temp) - 0.255774 / x_crit;

  Real C = +0.00322308 + 0.000852637 * temp - 0.0000154757 * square(temp) +
           5.66661e-8 * cube(temp) + 0.0338444 / x_crit;

  Real D = +0.0474323 - 0.000625104 * temp + 2.65066e-6 * square(temp) -
           3.67471e-9 * cube(temp) - 0.000267251 / x_crit;

  Real E = -0.0125211 + 0.00580655 * temp - 0.000101674 * square(temp) +
           2.88195e-7 * cube(temp) + 0.0942243 / x_crit;

  Real F = -0.038546 - 0.000672316 * temp + 2.60288e-6 * square(temp) +
           1.19416e-8 * cube(temp) - 0.00851515 / x_crit;

  Real G = -0.0183749 + 0.000172072 * temp - 3.71766e-7 * square(temp) -
           5.14875e-10 * cube(temp) + 0.00026866 / x_crit;

  Real H = -0.0619974 + 0.000906958 * temp - 9.11728e-7 * square(temp) -
           5.36796e-9 * cube(temp) - 0.00774234 / x_crit;

  Real I = +0.0121827 - 0.00010665 * temp + 2.5346e-7 * square(temp) -
           3.63519e-10 * cube(temp) + 0.000610065 / x_crit;

  Real J = +0.000320184 - 0.0000174762 * temp + 6.06504e-8 * square(temp) -
           1.4177e-11 * cube(temp) + 0.000135751 / x_crit;

  // Compute n_tot using eq 13.
  auto N_a = c_h2so4;
  return exp(A + B * log(rel_hum) + C * square(log(rel_hum)) +
             D * cube(log(rel_hum)) + E * log(N_a) +
             F * log(rel_hum) * log(N_a) + G * square(log(rel_hum)) * log(N_a) +
             H * square(log(N_a)) + I * log(rel_hum) * square(log(N_a)) +
             J * cube(log(N_a)));
}

/// Computes the radius [nm] of a critical cluster as parameterized in Vehkamaki
/// et al (2002), eq 14.
/// @param [in] x_crit The mole fraction of H2SO4 in a critical cluster [-]
/// @param [in] n_tot The total number of molecules in the critical cluster [-]
KOKKOS_INLINE_FUNCTION
Real critical_radius(Real x_crit, Real n_tot) {
  return exp(-1.6524245 + 0.42316402 * x_crit + 0.3346648 * log(n_tot));
}

/// Computes the threshold number concentration of H2SO4 [cm-3] that produces a
/// nucleation rate of 1 cm-3 s-1 at the given temperature and relative
/// humidity as parameterized by Vehkamaki et al (2002), eq 15.
/// @param [in] temp The atmospheric temperature [K]
/// @param [in] rel_hum The relative humidity [-]
KOKKOS_INLINE_FUNCTION
Real h2so4_nucleation_threshold(Real temp, Real rel_hum) {
  return exp(-279.243 + 11.7344 * rel_hum + 22700.9 / temp -
             1088.64 * rel_hum / temp + 1.14436 * temp -
             0.0302331 * rel_hum * temp - 0.00130254 * square(temp) -
             6.38697 * log(rel_hum) + 854.98 * log(rel_hum) / temp +
             0.00879662 * temp * log(rel_hum));
}

} // namespace mam4::vehkamaki2002
#endif
