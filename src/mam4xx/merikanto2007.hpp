// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MAM4XX_MERIKANTO2007_HPP
#define MAM4XX_MERIKANTO2007_HPP

#include "mam4_math.hpp"

#include <ekat_kokkos_types.hpp>

namespace mam4::merikanto2007 {

/// The functions in this file implement parameterizations described in
/// Merikanto et al, New parameterization of sulfuric acid-ammonia-water ternary
/// nucleation rates at tropospheric conditions, Journal of Geophysical Research
/// 112 (2007). Also included are corrections described in Merikanto et al,
/// Correction to "New parameterization...", Journal of Geophysical Research
/// 114 (2009). The implementations here were downloaded from
/// https://doi.org/10.1029/2009JD012136

/// These parameterizations are valid for the following ranges:
/// temperature:                235 - 295 K
/// relative humidity:          0.05 - 0.95
/// H2SO4 number concentration: 5e4 - 1e9 cm-3
/// NH3 molar mixing ratio:     0.1 - 1000 ppt

/// Returns the temperature range [K] for which the Merikanto el al (2007)
/// parameterizations are valid.
KOKKOS_INLINE_FUNCTION
Kokkos::pair<Real, Real> valid_temp_range() {
  return Kokkos::pair<Real, Real>({235.0, 295.0});
}

/// Returns the relative humidity range [-] for which the Merikanto el al (2007)
/// parameterizations are valid.
KOKKOS_INLINE_FUNCTION
Kokkos::pair<Real, Real> valid_rel_hum_range() {
  return Kokkos::pair<Real, Real>({0.05, 0.95});
}

/// Returns the H2SO4 number concentration range [cm-3] for which the Merikanto
/// et al (2000) parameterizations are valid.
KOKKOS_INLINE_FUNCTION
Kokkos::pair<Real, Real> valid_c_h2so4_range() {
  return Kokkos::pair<Real, Real>({5e4, 1e9});
}

/// Returns the NH3 molar mixing ratio range [ppt] for which the Merikanto
/// et al (2000) parameterizations are valid.
KOKKOS_INLINE_FUNCTION
Kokkos::pair<Real, Real> valid_xi_nh3_range() {
  return Kokkos::pair<Real, Real>({0.1, 1e3});
}

/// Computes the logarithm of the ternary nucleation rate [cm-3 s-1] as
/// parameterized by Merikanto et al (2007), eq 8.
/// @param [in] temp The atmospheric temperature [K]
/// @param [in] rel_hum The relative humidity [-]
/// @param [in] c_h2so4 The number concentration of H2SO4 gas [cm-3]
/// @param [in] xi_nh3 The molar mixing ratio of NH3 [ppt]
KOKKOS_INLINE_FUNCTION
Real log_nucleation_rate(Real temp, Real rel_hum, Real c_h2so4, Real xi_nh3) {
  auto c = c_h2so4;
  auto xi = xi_nh3;
  return -12.861848898625231 + 4.905527742256349 * xi -
         358.2337705052991 * rel_hum - 0.05463019231872484 * xi * temp +
         4.8630382337426985 * rel_hum * temp +
         0.00020258394697064567 * xi * square(temp) -
         0.02175548069741675 * rel_hum * square(temp) -
         2.502406532869512e-7 * xi * cube(temp) +
         0.00003212869941055865 * rel_hum * cube(temp) -
         4.39129415725234e6 / square(mam4::log(c)) +
         (56383.93843154586 * temp) / square(mam4::log(c)) -
         (239.835990963361 * square(temp)) / square(mam4::log(c)) +
         (0.33765136625580167 * cube(temp)) / square(mam4::log(c)) -
         (629.7882041830943 * rel_hum) / (cube(xi) * mam4::log(c)) +
         (7.772806552631709 * rel_hum * temp) / (cube(xi) * mam4::log(c)) -
         (0.031974053936299256 * rel_hum * square(temp)) /
             (cube(xi) * mam4::log(c)) +
         (0.00004383764128775082 * rel_hum * cube(temp)) /
             (cube(xi) * mam4::log(c)) +
         1200.472096232311 * mam4::log(c) -
         17.37107890065621 * temp * mam4::log(c) +
         0.08170681335921742 * square(temp) * mam4::log(c) -
         0.00012534476159729881 * cube(temp) * mam4::log(c) -
         14.833042158178936 * square(mam4::log(c)) +
         0.2932631303555295 * temp * square(mam4::log(c)) -
         0.0016497524241142845 * square(temp) * square(mam4::log(c)) +
         2.844074805239367e-6 * cube(temp) * square(mam4::log(c)) -
         231375.56676032578 * mam4::log(xi) -
         100.21645273730675 * rel_hum * mam4::log(xi) +
         2919.2852552424706 * temp * mam4::log(xi) +
         0.977886555834732 * rel_hum * temp * mam4::log(xi) -
         12.286497122264588 * square(temp) * mam4::log(xi) -
         0.0030511783284506377 * rel_hum * square(temp) * mam4::log(xi) +
         0.017249301826661612 * cube(temp) * mam4::log(xi) +
         2.967320346100855e-6 * rel_hum * cube(temp) * mam4::log(xi) +
         (2.360931724951942e6 * mam4::log(xi)) / mam4::log(c) -
         (29752.130254319443 * temp * mam4::log(xi)) / mam4::log(c) +
         (125.04965118142027 * square(temp) * mam4::log(xi)) / mam4::log(c) -
         (0.1752996881934318 * cube(temp) * mam4::log(xi)) / mam4::log(c) +
         5599.912337254629 * mam4::log(c) * mam4::log(xi) -
         70.70896612937771 * temp * mam4::log(c) * mam4::log(xi) +
         0.2978801613269466 * square(temp) * mam4::log(c) * mam4::log(xi) -
         0.00041866525019504 * cube(temp) * mam4::log(c) * mam4::log(xi) +
         75061.15281456841 * square(mam4::log(xi)) -
         931.8802278173565 * temp * square(mam4::log(xi)) +
         3.863266220840964 * square(temp) * square(mam4::log(xi)) -
         0.005349472062284983 * cube(temp) * square(mam4::log(xi)) -
         (732006.8180571689 * square(mam4::log(xi))) / mam4::log(c) +
         (9100.06398573816 * temp * square(mam4::log(xi))) / mam4::log(c) -
         (37.771091915932004 * square(temp) * square(mam4::log(xi))) /
             mam4::log(c) +
         (0.05235455395566905 * cube(temp) * square(mam4::log(xi))) /
             mam4::log(c) -
         1911.0303773001353 * mam4::log(c) * square(mam4::log(xi)) +
         23.6903969622286 * temp * mam4::log(c) * square(mam4::log(xi)) -
         0.09807872005428583 * square(temp) * mam4::log(c) *
             square(mam4::log(xi)) +
         0.00013564560238552576 * cube(temp) * mam4::log(c) *
             square(mam4::log(xi)) -
         3180.5610833308 * cube(mam4::log(xi)) +
         39.08268568672095 * temp * cube(mam4::log(xi)) -
         0.16048521066690752 * square(temp) * cube(mam4::log(xi)) +
         0.00022031380023793877 * cube(temp) * cube(mam4::log(xi)) +
         (40751.075322248245 * cube(mam4::log(xi))) / mam4::log(c) -
         (501.66977622013934 * temp * cube(mam4::log(xi))) / mam4::log(c) +
         (2.063469732254135 * square(temp) * cube(mam4::log(xi))) /
             mam4::log(c) -
         (0.002836873785758324 * cube(temp) * cube(mam4::log(xi))) /
             mam4::log(c) +
         2.792313345723013 * square(mam4::log(c)) * cube(mam4::log(xi)) -
         0.03422552111802899 * temp * square(mam4::log(c)) *
             cube(mam4::log(xi)) +
         0.00014019195277521142 * square(temp) * square(mam4::log(c)) *
             cube(mam4::log(xi)) -
         1.9201227328396297e-7 * cube(temp) * square(mam4::log(c)) *
             cube(mam4::log(xi)) -
         980.923146020468 * mam4::log(rel_hum) +
         10.054155220444462 * temp * mam4::log(rel_hum) -
         0.03306644502023841 * square(temp) * mam4::log(rel_hum) +
         0.000034274041225891804 * cube(temp) * mam4::log(rel_hum) +
         (16597.75554295064 * mam4::log(rel_hum)) / mam4::log(c) -
         (175.2365504237746 * temp * mam4::log(rel_hum)) / mam4::log(c) +
         (0.6033215603167458 * square(temp) * mam4::log(rel_hum)) /
             mam4::log(c) -
         (0.0006731787599587544 * cube(temp) * mam4::log(rel_hum)) /
             mam4::log(c) -
         89.38961120336789 * mam4::log(xi) * mam4::log(rel_hum) +
         1.153344219304926 * temp * mam4::log(xi) * mam4::log(rel_hum) -
         0.004954549700267233 * square(temp) * mam4::log(xi) *
             mam4::log(rel_hum) +
         7.096309866238719e-6 * cube(temp) * mam4::log(xi) *
             mam4::log(rel_hum) +
         3.1712136610383244 * cube(mam4::log(xi)) * mam4::log(rel_hum) -
         0.037822330602328806 * temp * cube(mam4::log(xi)) *
             mam4::log(rel_hum) +
         0.0001500555743561457 * square(temp) * cube(mam4::log(xi)) *
             mam4::log(rel_hum) -
         1.9828365865570703e-7 * cube(temp) * cube(mam4::log(xi)) *
             mam4::log(rel_hum);
}

/// Computes the "onset temperature" [K] (eq 10) above which Merikanto's
/// parameterization for the nucleation rate (eq 8) cannot be used (in which
/// case the authors suggest setting the nucleation rate to zero).
/// @param [in] rel_hum The relative humidity [-]
/// @param [in] c_h2so4 The number concentration of H2SO4 gas [cm-3]
/// @param [in] xi_nh3 The molar mixing ratio of NH3 [ppt]
KOKKOS_INLINE_FUNCTION
Real onset_temperature(Real rel_hum, Real c_h2so4, Real xi_nh3) {
  return 143.6002929064716 + 1.0178856665693992 * rel_hum +
         10.196398812974294 * mam4::log(c_h2so4) -
         0.1849879416839113 * square(mam4::log(c_h2so4)) -
         17.161783213150173 * mam4::log(xi_nh3) +
         109.92469248546053 * mam4::log(xi_nh3) / mam4::log(c_h2so4) +
         0.7734119613144357 * mam4::log(c_h2so4) * mam4::log(xi_nh3) -
         0.15576469879527022 * square(mam4::log(xi_nh3));
}

/// Computes the radius of a critical cluster [nm] as parameterized in Merikanto
/// et al (2007), eq 11.
/// @param [in] log_J The logarithm of the nucleation rate ["log (cm-3 s-1)"]
/// @param [in] temp The atmospheric temperature [K]
/// @param [in] c_h2so4 The number concentration of H2SO4 gas [cm-3]
/// @param [in] xi_nh3 The molar mixing ratio of NH3 [ppt]
KOKKOS_INLINE_FUNCTION
Real critical_radius(Real log_J, Real temp, Real c_h2so4, Real xi_nh3) {
  auto c = c_h2so4;
  auto xi = xi_nh3;
  return 3.2888553966535506e-1 - 3.374171768439839e-3 * temp +
         1.8347359507774313e-5 * square(temp) +
         2.5419844298881856e-3 * mam4::log(c) -
         9.498107643050827e-5 * temp * mam4::log(c) +
         7.446266520834559e-4 * square(mam4::log(c)) +
         2.4303397746137294e-2 * mam4::log(xi) +
         1.589324325956633e-5 * temp * mam4::log(xi) -
         2.034596219775266e-3 * mam4::log(c) * mam4::log(xi) -
         5.59303954457172e-4 * square(mam4::log(xi)) -
         4.889507104645867e-7 * temp * square(mam4::log(xi)) +
         1.3847024107506764e-4 * cube(mam4::log(xi)) +
         4.141077193427042e-6 * log_J - 2.6813110884009767e-5 * temp * log_J +
         1.2879071621313094e-3 * mam4::log(xi) * log_J -
         3.80352446061867e-6 * temp * mam4::log(xi) * log_J -
         1.8790172502456827e-5 * square(log_J);
}

/// Computes the total number of molecules in a critical cluster as
/// parameterized in Merikanto et al (2007), eq 12.
/// @param [in] log_J The logarithm of the nucleation rate ["log (cm-3 s-1)"]
/// @param [in] temp The atmospheric temperature [K]
/// @param [in] c_h2so4 The number concentration of H2SO4 gas [cm-3]
/// @param [in] xi_nh3 The molar mixing ratio of NH3 [ppt]
KOKKOS_INLINE_FUNCTION
Real num_critical_molecules(Real log_J, Real temp, Real c_h2so4, Real xi_nh3) {
  auto c = c_h2so4;
  auto xi = xi_nh3;
  return 57.40091052369212 - 0.2996341884645408 * temp +
         0.0007395477768531926 * square(temp) -
         5.090604835032423 * mam4::log(c) +
         0.011016634044531128 * temp * mam4::log(c) +
         0.06750032251225707 * square(mam4::log(c)) -
         0.8102831333223962 * mam4::log(xi) +
         0.015905081275952426 * temp * mam4::log(xi) -
         0.2044174683159531 * mam4::log(c) * mam4::log(xi) +
         0.08918159167625832 * square(mam4::log(xi)) -
         0.0004969033586666147 * temp * square(mam4::log(xi)) +
         0.005704394549007816 * cube(mam4::log(xi)) +
         3.4098703903474368 * log_J - 0.014916956508210809 * temp * log_J +
         0.08459090011666293 * mam4::log(xi) * log_J -
         0.00014800625143907616 * temp * mam4::log(xi) * log_J +
         0.00503804694656905 * square(log_J);
}

/// Computes the total number of H2SO4 molecules in a critical cluster as
/// parameterized in Merikanto et al (2007), eq 13.
/// @param [in] log_J The logarithm of the nucleation rate ["log (cm-3 s-1)"]
/// @param [in] temp The atmospheric temperature [K]
/// @param [in] c_h2so4 The number concentration of H2SO4 gas [cm-3]
/// @param [in] xi_nh3 The molar mixing ratio of NH3 [ppt]
KOKKOS_INLINE_FUNCTION
Real num_h2so4_molecules(Real log_J, Real temp, Real c_h2so4, Real xi_nh3) {
  auto c = c_h2so4;
  auto xi = xi_nh3;
  return -4.7154180661803595 + 0.13436423483953885 * temp -
         0.00047184686478816176 * square(temp) -
         2.564010713640308 * mam4::log(c) +
         0.011353312899114723 * temp * mam4::log(c) +
         0.0010801941974317014 * square(mam4::log(c)) +
         0.5171368624197119 * mam4::log(xi) -
         0.0027882479896204665 * temp * mam4::log(xi) +
         0.8066971907026886 * square(mam4::log(xi)) -
         0.0031849094214409335 * temp * square(mam4::log(xi)) -
         0.09951184152927882 * cube(mam4::log(xi)) +
         0.00040072788891745513 * temp * cube(mam4::log(xi)) +
         1.3276469271073974 * log_J - 0.006167654171986281 * temp * log_J -
         0.11061390967822708 * mam4::log(xi) * log_J +
         0.0004367575329273496 * temp * mam4::log(xi) * log_J +
         0.000916366357266258 * square(log_J);
}

/// Computes the total number of NH3 molecules in a critical cluster as
/// parameterized in Merikanto et al (2007), eq 14.
/// @param [in] log_J The logarithm of the nucleation rate ["log (cm-3 s-1)"]
/// @param [in] temp The atmospheric temperature [K]
/// @param [in] c_h2so4 The number concentration of H2SO4 gas [cm-3]
/// @param [in] xi_nh3 The molar mixing ratio of NH3 [ppt]
KOKKOS_INLINE_FUNCTION
Real num_nh3_molecules(Real log_J, Real temp, Real c_h2so4, Real xi_nh3) {
  auto c = c_h2so4;
  auto xi = xi_nh3;
  return 71.20073903979772 - 0.8409600103431923 * temp +
         0.0024803006590334922 * square(temp) +
         2.7798606841602607 * mam4::log(c) -
         0.01475023348171676 * temp * mam4::log(c) +
         0.012264508212031405 * square(mam4::log(c)) -
         2.009926050440182 * mam4::log(xi) +
         0.008689123511431527 * temp * mam4::log(xi) -
         0.009141180198955415 * mam4::log(c) * mam4::log(xi) +
         0.1374122553905617 * square(mam4::log(xi)) -
         0.0006253227821679215 * temp * square(mam4::log(xi)) +
         0.00009377332742098946 * cube(mam4::log(xi)) +
         0.5202974341687757 * log_J - 0.002419872323052805 * temp * log_J +
         0.07916392322884074 * mam4::log(xi) * log_J -
         0.0003021586030317366 * temp * mam4::log(xi) * log_J +
         0.0046977006608603395 * square(log_J);
}

} // namespace mam4::merikanto2007
#endif
