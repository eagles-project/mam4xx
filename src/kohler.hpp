#ifndef MAM4XX_KOHLER_HPP
#define MAM4XX_KOHLER_HPP

#include "mam4.hpp"
#include <ekat/ekat_pack_math.hpp>
#include <ekat/ekat_scalar_traits.hpp>
#include <haero/constants.hpp>
#include <haero/floating_point.hpp>
#include <haero/haero.hpp>
#include <haero/math.hpp>
#include <haero/root_finders.hpp>

namespace mam4 {

/// Surface tension of liquid water in air as a function of temperature
/**
  @param [in] T temperature [K]
  @return sigma [N/m]

  Called without an argument, the default input reproduces the value used by
  MAM4's approximation of constant surface tension, neglecting temperature
  dependence.

  This formula is valid from T = 248.16 K (supercooled liquid water) to the
   critical temperature Tc = 646.096 K (steam).

  IAPWS Release on Surface Tension of Ordinary Water Substance
  IAPWS R1-76(2014)
  http://www.iapws.org/relguide/Surf-H2O.html

  Referenced from: NIST WebBook Saturation Properties for Water
  https://webbook.nist.gov/cgi/fluid.cgi?TLow=273.16&THigh=343&TInc=1&Applet=on&Digits=5&ID=C7732185&Action=Load&Type=SatP&TUnit=K&PUnit=MPa&DUnit=mol%2Fl&HUnit=kJ%2Fmol&WUnit=m%2Fs&VisUnit=uPa*s&STUnit=N%2Fm&RefState=DEF

  accessed on September 20, 2022
*/
template <typename ScalarType>
KOKKOS_INLINE_FUNCTION ScalarType
surface_tension_water_air(const ScalarType T = Constants::triple_pt_h2o) {
  constexpr Real Tc = Constants::tc_water;
  constexpr Real B = 0.2358;
  constexpr Real b = -0.625;
  constexpr Real mu = 1.256;
  const auto tau = 1 - T / Tc;
  EKAT_KERNEL_ASSERT(T >= 248.16);
  EKAT_KERNEL_ASSERT(T < 646.096);
  return B * pow(tau, mu) * (1 + b * tau);
}

template <>
KOKKOS_INLINE_FUNCTION PackType
surface_tension_water_air<PackType>(const PackType T) {
  constexpr Real Tc = Constants::tc_water;
  constexpr Real B = 0.2358;
  constexpr Real b = -0.625;
  constexpr Real mu = 1.256;
  const auto tau = 1 - T / Tc;
  EKAT_KERNEL_ASSERT((T >= 248.16).all());
  EKAT_KERNEL_ASSERT((T < 646.096).all());
  return B * pow(tau, mu) * (1 + b * tau);
}

/// Kelvin coefficient
/**
  Equation (A1) from Ghan et al., 2011, Droplet nucleation: Physically-based
   parameterizations and comparative evaluation, J. Adv. Earth Sys. Mod. 3
   M10001.

  To reproduce MAM4 approximations, call this function without an argument;
  the default value is set to the constant value used by MAM4.

  ** Note: This subroutine uses SI units **

  MAM4 uses micrometers and milinewtons.

  @param [in] T temperature [K]
  @return Kelvin droplet coefficient [m]
*/
template <typename ScalarType>
KOKKOS_INLINE_FUNCTION ScalarType
kelvin_coefficient(const ScalarType T = Constants::triple_pt_h2o) {
  const Real density_h2o = Constants::density_h2o;
  const Real r_gas_h2o_vapor = Constants::r_gas_h2o_vapor;
  return 2 * surface_tension_water_air(T) / (r_gas_h2o_vapor * T * density_h2o);
}

/** @brief Struct that represents the Kohler polynomial.

  @f$ K(r_w) = \log(s) r_w^4 - A r_w^3 + (B - \log(s))r_d^3 r_w + A r_d^3 @f$

  where r_w is the wet radius, s is relative humidity, A is the Kelvin effect
  coefficient, B is hygroscopicity, and r_d is the dry radius.

  The Kohler polynomial is a quartic polynomial whose variable is particle wet
  radius. Equilibrium solutions are roots of this polynomial. Algebraically,
  there are 2 complex roots and 2 real roots.  Of the real roots, one is
  positive, the other is negative.
  Physically, only the real, positive root makes sense.

  Each instance corresponds to a separate set of coefficients, which are
  functions of the inputs.

  This struct conforms to the interface prescribed in math.hpp for
  scalar functions that are to be used with numerical rootfinding algorithms.

  This struct is templated on scalar type so that it can be used with PackType.
  If it is used with PackType, each element of the PackType corresponds to a
  separate KohlerPolynomial, with distinct coefficients.

  @warning This polynomial is severely ill-conditioned, to the point that it is
  sensitive to order-of-operations changes caused by compiler optimization
  flags.  We therefore require double precision.

  Properties of the Kohler Polynomial that are useful to finding its roots:

  1. K(0) = kelvin_droplet_effect_coeff * cube(r_dry) > 0
  2. K(r_dry) = r_dry**4 * hygroscopicity > 0

  Properties of the Kohler Polynomial that are useful to finding its roots given
  inputs that are within the bounds defined below:

  1. K(25*r_dry) < 0

*/
template <typename ScalarType = PackType> struct KohlerPolynomial {
  static_assert(
      std::is_same<typename ekat::ScalarTraits<ScalarType>::scalar_type,
                   double>::value,
      "double precision required.");

  /// Minimum value of relative humidity
  static constexpr double rel_humidity_min = 0.05;
  /// Above this relative humidity is considered saturated air, and cloud
  /// aerosol processes would apply
  static constexpr double rel_humidity_max = 0.98;
  /// Minimum hygroscopicity for E3SM v1 aerosol species
  static constexpr double hygro_min = 1e-6;
  /// Maximum hygroscopicity for E3SM v1 aerosol species
  static constexpr double hygro_max = 1.3;
  /// Minimum particle size for E3SM v1
  static constexpr double dry_radius_min_microns = 1e-3;
  /// Maximum particle size for Kohler theory
  static constexpr double dry_radius_max_microns = 30;

  using value_type = ScalarType;
  using scalar_type = typename ekat::ScalarTraits<ScalarType>::scalar_type;

  /// Coefficient in the Kohler polynomial
  ScalarType log_rel_humidity;
  /// Coefficient in the Kohler polynomial
  ScalarType hygroscopicity;
  /// Safe return value
  ScalarType dry_radius;
  /// Coefficient in the Kohler polynomial
  ScalarType dry_radius_cubed;
  /// Coefficient in the Kohler polynomial
  ScalarType kelvin_a;

  /** Constructor. Creates 1 instance of a KohlerPolynomial.

    @param [in] rel_h relative humidity
    @param [in] hygro hygroscopicity
    @param [in] dry_rad_microns particle dry radius [ 1e-6 m ]
    @param [in] temperature [K]
  */
  template <typename U>
  KOKKOS_INLINE_FUNCTION
  KohlerPolynomial(const U &rel_h, const U &hygro, const U &dry_rad_microns,
                   const U &temperature = Constants::triple_pt_h2o)
      : log_rel_humidity(log(rel_h)), hygroscopicity(hygro),
        dry_radius(dry_rad_microns), dry_radius_cubed(cube(dry_rad_microns)),
        kelvin_a(kelvin_coefficient(temperature)) {

    kelvin_a *= 1e6; /* convert from N to mN and m to micron */
    EKAT_KERNEL_ASSERT(valid_inputs(ScalarType(rel_h), ScalarType(hygro),
                                    ScalarType(dry_rad_microns)));
  }

  /** Constructor. Creates 1 instance of a KohlerPolynomial.

    @param m mask to skip padded pack values in valid_inputs check
    @param rel_h relative humidity
    @param hygro hygroscopicity
    @param dry_rad_microns particle dry radius [ 1e-6 m ]
    @param [in] temperature [K]
  */
  template <typename U>
  KOKKOS_INLINE_FUNCTION
  KohlerPolynomial(const MaskType &m, const U &rel_h, const U &hygro,
                   const U &dry_rad_microns,
                   const U &temperature = Constants::triple_pt_h2o)
      : log_rel_humidity(log(rel_h)), hygroscopicity(hygro),
        dry_radius(dry_rad_microns), dry_radius_cubed(cube(dry_rad_microns)),
        kelvin_a(kelvin_coefficient(temperature)) {

    kelvin_a *= 1e6; /* convert from N to mN and m to micron */

    EKAT_KERNEL_ASSERT(valid_inputs(m, ScalarType(rel_h), ScalarType(hygro),
                                    ScalarType(dry_rad_microns)));
  }

  /** Evaluates the Kohler polynomial.

    @f$ K(r_w) = \log(s) r_w^4 - A r_w^3 + (B - \log(s))r_d^3 r_w + A r_d^3 @f$

    where r_w is the wet radius, s is relative humidity, A is the Kelvin effect
    coefficient, B is hygroscopicity, and r_d is the dry radius.

    @param [in] Polynomial input value, wet_radius wet radius in microns [ 1e-6
    m]
    @return Polynomial value, wet_radius in microns [ 1e-6 m]
  */
  template <typename U>
  KOKKOS_INLINE_FUNCTION ScalarType operator()(const U &wet_radius) const {
    const ScalarType rwet = ScalarType(wet_radius);
    const ScalarType result =
        (log_rel_humidity * rwet - kelvin_a) * cube(rwet) +
        ((hygroscopicity - log_rel_humidity) * rwet + kelvin_a) *
            dry_radius_cubed;
    return result;
  }

  /** @brief Evaluates the derivative of the Kohler polynomial with respect to
    wet radius

    @f$ K'(r_w) = \frac{\partial K}(\partial r_w)(r_w) @f$

    @param [in] Polynomial input value, wet radius in microns [ 1e-6 m]
    @return Polynomial slope at input value
  */
  template <typename U>
  KOKKOS_INLINE_FUNCTION ScalarType derivative(const U &wet_radius) const {
    const ScalarType rwet = ScalarType(wet_radius);
    const ScalarType wet_radius_squared = square(rwet);
    const ScalarType result =
        (4 * log_rel_humidity * rwet - 3 * kelvin_a) * wet_radius_squared +
        (hygroscopicity - log_rel_humidity) * dry_radius_cubed;
    return result;
  }

  KOKKOS_INLINE_FUNCTION
  bool valid_inputs(const ScalarType &relh, const ScalarType &hyg,
                    const ScalarType &dry_rad) const {
    return (FloatingPoint<ScalarType>::in_bounds(relh, rel_humidity_min,
                                                 rel_humidity_max) and
            FloatingPoint<ScalarType>::in_bounds(hyg, hygro_min, hygro_max) and
            FloatingPoint<ScalarType>::in_bounds(
                dry_rad, dry_radius_min_microns, dry_radius_max_microns));
  }

  KOKKOS_INLINE_FUNCTION
  bool valid_inputs() const {
    return valid_inputs(exp(this->log_rel_humidity), this->hygroscopicity,
                        this->dry_radius);
  }

  template <typename ST = ScalarType>
  KOKKOS_INLINE_FUNCTION
      typename std::enable_if<ekat::ScalarTraits<ST>::is_simd, bool>::type
      valid_inputs(const MaskType &m) const {
    const ST relative_humidity = exp(this->log_rel_humidity);
    const double rhmin = rel_humidity_min;
    const double rhmax = rel_humidity_max;
    const double hmin = hygro_min;
    const double hmax = hygro_max;
    const double rmin = dry_radius_min_microns;
    const double rmax = dry_radius_max_microns;
    const double tol = FloatingPoint<double>::zero_tol;
    const auto in_bounds_mask = (relative_humidity >= (rhmin - tol)) &&
                                (relative_humidity <= (rhmax + tol)) &&
                                (this->hygroscopicity >= (hmin - tol)) &&
                                (this->hygroscopicity <= (hmax + tol)) &&
                                (this->dry_radius >= (rmin - tol)) &&
                                (this->dry_radius <= (rmax + tol));
    return (in_bounds_mask || (!m)).all();
  }

  template <typename ST = ScalarType>
  KOKKOS_INLINE_FUNCTION
      typename std::enable_if<ekat::ScalarTraits<ST>::is_simd, bool>::type
      valid_inputs(const MaskType &m, const ST &relh, const ST &hyg,
                   const ST &dry_rad) const {
    const double rhmin = rel_humidity_min;
    const double rhmax = rel_humidity_max;
    const double hmin = hygro_min;
    const double hmax = hygro_max;
    const double rmin = dry_radius_min_microns;
    const double rmax = dry_radius_max_microns;
    const double tol = FloatingPoint<double>::zero_tol;
    const auto in_bounds_mask =
        (relh >= (rhmin - tol)) && (relh <= (rhmax + tol)) &&
        (hyg >= (hmin - tol)) && (hyg <= (hmax + tol)) &&
        (dry_rad >= (rmin - tol)) && (dry_rad <= (rmax + tol));
    return (in_bounds_mask || (!m)).all();
  }
};

template <typename SolverType> struct KohlerSolver {
  typedef KohlerPolynomial<PackType> polynomial_type;
  typedef PackType value_type;
  PackType relative_humidity;
  PackType hygroscopicity;
  PackType dry_radius_microns;
  Real conv_tol;
  int n_iter;

  KOKKOS_INLINE_FUNCTION
  KohlerSolver(const PackType &rel_h, const PackType &hyg, const PackType &rdry,
               const Real tol)
      : relative_humidity(rel_h), hygroscopicity(hyg), dry_radius_microns(rdry),
        conv_tol(tol), n_iter(0) {}

  KOKKOS_INLINE_FUNCTION
  PackType solve() {
    PackType wet_radius_left(0.9 * dry_radius_microns);
    PackType wet_radius_right(50 * dry_radius_microns);
    PackType wet_radius_init(25 * dry_radius_microns);
    const Real triple_pt_h2o = Constants::triple_pt_h2o;
    const PackType default_T(triple_pt_h2o);
    const auto kpoly = polynomial_type(relative_humidity, hygroscopicity,
                                       dry_radius_microns, default_T);
    auto solver = SolverType(wet_radius_init, wet_radius_left, wet_radius_right,
                             conv_tol, kpoly);
    const PackType result = solver.solve();
    n_iter = solver.counter;
    return result;
  }
};

} // namespace mam4
#endif
