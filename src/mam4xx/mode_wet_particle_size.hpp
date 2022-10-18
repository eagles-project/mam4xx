#ifndef MAM4XX_WET_PARTICLE_SIZE_HPP
#define MAM4XX_WET_PARTICLE_SIZE_HPP

#include "mam4_types.hpp"
#include "aero_modes.hpp"
#include "aero_config.hpp"
#include "kohler.hpp"
#include "conversions.hpp"

#include "haero/haero.hpp"
#include "haero/atmosphere.hpp"

#include <ekat/ekat_pack.hpp>
#include <ekat/ekat_pack_math.hpp>

#include <sstream>

namespace mam4 {

static constexpr Real meters_to_microns = 1e6;
static constexpr Real microns_to_meters = 1e-6;
static constexpr Real dry_radius_min_microns =
  KohlerPolynomial<double>::dry_radius_min_microns;
static constexpr Real dry_radius_max_microns =
  KohlerPolynomial<double>::dry_radius_max_microns;
static constexpr Real solver_convergence_tol = 1e-10;

KOKKOS_INLINE_FUNCTION
void mode_avg_wet_particle_diam(const Diagnostics& diags, const Prognostics& progs,
  const Atmosphere& atm, const int mode_idx, const int pack_idx) {

  // set mask to skip padded values
  const auto nans_in = isnan(diags.dry_geometric_mean_diameter[mode_idx](pack_idx));

  // check hygroscopicity is in bounds for water uptake
  EKAT_KERNEL_ASSERT(FloatingPoint<PackType>::in_bounds(
    diags.hygroscopicity[mode_idx](pack_idx),
    KohlerPolynomial<PackType>::hygro_min,
    KohlerPolynomial<PackType>::hygro_max) || nans_in);

  // unit conversion multipliers
  const Real to_microns = meters_to_microns;
  const Real to_meters = microns_to_meters;
  const Real rdry_min = dry_radius_min_microns;
  const Real rdry_max = dry_radius_max_microns;

  // initialize result (default initializer is quiet nan)
  PackType wet_diam;

  // compute relative humidity
  PackType rel_humidity = conversions::relative_humidity_from_vapor_mixing_ratio(
    atm.vapor_mixing_ratio(pack_idx),
    atm.temperature(pack_idx),
    atm.pressure(pack_idx));
  // check that relative humidity is in bounds for interstitial water uptake
  // and Kohler theory
  EKAT_KERNEL_ASSERT(FloatingPoint<PackType>::in_bounds(rel_humidity,
    KohlerPolynomial<PackType>::rel_humidity_min,
    KohlerPolynomial<PackType>::rel_humidity_max) || nans_in);

  // set masks:
  //  case 1: dry air
  const auto rh_low = (rel_humidity <= modes[mode_idx].crystallization_pt);
  //  case 2: unsaturated air
  const auto rh_mid = (rel_humidity > modes[mode_idx].crystallization_pt) &&
                      (rel_humidity <= modes[mode_idx].deliquescence_pt);
  //  case 3: nearly saturated air
  const auto rh_high = (rel_humidity > modes[mode_idx].deliquescence_pt);
  //  case 4: particles too small
  const auto too_small =
    (0.5*to_microns * diags.dry_geometric_mean_diameter[mode_idx](pack_idx) <
      rdry_min);

  // no water uptake occurs if particles are too small or if air is too dry
  const auto use_dry_radius = (rh_low || too_small);
  wet_diam.set(use_dry_radius, diags.dry_geometric_mean_diameter[mode_idx](pack_idx));
  // for all other cases, we need a Kohler polynomial solve
  const auto needs_kohler = !use_dry_radius;

  if (needs_kohler.any()) {

    // convert from diameter in meters to radius in microns
    const PackType dry_radius_microns = 0.5*to_microns*
      diags.dry_geometric_mean_diameter[mode_idx](pack_idx);

    // check dry particle size is in bounds
    EKAT_KERNEL_ASSERT( ((dry_radius_microns <= rdry_max) || nans_in) .all() );

    // Set up Kohler solver
    // (requires double precision)
    typedef ekat::Pack<double, haero::HAERO_PACK_SIZE> DoublePack;
    typedef KohlerPolynomial<DoublePack> PolynomialType;
    typedef KohlerSolver<haero::math::NewtonSolver<PolynomialType>> SolverType;
    const Real tol = solver_convergence_tol;
    // Solve for the roots of the Kohler polynomial
    auto kohler_solver = SolverType(rel_humidity, diags.hygroscopicity[mode_idx](pack_idx),
      dry_radius_microns, tol);
    PackType rwet_microns = kohler_solver.solve();

    // Compute wet and dry particle volume in cubic meters
    const PackType dry_vol = conversions::mean_particle_volume_from_diameter(
      diags.dry_geometric_mean_diameter[mode_idx](pack_idx),
      modes[mode_idx].mean_std_dev);

    PackType wet_vol = conversions::mean_particle_volume_from_diameter(
      2*to_meters*rwet_microns, modes[mode_idx].mean_std_dev);

    // check that wet particle volume >= dry particle volume
    EKAT_KERNEL_ASSERT( ((wet_vol >= dry_vol) || nans_in).all() );
    // which implies that water volume is nonnegative
    PackType water_vol = wet_vol - dry_vol;

    // apply hysteresis to intermediate relative humidity values by
    // adjusting water volume
    //
    // comments from modal_aero_wateruptake.F90, lines 549--551,
    // subroutine modal_aero_wateruptake_sub:
    //   ! apply simple treatment of deliquesence/crystallization hysteresis
    //   ! for rhcrystal < rh < rhdeliques, aerosol water is a fraction of
    //   ! the "upper curve" value, and the fraction is a linear function of rh
    const Real hysteresis_factor =
      1 / (modes[mode_idx].deliquescence_pt - modes[mode_idx].crystallization_pt);
    water_vol.set(rh_mid, hysteresis_factor*water_vol*(rel_humidity -
      modes[mode_idx].crystallization_pt));

    // check that hysteresis does not cause negative water content
    EKAT_KERNEL_ASSERT( ((water_vol >= 0) || nans_in).all() );

    wet_vol = dry_vol + water_vol;

    const PackType rwet_hyst =
      0.5 * conversions::mean_particle_diameter_from_volume(
        wet_vol, modes[mode_idx].mean_std_dev);

    wet_diam.set(rh_mid, rwet_hyst);
    wet_diam.set(rh_high, 2*to_meters*rwet_microns);
  }

  diags.wet_geometric_mean_diameter[mode_idx](pack_idx) = wet_diam;
}

KOKKOS_INLINE_FUNCTION
void mode_avg_wet_particle_diam(const Diagnostics& diags, const Prognostics& progs,
  const Atmosphere& atm, const int pack_idx) {
  for (int m=0; m<AeroConfig::num_modes(); ++m) {
    mode_avg_wet_particle_diam(diags, progs, atm, m, pack_idx);
  }
}

} // namespace mam4
#endif
