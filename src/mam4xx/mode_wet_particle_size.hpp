#ifndef MAM4XX_WET_PARTICLE_SIZE_HPP
#define MAM4XX_WET_PARTICLE_SIZE_HPP

#include <mam4xx/aero_config.hpp>
#include <mam4xx/aero_modes.hpp>
#include <mam4xx/conversions.hpp>
#include <mam4xx/kohler.hpp>
#include <mam4xx/mam4_types.hpp>

#include <haero/atmosphere.hpp>
#include <haero/haero.hpp>

#include <ekat/ekat_pack.hpp>
#include <ekat/ekat_pack_math.hpp>

namespace mam4 {

static constexpr Real meters_to_microns = 1e6;
static constexpr Real microns_to_meters = 1e-6;
static constexpr Real dry_radius_min_microns =
    KohlerPolynomial<double>::dry_radius_min_microns;
static constexpr Real dry_radius_max_microns =
    KohlerPolynomial<double>::dry_radius_max_microns;
static constexpr Real wet_radius_max_microns = 30.0;
static constexpr Real solver_convergence_tol = 1e-10;

///  Compute aerosol particle wet diameter for interstitial aerosols
///  in a single mode.
///
///  This version can be called in parallel over both modes and column packs.
///
///  This function replaces subroutine modal_aero_wateruptake_dr from
///  file modal_aero_wateruptake.F90.
///
///  Inputs are the mode averages contained in @ref Diagnostics (dry particle size,
///  hygroscopicity) and @ref Atmosphere (vapor mass mixing ratio).
///  Diags are marked 'const' because they need to be able to be captured
///  by value by a lambda.  The Views inside the Diags struct are const,
///  but the data contained by the Views can change.
///
///  @param [in/out] diags dry/wet particle geometric mean diameter
///  @param [in] atm Atmosphere contains (T, P, w) data
///  @param [in] mode_idx mode that needs wet particle size data
///  @param [in] pack_idx column pack index
KOKKOS_INLINE_FUNCTION
void mode_avg_wet_particle_diam(const Diagnostics &diags, const Atmosphere &atm,
                                const int mode_idx, const int pack_idx) {

  // check hygroscopicity is in bounds for water uptake
  EKAT_KERNEL_ASSERT(FloatingPoint<PackType>::in_bounds(
      diags.hygroscopicity[mode_idx](pack_idx),
      KohlerPolynomial<double>::hygro_min,
      KohlerPolynomial<double>::hygro_max));

  // unit conversion multipliers
  const Real to_microns = meters_to_microns;
  const Real to_meters = microns_to_meters;
  const Real rdry_min = dry_radius_min_microns;
  const Real rdry_max = dry_radius_max_microns;

  // initialize result (default initializer is quiet nan)
  PackType wet_diam;

  // compute relative humidity
  PackType rel_humidity =
      conversions::relative_humidity_from_vapor_mixing_ratio(
          atm.vapor_mixing_ratio(pack_idx), atm.temperature(pack_idx),
          atm.pressure(pack_idx));
  // check that relative humidity is in bounds for interstitial water uptake
  // and Kohler theory
  EKAT_KERNEL_ASSERT(FloatingPoint<PackType>::in_bounds(
      rel_humidity, KohlerPolynomial<double>::rel_humidity_min,
      KohlerPolynomial<double>::rel_humidity_max));

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
      (0.5 * to_microns *
           diags.dry_geometric_mean_diameter[mode_idx](pack_idx) <
       rdry_min);

  // no water uptake occurs if particles are too small or if air is too dry
  const auto use_dry_radius = (rh_low || too_small);
  wet_diam.set(use_dry_radius,
               diags.dry_geometric_mean_diameter[mode_idx](pack_idx));
  // for all other cases, we need a Kohler polynomial solve
  const auto needs_kohler = !use_dry_radius;

  if (needs_kohler.any()) {

    // convert from diameter in meters to radius in microns
    const PackType dry_radius_microns =
        0.5 * to_microns *
        diags.dry_geometric_mean_diameter[mode_idx](pack_idx);

    // check dry particle size is in bounds
    EKAT_KERNEL_ASSERT((dry_radius_microns <= rdry_max).all());

    // Set up Kohler solver
    // (requires double precision)
    typedef ekat::Pack<double, haero::HAERO_PACK_SIZE> DoublePack;
    typedef KohlerPolynomial<DoublePack> PolynomialType;
    typedef KohlerSolver<haero::math::NewtonSolver<PolynomialType>> SolverType;
    const Real tol = solver_convergence_tol;
    // Solve for the roots of the Kohler polynomial
    //
    //  This step replaces the mam4 subroutine modal_aero_kohler with
    //  a new solver that is better conditioned and stable for
    //  finite precision computations.
    SolverType kohler_solver =
        SolverType(rel_humidity, diags.hygroscopicity[mode_idx](pack_idx),
                   dry_radius_microns, tol);
    auto rwet_microns = PackType(kohler_solver.solve());

    // set maximum wet radius of 30 microns
    //
    //  from modal_aero_wateruptake.F90 line 721, this upper bound is
    //  ! based on 1 day lifetime
    const Real rwet_max = wet_radius_max_microns;
    const auto too_big = (rwet_microns > rwet_max);
    rwet_microns.set(too_big, rwet_max);

    // Compute wet and dry particle volume in cubic meters
    //
    //  This volume calculation represents a bug fix from legacy Mam4.
    //  In subroutine modal_aero_wateruptake_sub from file
    //  modal_aero_wateruptake.F90, particle volumes are computed using the
    //  spherical geometric formulas without accounting for the probability
    //  density function (PDF) that represents the modal particle size
    //  distribution, which is an inconsistency: the dry_geometric_mean_diameter
    //  input accounts for the PDF while the same quantity for wet particles
    //  does not.
    //
    //  Here, we use the PDF functions for both.
    const PackType dry_vol = conversions::mean_particle_volume_from_diameter(
        diags.dry_geometric_mean_diameter[mode_idx](pack_idx),
        modes[mode_idx].mean_std_dev);

    PackType wet_vol = conversions::mean_particle_volume_from_diameter(
        2 * to_meters * rwet_microns, modes[mode_idx].mean_std_dev);

    // check that wet particle volume >= dry particle volume
    EKAT_KERNEL_ASSERT((wet_vol >= dry_vol).all());
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
    const Real hysteresis_factor = 1 / (modes[mode_idx].deliquescence_pt -
                                        modes[mode_idx].crystallization_pt);
    EKAT_KERNEL_ASSERT(hysteresis_factor > 0);
    water_vol.set(rh_mid,
                  hysteresis_factor * water_vol *
                      (rel_humidity - modes[mode_idx].crystallization_pt));

    // check that hysteresis does not cause negative water content
    EKAT_KERNEL_ASSERT((water_vol >= 0).all());

    wet_vol = dry_vol + water_vol;

    const PackType rwet_hyst =
        0.5 * conversions::mean_particle_diameter_from_volume(
                  wet_vol, modes[mode_idx].mean_std_dev);

    wet_diam.set(rh_mid, 2 * rwet_hyst);
    wet_diam.set(rh_high, 2 * to_meters * rwet_microns);
  }

  diags.wet_geometric_mean_diameter[mode_idx](pack_idx) = wet_diam;
}

///  Compute aerosol particle wet diameter for interstitial aerosols
///  in all modes.
///
///  This version can be called in parallel column packs.
///
///  This function replaces subroutine modal_aero_wateruptake_dr from
///  file modal_aero_wateruptake.F90.
///
///  Inputs are the mode averages contained in @ref Diagnostics (dry particle
///  size, hygroscopicity) and @ref Atmosphere (vapor mass mixing ratio).
///  Diags are marked 'const' because they need to be able to be captured
///  by value by a lambda.  The Views inside the Diags struct are const,
///  but the data contained by the Views can change.
///
///  @param [in/out] diags dry/wet particle geometric mean diameter
///  @param [in] atm Atmosphere contains (T, P, w) data
///  @param [in] pack_idx column pack index
KOKKOS_INLINE_FUNCTION
void mode_avg_wet_particle_diam(const Diagnostics &diags, const Atmosphere &atm,
                                const int pack_idx) {
  for (int m = 0; m < AeroConfig::num_modes(); ++m) {
    mode_avg_wet_particle_diam(diags, atm, m, pack_idx);
  }
}

} // namespace mam4
#endif
