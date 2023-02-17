// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MAM4XX_WET_PARTICLE_SIZE_HPP
#define MAM4XX_WET_PARTICLE_SIZE_HPP

#include <mam4xx/aero_config.hpp>
#include <mam4xx/aero_modes.hpp>
#include <mam4xx/conversions.hpp>
#include <mam4xx/kohler.hpp>
#include <mam4xx/mam4_types.hpp>

#include <haero/atmosphere.hpp>
#include <haero/haero.hpp>

namespace mam4 {

static constexpr Real meters_to_microns = 1e6;
static constexpr Real microns_to_meters = 1e-6;
static constexpr Real dry_radius_min_microns =
    KohlerPolynomial::dry_radius_min_microns;
static constexpr Real dry_radius_max_microns =
    KohlerPolynomial::dry_radius_max_microns;
static constexpr Real wet_radius_max_microns = 30.0;
static constexpr Real solver_convergence_tol = 1e-10;

///  Compute aerosol particle wet diameter for interstitial aerosols
///  in a single mode.
///
///  This version can be called in parallel over both modes and vertical levels.
///
///  This function replaces subroutine modal_aero_wateruptake_dr from
///  file modal_aero_wateruptake.F90.
///
///  Inputs are the mode averages contained in @ref Diagnostics (dry particle
///  size, hygroscopicity) and @ref Atmosphere (vapor mass mixing ratio). Diags
///  are marked 'const' because they need to be able to be captured by value by
///  a lambda.  The Views inside the Diags struct are const, but the data
///  contained by the Views can change.
///
///  @param [in/out] diags dry/wet particle geometric mean diameter
///  @param [in] atm Atmosphere contains (T, P, w) data
///  @param [in] mode_idx mode that needs wet particle size data
///  @param [in] k column vertical level index
KOKKOS_INLINE_FUNCTION
void mode_avg_wet_particle_diam_water_uptake(const Diagnostics &diags,
                                             const Atmosphere &atm,
                                             int mode_idx, int k) {

  // check hygroscopicity is in bounds for water uptake
  EKAT_KERNEL_ASSERT(FloatingPoint<Real>::in_bounds(
      diags.hygroscopicity[mode_idx](k), KohlerPolynomial::hygro_min,
      KohlerPolynomial::hygro_max));

  // unit conversion multipliers
  const Real to_microns = meters_to_microns;
  const Real to_meters = microns_to_meters;
  const Real rdry_min = dry_radius_min_microns;
  const Real rdry_max = dry_radius_max_microns;

  // initialize result (default initializer is quiet nan)
  Real wet_diam = 0;

  // compute relative humidity
  Real rel_humidity = conversions::relative_humidity_from_vapor_mixing_ratio(
      atm.vapor_mixing_ratio(k), atm.temperature(k), atm.pressure(k));
  // check that relative humidity is in bounds for interstitial water uptake
  // and Kohler theory
  EKAT_KERNEL_ASSERT(FloatingPoint<Real>::in_bounds(
      rel_humidity, KohlerPolynomial::rel_humidity_min,
      KohlerPolynomial::rel_humidity_max));

  // determine humidity conditions
  //  case 1: dry air
  const auto rh_low = (rel_humidity <= modes(mode_idx).crystallization_pt);
  //  case 2: unsaturated air
  const auto rh_mid = (rel_humidity > modes(mode_idx).crystallization_pt) &&
                      (rel_humidity <= modes(mode_idx).deliquescence_pt);
  //  case 3: nearly saturated air
  const auto rh_high = (rel_humidity > modes(mode_idx).deliquescence_pt);
  //  case 4: particles too small
  const auto too_small =
      (0.5 * to_microns * diags.dry_geometric_mean_diameter_i[mode_idx](k) <
       rdry_min);

  // no water uptake occurs if particles are too small or if air is too dry
  if (rh_low || too_small) {
    wet_diam = diags.dry_geometric_mean_diameter_i[mode_idx](k);
  } else {
    // for all other cases, we need a Kohler polynomial solve

    // convert from diameter in meters to radius in microns
    const Real dry_radius_microns =
        0.5 * to_microns * diags.dry_geometric_mean_diameter_i[mode_idx](k);

    // check dry particle size is in bounds
    EKAT_KERNEL_ASSERT((dry_radius_microns <= rdry_max));

    // Set up Kohler solver
    // (requires double precision)
    typedef KohlerSolver<haero::math::NewtonSolver<KohlerPolynomial>>
        SolverType;
    const Real tol = solver_convergence_tol;
    // Solve for the roots of the Kohler polynomial
    //
    //  This step replaces the mam4 subroutine modal_aero_kohler with
    //  a new solver that is better conditioned and stable for
    //  finite precision computations.
    SolverType kohler_solver =
        SolverType(rel_humidity, diags.hygroscopicity[mode_idx](k),
                   dry_radius_microns, tol);
    auto rwet_microns = kohler_solver.solve();

    // set maximum wet radius of 30 microns
    //
    //  from modal_aero_wateruptake.F90 line 721, this upper bound is
    //  ! based on 1 day lifetime
    const Real rwet_max = wet_radius_max_microns;
    if (rwet_microns > rwet_max) { // too big
      rwet_microns = rwet_max;
    }

    // Compute wet and dry particle volume in cubic meters
    //
    //  This volume calculation represents a bug fix from legacy Mam4.
    //  In subroutine modal_aero_wateruptake_sub from file
    //  modal_aero_wateruptake.F90, particle volumes are computed using the
    //  spherical geometric formulas without accounting for the probability
    //  density function (PDF) that represents the modal particle size
    //  distribution, which is an inconsistency: the
    //  dry_geometric_mean_diameter_i input accounts for the PDF while the same
    //  quantity for wet particles does not.
    //
    //  Here, we use the PDF functions for both.
    const Real dry_vol = conversions::mean_particle_volume_from_diameter(
        diags.dry_geometric_mean_diameter_i[mode_idx](k),
        modes(mode_idx).mean_std_dev);

    Real wet_vol = conversions::mean_particle_volume_from_diameter(
        2 * to_meters * rwet_microns, modes(mode_idx).mean_std_dev);

    // check that wet particle volume >= dry particle volume
    EKAT_KERNEL_ASSERT(wet_vol >= dry_vol);
    // which implies that water volume is nonnegative
    Real water_vol = wet_vol - dry_vol;

    // apply hysteresis to intermediate relative humidity values by
    // adjusting water volume
    //
    // comments from modal_aero_wateruptake.F90, lines 549--551,
    // subroutine modal_aero_wateruptake_sub:
    //   ! apply simple treatment of deliquesence/crystallization hysteresis
    //   ! for rhcrystal < rh < rhdeliques, aerosol water is a fraction of
    //   ! the "upper curve" value, and the fraction is a linear function of rh
    const Real hysteresis_factor = 1 / (modes(mode_idx).deliquescence_pt -
                                        modes(mode_idx).crystallization_pt);
    EKAT_KERNEL_ASSERT(hysteresis_factor > 0);
    if (rh_mid) {
      water_vol = hysteresis_factor * water_vol *
                  (rel_humidity - modes(mode_idx).crystallization_pt);
    }

    // check that hysteresis does not cause negative water content
    EKAT_KERNEL_ASSERT(water_vol >= 0);

    wet_vol = dry_vol + water_vol;

    const Real rwet_hyst =
        0.5 * conversions::mean_particle_diameter_from_volume(
                  wet_vol, modes(mode_idx).mean_std_dev);

    if (rh_mid) {
      wet_diam = 2 * rwet_hyst;
    } else if (rh_high) {
      wet_diam = 2 * to_meters * rwet_microns;
    }
  }

  diags.wet_geometric_mean_diameter_i[mode_idx](k) = wet_diam;
}

///  Compute aerosol particle wet diameter for interstitial aerosols
///  in all modes.
///
///  This version can be called in parallel across column vertical levels.
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
///  @param [in] k column vertical levelindex
KOKKOS_INLINE_FUNCTION
void mode_avg_wet_particle_diam_water_uptake(const Diagnostics &diags,
                                             const Atmosphere &atm, int k) {
  for (int m = 0; m < AeroConfig::num_modes(); ++m) {
    mode_avg_wet_particle_diam_water_uptake(diags, atm, m, k);
  }
}

// ------------------------------------------------------------------------
//  Subroutine for calculating wet geometric mean diameter from
//  the aerosol mass and number concentrations, using a prescribed
//  wet-to-dry diameter ratio.
//
//  History:
//    Original code by Richard (Dick) C. Easter, PNNL
//    Ported to driver by Qiyang Yan and Hui Wan, PNNL, 2022.
//
//    @param qaer_cur [in] current aerosol mass mix ratios (mol/mol)
//    @param qnum_cur [in] current aerosol number mix ratios (#/kmol)
//    @param dwet_ddry_ratio [in] ratio between wet and dry diameters
//    @param dgn_awet [out] geometric mean diameter (m) of each aerosol mode
// ------------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION
void diag_dgn_wet(
    const Real qaer_cur[mam4::AeroConfig::num_aerosol_ids()]
                       [mam4::AeroConfig::num_modes()],
    const Real qnum_cur[mam4::AeroConfig::num_modes()],
    const Real molecular_weight_gm[mam4::AeroConfig::num_aerosol_ids()],
    const Real dwet_ddry_ratio, Real dgn_awet[mam4::AeroConfig::num_modes()]) {
  static constexpr int num_aer = mam4::AeroConfig::num_aerosol_ids();
  static constexpr int num_modes = mam4::AeroConfig::num_modes();
  // --------------------------
  //  Calculation
  // --------------------------
  for (int n = 0; n < num_modes; ++n) {
    Real tmp_dryvol = 0.0;
    // Sum up the volume of all species in this mode
    for (int iaer = 0; iaer < num_aer; ++iaer) {
      const Real weight_gm_per_mol = molecular_weight_gm[iaer];
      const Real tmpa = qaer_cur[iaer][n] * weight_gm_per_mol;
      tmp_dryvol += tmpa / mam4::aero_species(iaer).density;
    }
    // Convert dry volume to dry diameter, then to wet diameter
    const Real sx = std::log(mam4::modes(n).mean_std_dev);
    const Real tmpb =
        tmp_dryvol / haero::max(1.0e-30, qnum_cur[n] * (Constants::pi / 6.0) *
                                             exp(4.5 * sx * sx));
    dgn_awet[n] = pow(tmpb, (1.0 / 3.0) * dwet_ddry_ratio);
  }
}

} // namespace mam4
#endif
