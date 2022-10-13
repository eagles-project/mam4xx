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

namespace mam4 {

static constexpr Real meters_to_microns = 1e6;
static constexpr Real microns_to_meters = 1e-6;
static constexpr Real dry_radius_min_microns =
  KohlerPolynomial<double>::dry_radius_min_microns;
static constexpr Real solver_convergence_tol = 1e-10;

KOKKOS_INLINE_FUNCTION
void mode_avg_wet_particle_diam(const Diagnostics& diags, const Prognostics& progs,
  const Atmosphere& atm, const int mode_idx, const int pack_idx) {

  const Real to_microns = meters_to_microns;
  const Real to_meters = microns_to_meters;
  const Real rdry_min = dry_radius_min_microns;

  // initialize to dry diameter
  PackType wet_diam(diags.dry_geometric_mean_diameter[mode_idx](pack_idx));

  // compute relative humidity
  PackType rel_humidity = conversions::relative_humidity_from_vapor_mixing_ratio(
    atm.vapor_mixing_ratio(pack_idx), atm.pressure(pack_idx),
    atm.temperature(pack_idx));

  // set masks:
  //  case 1: dry air
  const auto rh_too_low = (rel_humidity <= modes[mode_idx].crystallization_pt);
  //  case 2: unsaturated air
  const auto rh_mid = (rel_humidity > modes[mode_idx].crystallization_pt) &&
                      (rel_humidity <= modes[mode_idx].deliquescence_pt);
  //  case 3: particles too small
  const auto too_small =
    (to_microns * diags.dry_geometric_mean_diameter[mode_idx](pack_idx) <
    2 * rdry_min);

  const auto use_dry_radius = (rh_too_low || too_small);
  const auto needs_kohler = !use_dry_radius;
  if (needs_kohler.any()) {

    // Kohler polynomial solve requires double precision
    typedef ekat::Pack<double, HAERO_PACK_SIZE> DoublePack;
    typedef KohlerPolynomial<DoublePack> PolynomialType;
    typedef KohlerSolver<haero::math::NewtonSolver<PolynomialType>> SolverType;
    const Real tol = solver_convergence_tol;

    const auto kpoly = PolynomialType(needs_kohler,
      rel_humidity, diags.hygroscopicity(pack_idx),
      0.5*to_microns*diags.dry_geometric_mean_diameter[mode_idx](pack_idx));

    auto solver = SolverType(rel_humidity, diags.hygroscopicity(pack_idx),
      0.5*to_microns*diags.dry_geometric_mean_diameter[mode_idx](pack_idx),
      tol);
    PackType rwet_microns = solver.solve();

    wet_diam.set(needs_kohler, 2*to_meters*rwet_microns);

    // Apply hysteresis to intermediate relative humidities
    const PackType dry_vol = conversions::mean_particle_volume_from_diameter(
      diags.dry_geometric_mean_diameter[mode_idx](pack_idx),
      modes[m].mean_std_dev);

    PackType wet_vol = conversions::mean_particle_volume_from_diameter(
      2*to_meters*rwet_microns, modes[m].mean_std_dev);

    const Real hysteresis_factor =
      1 / (modes[mode_idx].deliquescence_pt - modes[mode_idx].crystallization_pt);

    EKAT_KERNEL_ASSERT( (wet_vol >= dry_vol).all() );

    const PackType water_vol = (wet_vol - dry_vol) *
      (rel_humidity - modes[mode_idx].crystallization_pt) * hysteresis_factor;

    wet_vol = dry_vol + water_vol;

    const PackType rwet_hyst =
      0.5 * conversions::mean_particle_diameter_from_volume(
        wet_vol, modes[mode_idx].mean_std_dev);

    wet_diam.set(rh_mid, rwet_hyst);
  }

  diags.wet_geometric_mean_diameter[mode_idx](pack_idx) = wet_diam;
}

} // namespace mam4
#endif
