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

#include <sstream>

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
//   if ( (rel_humidity > 0.8).any() ) {
    std::ostringstream ss;
    ss << "T = " << atm.temperature(pack_idx) << " P = " << atm.pressure(pack_idx) << " qv = " << atm.vapor_mixing_ratio(pack_idx) << " rh = " << rel_humidity << "\n";
    std::cout << ss.str();
//   }
  if (!FloatingPoint<PackType>::in_bounds(rel_humidity, 0, 1)) {
    std::ostringstream ss;
    ss << "error: relative humidity = " << rel_humidity << "\n";
    ss << "\t" << "qv = " << atm.vapor_mixing_ratio(pack_idx) << "\n";
    ss << "\t" << "p  = " << atm.pressure(pack_idx) << "\n";
    ss << "\t" << "T  = " << atm.temperature(pack_idx) << "\n";
    std::cout << ss.str();
  }
  EKAT_KERNEL_ASSERT(FloatingPoint<PackType>::in_bounds(rel_humidity, 0, 1));

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

    const PackType dry_radius_microns = 0.5*to_microns*diags.dry_geometric_mean_diameter[mode_idx](pack_idx);

    // Kohler polynomial solve requires double precision
    typedef ekat::Pack<double, haero::HAERO_PACK_SIZE> DoublePack;
    typedef KohlerPolynomial<DoublePack> PolynomialType;
    typedef KohlerSolver<haero::math::NewtonSolver<PolynomialType>> SolverType;
    const Real tol = solver_convergence_tol;

    auto kohler_solver = SolverType(rel_humidity, diags.hygroscopicity[mode_idx](pack_idx),
      dry_radius_microns, tol);
    PackType rwet_microns = kohler_solver.solve();

    wet_diam.set(needs_kohler, 2*to_meters*rwet_microns);

    // Apply hysteresis to intermediate relative humidities
    const PackType dry_vol = conversions::mean_particle_volume_from_diameter(
      diags.dry_geometric_mean_diameter[mode_idx](pack_idx),
      modes[mode_idx].mean_std_dev);

    PackType wet_vol = conversions::mean_particle_volume_from_diameter(
      2*to_meters*rwet_microns, modes[mode_idx].mean_std_dev);

    const Real hysteresis_factor =
      1 / (modes[mode_idx].deliquescence_pt - modes[mode_idx].crystallization_pt);

//     EKAT_KERNEL_ASSERT( (wet_vol >= dry_vol).all() );
    if ( (wet_vol < dry_vol).any() ) {
      std::ostringstream ss;
      ss << "error: wet_vol = " << wet_vol << " dry_vol = " << dry_vol << "\n";
      ss << "\t" << "hygroscopicity = " << diags.hygroscopicity[mode_idx](pack_idx) << "\n";
      ss << "\t" << "dry_geometric_mean_diameter = " << diags.dry_geometric_mean_diameter[mode_idx](pack_idx) << "\n";
      ss << "\t" << "dry_radius_microns = " << dry_radius_microns << "\n";
      ss << "\t" << "relative humidity = " << rel_humidity << "\n";
      ss << "\t" << "rwet_microns = " << rwet_microns << "\n";
      std::cout << ss.str();
    }

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

KOKKOS_INLINE_FUNCTION
void mode_avg_wet_particle_diam(const Diagnostics& diags, const Prognostics& progs,
  const Atmosphere& atm, const int pack_idx) {
  for (int m=0; m<AeroConfig::num_modes(); ++m) {
    mode_avg_wet_particle_diam(diags, progs, atm, m, pack_idx);
  }
}

} // namespace mam4
#endif
