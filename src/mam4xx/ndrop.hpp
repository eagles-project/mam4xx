#ifndef MAM4XX_NDROP_HPP
#define MAM4XX_NDROP_HPP

#include <haero/aero_species.hpp>
#include <haero/atmosphere.hpp>
#include <haero/constants.hpp>
#include <haero/math.hpp>

#include <mam4xx/aero_config.hpp>
#include <mam4xx/conversions.hpp>
#include <mam4xx/mam4_types.hpp>
#include <mam4xx/utils.hpp>

using Real = haero::Real;

namespace mam4 {

KOKKOS_INLINE_FUNCTION
void get_aer_num(const Diagnostics &diags, const Prognostics &progs,
                 const Atmosphere &atm, int mode_idx, int icol,
                 Real naerosol[AeroConfig::num_modes()]) {

  Real rho = conversions::density_of_ideal_gas(atm.temperature(icol),
                                               atm.pressure(icol));

  Real vaerosol;

  vaerosol = conversions::mean_particle_volume_from_diameter(
      diags.dry_geometric_mean_diameter_total[mode_idx](icol),
      modes(mode_idx).mean_std_dev);

  Real min_diameter = modes(mode_idx).min_diameter;
  Real max_diameter = modes(mode_idx).max_diameter;
  Real mean_std_dev = modes(mode_idx).mean_std_dev;

  Real num2vol_ratio_min =
      1.0 / conversions::mean_particle_volume_from_diameter(min_diameter,
                                                            mean_std_dev);
  Real num2vol_ratio_max =
      1.0 / conversions::mean_particle_volume_from_diameter(max_diameter,
                                                            mean_std_dev);

  // convert number mixing ratios to number concentrations
  naerosol[mode_idx] =
      (progs.n_mode_i[mode_idx](icol) + progs.n_mode_c[mode_idx](icol)) * rho;

  // adjust number so that dgnumlo < dgnum < dgnumhi
  naerosol[mode_idx] = min(naerosol[mode_idx], vaerosol * num2vol_ratio_min);
  naerosol[mode_idx] = max(naerosol[mode_idx], vaerosol * num2vol_ratio_max);
}
} // namespace mam4
#endif