// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MAM4XX_DRY_PARTICLE_SIZE_HPP
#define MAM4XX_DRY_PARTICLE_SIZE_HPP

#include <mam4xx/aero_config.hpp>
#include <mam4xx/aero_modes.hpp>
#include <mam4xx/mam4_types.hpp>

namespace mam4 {

///  Compute the dry geometric mean particle size (volume and diameter)
///  from the log-normal size distribution for a single mode,
///  both interstitial and cloudeborne aerosols, separately,
///  as well as the total (interstitial + cloudborne).
///
///  This version can be called in parallel over both modes and vertical levels.
///
///  Diags are marked 'const' because they need to be able to be captured
///  by value by a lambda.  The Views inside the Diags struct are const,
///  but the data contained by the Views can change.
///
///  @param [in/out] diags Diagnostics: output container for particle size data
///  @param [in] progs Prognostics contain mode number mixing ratios and
///      aerosol mass mixing ratios
///  @param [in] mode_idx Mode whose average size is needed
///  @param [in] k Column vertical level where size data are needed
KOKKOS_INLINE_FUNCTION
void mode_avg_dry_particle_diam(const Diagnostics &diags,
                                const Prognostics &progs, int mode_idx, int k) {
  Real volume_mixing_ratio_i = 0.0; // [m3 aerosol / kg air]
  Real volume_mixing_ratio_c = 0.0; // [m3 aerosol / kg air]
  for (int aid = 0; aid < AeroConfig::num_aerosol_ids(); ++aid) {
    const int s = aerosol_index_for_mode(static_cast<ModeIndex>(mode_idx),
                                         static_cast<AeroId>(aid));
    if (s >= 0) {
      volume_mixing_ratio_i +=
          progs.q_aero_i[mode_idx][s](k) / aero_species(s).density;

      volume_mixing_ratio_c +=
          progs.q_aero_c[mode_idx][s](k) / aero_species(s).density;
    }
  }
  const Real mean_vol_i = volume_mixing_ratio_i / progs.n_mode_i[mode_idx](k);
  const Real mean_vol_c = volume_mixing_ratio_c / progs.n_mode_c[mode_idx](k);
  diags.dry_geometric_mean_diameter_i[mode_idx](k) =
      conversions::mean_particle_diameter_from_volume(
          mean_vol_i, modes(mode_idx).mean_std_dev);
  diags.dry_geometric_mean_diameter_c[mode_idx](k) =
      conversions::mean_particle_diameter_from_volume(
          mean_vol_c, modes(mode_idx).mean_std_dev);
  diags.dry_geometric_mean_diameter_total[mode_idx](k) =
      conversions::mean_particle_diameter_from_volume(
          mean_vol_c + mean_vol_i, modes(mode_idx).mean_std_dev);
}

///  Compute the dry geometric mean particle size (volume and diameter)
///  from the log-normal size distribution for all modes.
///
///  This version can be called in parallel over vertical levels, and computes
///  all modal averages serially.
///
///  @param [in/out] diags Diagnostics: output container for particle size data
///  @param [in] progs Prognostics contain mode number mixing ratios and
///      aerosol mass mixing ratios
///  @param [in] k Column vertical level where size data are needed
KOKKOS_INLINE_FUNCTION
void mode_avg_dry_particle_diam(const Diagnostics &diags,
                                const Prognostics &progs, int k) {
  for (int m = 0; m < AeroConfig::num_modes(); ++m) {
    mode_avg_dry_particle_diam(diags, progs, m, k);
  }
}

} // namespace mam4
#endif
