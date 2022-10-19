#ifndef MAM4XX_DRY_PARTICLE_SIZE_HPP
#define MAM4XX_DRY_PARTICLE_SIZE_HPP

#include <mam4xx/aero_config.hpp>
#include <mam4xx/aero_modes.hpp>
#include <mam4xx/mam4_types.hpp>

namespace mam4 {

///  Compute the dry geometric mean particle size (volume and diameter)
///  from the log-normal size distribution for a single mode.
///
///  This version can be called in parallel over both modes and column packs.
///
///  Diags are marked 'const' because they need to be able to be captured
///  by value by a lambda.  The Views inside the Diags struct are const,
///  but the data contained by the Views can change.
///
///  @param [in/out] diags Diagnostics: output container for particle size data
///  @param [in] progs Prognostics contain mode number mixing ratios and
///      aerosol mass mixing ratios
///  @param [in] mode_idx Mode whose average size is needed
///  @param [in] pack_idx Column pack where size data are needed
KOKKOS_INLINE_FUNCTION
void mode_avg_dry_particle_diam(const Diagnostics &diags,
                                const Prognostics &progs, const int mode_idx,
                                const int pack_idx) {
  PackType volume_mixing_ratio(0); // [m3 aerosol / kg air]
  for (int aid = 0; aid < AeroConfig::num_aerosol_ids(); ++aid) {
    const int s = aerosol_index_for_mode(static_cast<ModeIndex>(mode_idx),
                                         static_cast<AeroId>(aid));
    if (s >= 0) {
      volume_mixing_ratio +=
          progs.q_aero_i[mode_idx][s](pack_idx) / aero_species[s].density;
    }
  }
  const PackType mean_vol =
      volume_mixing_ratio / progs.n_mode[mode_idx](pack_idx);
  diags.dry_geometric_mean_diameter[mode_idx](pack_idx) =
      conversions::mean_particle_diameter_from_volume(
          mean_vol, modes[mode_idx].mean_std_dev);
}

///  Compute the dry geometric mean particle size (volume and diameter)
///  from the log-normal size distribution for all modes.
///
///  This version can be called in parallel over column packs, and computes
///  all modal averages serially.
///
///  @param [in/out] diags Diagnostics: output container for particle size data
///  @param [in] progs Prognostics contain mode number mixing ratios and
///      aerosol mass mixing ratios
///  @param [in] pack_idx Column pack where size data are needed
KOKKOS_INLINE_FUNCTION
void mode_avg_dry_particle_diam(const Diagnostics &diags,
                                const Prognostics &progs, const int pack_idx) {
  for (int m = 0; m < AeroConfig::num_modes(); ++m) {
    mode_avg_dry_particle_diam(diags, progs, m, pack_idx);
  }
}

} // namespace mam4
#endif
