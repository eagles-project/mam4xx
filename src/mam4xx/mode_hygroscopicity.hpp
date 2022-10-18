#ifndef MAM4XX_HYGROSCOPICITY_HPP
#define MAM4XX_HYGROSCOPICITY_HPP

#include "aero_config.hpp"
#include "aero_modes.hpp"
#include "mam4_types.hpp"

namespace mam4 {

/**  Compute the modal average hygroscopicity.

  This version can be called in parallel over both modes and column packs.

  Diags are marked 'const' because they need to be able to be captured
  by value by a lambda.  The Views inside the Diags struct are const,
  but the data contained by the Views can change.

  Equation (A2) from Ghan et al., 2011, Droplet nucleation: Physically-based
   parameterizations and comparative evaluation, J. Adv. Earth Sys. Mod. 3
   M10001.

  Note that equation (A3) from that paper, which sets the hygroscopicity
  value for each species, is not used by MAM4 (whose values of hygroscopicity
  are set explicitly -- see aero_modes.hpp).

  @param [in/out] diags Diagnostics: output container for hygroscopicity data
  @param [in] progs Prognostics contain mode number mixing ratios and
      aerosol mass mixing ratios
  @param [in] mode_idx Mode whose average hygroscopicity is needed
  @param [in] pack_idx Column pack where size data are needed
*/
void mode_hygroscopicity(const Diagnostics &diags, const Prognostics &progs,
                         const int mode_idx, const int pack_idx) {
  PackType hyg(0);
  PackType volume_mixing_ratio(0); // [m3 aerosol / kg air]
  for (int aid = 0; aid < AeroConfig::num_aerosol_ids(); ++aid) {
    const int s = aerosol_index_for_mode(static_cast<ModeIndex>(mode_idx),
                                         static_cast<AeroId>(aid));
    if (s >= 0) {
      const PackType mass_mix_ratio = progs.q_aero_i[mode_idx][s](pack_idx);
      volume_mixing_ratio += mass_mix_ratio / aero_species[s].density;
      hyg += mass_mix_ratio * aero_species[s].hygroscopicity /
             aero_species[s].density;
    }
    diags.hygroscopicity[mode_idx](pack_idx) = hyg / volume_mixing_ratio;
  }
}

/**  Compute the modal average hygroscopicity.

  This version can be called in parallel over column packs.

  Diags are marked 'const' because they need to be able to be captured
  by value by a lambda.  The Views inside the Diags struct are const,
  but the data contained by the Views can change.

  @param [in/out] diags Diagnostics: output container for hygroscopicity data
  @param [in] progs Prognostics contain mode number mixing ratios and
      aerosol mass mixing ratios
  @param [in] pack_idx Column pack where size data are needed
*/
void mode_hygroscopicity(const Diagnostics &diags, const Prognostics &progs,
                         const int pack_idx) {
  for (int m = 0; m < AeroConfig::num_modes(); ++m) {
    mode_hygroscopicity(diags, progs, m, pack_idx);
  }
}

} // namespace mam4
#endif
