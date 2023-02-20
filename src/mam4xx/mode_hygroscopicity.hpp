// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MAM4XX_HYGROSCOPICITY_HPP
#define MAM4XX_HYGROSCOPICITY_HPP

#include <mam4xx/aero_config.hpp>
#include <mam4xx/aero_modes.hpp>
#include <mam4xx/mam4_types.hpp>

namespace mam4 {

///  Compute the modal average hygroscopicity for a single mode's interstitial
///  aerosols using
///  Equation (A2) from Ghan et al., 2011, Droplet nucleation: Physically-based
///  parameterizations and comparative evaluation, J. Adv. Earth Sys. Mod. 3
///  M10001.
///
///  Note that equation (A3) from that paper, which sets the hygroscopicity
///  value for each species, is not used by MAM4 (whose species' hygroscopicity
///  are set explicitly -- see aero_modes.hpp).
///
///  This version can be called in parallel over both modes and vertical levels.
///
///  Diags are marked 'const' because they need to be able to be captured
///  by value by a lambda.  The Views inside the Diags struct are const,
///  but the data contained by the Views can change.
///
///  @param [in/out] diags Diagnostics: output container for hygroscopicity data
///  @param [in] progs Prognostics contain mode number mixing ratios and
///      aerosol mass mixing ratios
///  @param [in] mode_idx Mode whose average hygroscopicity is needed
///  @param [in] k Column vertical level where size data are needed
KOKKOS_INLINE_FUNCTION
void mode_hygroscopicity_i(const Diagnostics &diags, const Prognostics &progs,
                           int mode_idx, int k) {
  Real hyg = 0.0;
  Real volume_mixing_ratio = 0.0; // [m3 aerosol / kg air]
  for (int aid = 0; aid < AeroConfig::num_aerosol_ids(); ++aid) {
    const int s = aerosol_index_for_mode(static_cast<ModeIndex>(mode_idx),
                                         static_cast<AeroId>(aid));
    if (s >= 0) {
      const Real mass_mix_ratio = progs.q_aero_i[mode_idx][s](k);
      volume_mixing_ratio += mass_mix_ratio / aero_species(s).density;
      hyg += mass_mix_ratio * aero_species(s).hygroscopicity /
             aero_species(s).density;
    }
    diags.hygroscopicity[mode_idx](k) = hyg / volume_mixing_ratio;
  }
}

///  Compute the modal average hygroscopicity for all modes.
///
///  This version can be called in parallel over vertical levels.
///
///  Diags are marked 'const' because they need to be able to be captured
///  by value by a lambda.  The Views inside the Diags struct are const,
///  but the data contained by the Views can change.
///
///  @param [in/out] diags Diagnostics: output container for hygroscopicity data
///  @param [in] progs Prognostics contain mode number mixing ratios and
///      aerosol mass mixing ratios
///  @param [in] k Column vertical level where size data are needed
KOKKOS_INLINE_FUNCTION
void mode_hygroscopicity(const Diagnostics &diags, const Prognostics &progs,
                         int k) {
  for (int m = 0; m < AeroConfig::num_modes(); ++m) {
    mode_hygroscopicity_i(diags, progs, m, k);
  }
}

} // namespace mam4
#endif
