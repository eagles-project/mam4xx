#ifndef MAM4XX_NDROP_HPP
#define MAM4XX_NDROP_HPP

#include <haero/aero_species.hpp>
#include <haero/constants.hpp>
#include <haero/atmosphere.hpp>
#include <haero/math.hpp>

#include <mam4xx/aero_config.hpp>
#include <mam4xx/conversions.hpp>
#include <mam4xx/mam4_types.hpp>
#include <mam4xx/utils.hpp>

using Real = haero::Real;

namespace mam4 {

namespace ndrop {

  KOKKOS_INLINE_FUNCTION
  void get_aer_num(const Diagnostics &diags,
                                  const Prognostics &progs, int mode_idx, int k, Real naerosol[AeroConfig::num_aerosol_ids()]) {
    //out
    Real naerosol = 0.0; // number concentration [#/m3]


    for (int aid = 0; aid < AeroConfig::num_aerosol_ids(); ++aid) {
      const int s = aerosol_index_for_mode(static_cast<ModeIndex>(mode_idx),
                                          static_cast<AeroId>(aid));

      if(s >= 0) {

          Real rho = conversions::density_of_ideal_gas(haero::Atmosphere.temperature[s], haero::Atmosphere.pressure[s]);
          Real vaerosol = haero::AeroSpecies.density;
          naerosol[s] = (progs.q_aero_i[mode_idx][s](k) + progs.q_aero_c[mode_idx][s](k)) * rho;
          //adjust number so that dgnumlo < dgnum < dgnumhi
          
          Real min_diameter = modes(mode_idx).min_diameter;
          Real max_diameter = modes(mode_idx).max_diameter;
          Real mean_std_dev = modes(mode_idx).mean_std_dev;

          num2vol_ratio_min = 1.0 / conversions::mean_particle_volume_from_diameter(min_diameter, mean_std_dev);
          num2vol_ratio_max = 1.0 / conversions::mean_particle_volume_from_diameter(max_diameter, mean_std_dev);

          naerosol[s] = min(naerosol[s], vaerosol*num2vol_ratio_min);
          naerosol[s] = max(naerosol[s], vaerosol*num2vol_ratio_max);
      }

    }
  }
}
}