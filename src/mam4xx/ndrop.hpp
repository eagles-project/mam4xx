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

class NDrop {

public:
};

namespace ndrop {

// TODO: this function signature may need to change to work properly on GPU
//  come back when this function is being used in a ported parameterization
KOKKOS_INLINE_FUNCTION
void get_aer_num(const Diagnostics &diags, const Prognostics &progs,
                 const Atmosphere &atm, const int mode_idx, const int k,
                 Real naerosol[AeroConfig::num_modes()]) {

  Real rho =
      conversions::density_of_ideal_gas(atm.temperature(k), atm.pressure(k));
  Real vaerosol = conversions::mean_particle_volume_from_diameter(
      diags.dry_geometric_mean_diameter_total[mode_idx](k),
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
      (progs.n_mode_i[mode_idx](k) + progs.n_mode_c[mode_idx](k)) * rho;

  // adjust number so that dgnumlo < dgnum < dgnumhi
  naerosol[mode_idx] = max(naerosol[mode_idx], vaerosol * num2vol_ratio_max);
  naerosol[mode_idx] = min(naerosol[mode_idx], vaerosol * num2vol_ratio_min);
}

KOKKOS_INLINE_FUNCTION
void explmix(
    const ThreadTeam &team, // ThreadTeam for parallel_for
    int nlev,               // number of levels
    ColumnView q,    // number / mass mixing ratio to be updated [# or kg / kg]
    ColumnView src,  // source due to activation/nucleation [# or kg / (kg-s)]
    ColumnView ekkp, // zn*zs*density*diffusivity (kg/m3 m2/s) at interface
                     // [/s]; below layer k  (k,k+1 interface)
    ColumnView ekkm, // zn*zs*density*diffusivity (kg/m3 m2/s) at interface
                     // [/s]; above layer k  (k,k+1 interface)
    ColumnView overlapp, // cloud overlap below [fraction]
    ColumnView overlapm, // cloud overlap above [fraction]
    ColumnView qold, // number / mass mixing ratio from previous time step [# or
                     // kg / kg]
    Real dt,         // time step [s]
    bool is_unact,   // true if this is an unactivated species
    ColumnView
        qactold // optional: number / mass mixing ratio of ACTIVATED species
                // from previous step *** this should only be present if the
                // current species is unactivated number/sfc/mass
) {

  int top_lev = 0;

  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team, nlev), KOKKOS_LAMBDA(int k) {
        int kp1 = min(k + 1, nlev - 1);
        int km1 = max(k - 1, top_lev);

        // the qactold*(1-overlap) terms are resuspension of activated material

        if (is_unact) {
          q(k) = qold(k) +
                 (dt * (-src(k) +
                        (ekkp(k) * (qold(kp1) - qold(k) +
                                    (qactold(kp1) * (1 - overlapp(k))))) +
                        (ekkm(k) * (qold(km1) - qold(k) +
                                    (qactold(km1) * (1 - overlapm(k)))))));
        } else {
          q(k) = qold(k) +
                 (dt *
                  (src(k) + (ekkp(k) * ((overlapp(k) * qold(kp1)) - qold(k))) +
                   (ekkm(k) * ((overlapm(k) * qold(k)) - qold(k)))));
        }
        // force to non-negative
        q(k) = max(q(k), 0);
      });
}
} // namespace ndrop
} // namespace mam4
#endif
