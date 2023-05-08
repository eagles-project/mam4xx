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

// get_aer_num is being refactored by oscar in collab/ndrop
KOKKOS_INLINE_FUNCTION
void get_aer_num(const int voltonumbhi_amode, const int voltonumblo_amode,
                 const int num_idx, const Real state_q[7],
                 const Real air_density, const Real vaerosol,
                 const Real qcldbrn1d_num, Real &naerosol) {}

KOKKOS_INLINE_FUNCTION
void explmix(
    const Real qold_km1, // number / mass mixing ratio from previous time step
                         // at level k-1 [# or kg / kg]
    const Real qold_k, // number / mass mixing ratio from previous time step at
                       // level k [# or kg / kg]
    const Real qold_kp1, // number / mass mixing ratio from previous time step
                         // at level k+1 [# or kg / kg]
    Real &q, // OUTPUT, number / mass mixing ratio to be updated [# or kg / kg]
    const Real src, // source due to activation/nucleation at level k [# or kg /
                    // (kg-s)]
    const Real ek_kp1, // zn*zs*density*diffusivity (kg/m3 m2/s) at interface
                       // [/s]; below layer k  (k,k+1 interface)
    const Real ek_km1, // zn*zs*density*diffusivity (kg/m3 m2/s) at interface
                       // [/s]; above layer k  (k,k+1 interface)
    const Real overlap_kp1, // cloud overlap below [fraction]
    const Real overlap_km1, // cloud overlap above [fraction]
    const Real dt,          // time step [s]
    const bool is_unact,    // true if this is an unactivated species
    const Real qactold_km1 =
        1, // optional: number / mass mixing ratio of ACTIVATED species
           // from previous step at level k-1 *** this should only be present if
           // the current species is unactivated number/sfc/mass
    const Real qactold_kp1 =
        1 // optional: number / mass mixing ratio of ACTIVATED species
          // from previous step at level k+1 *** this should only be present if
          // the current species is unactivated number/sfc/mass
) {

  // the qactold*(1-overlap) terms are resuspension of activated material

  if (is_unact) {
    q = qold_k +
        (dt *
         (-src +
          (ek_kp1 * (qold_kp1 - qold_k + (qactold_kp1 * (1 - overlap_kp1)))) +
          (ek_km1 * (qold_km1 - qold_k + (qactold_km1 * (1 - overlap_km1))))));
  } else {
    q = qold_k + (dt * (src + (ek_kp1 * ((overlap_kp1 * qold_kp1) - qold_k)) +
                        (ek_km1 * ((overlap_km1 * qold_k) - qold_k))));
  }
  // force to non-negative
  q = haero::max(q, 0);
} // end explmix

// calculates maximum supersaturation for multiple
// competing aerosol modes.
// Abdul-Razzak and Ghan, A parameterization of aerosol activation.
// 2. Multiple aerosol types. J. Geophys. Res., 105, 6837-6844.
KOKKOS_INLINE_FUNCTION
void maxsat(
    const Real zeta,                         // [dimensionless]
    const Real eta[AeroConfig::num_modes()], // [dimensionless]
    const Real nmode,                        // number of modes
    const Real smc[AeroConfig::num_modes()], // critical supersaturation for
                                             // number mode radius [fraction]
    Real &smax // maximum supersaturation [fraction] (output)
) {
  // abdul-razzak functions of width
  Real f1[AeroConfig::num_modes()];
  Real f2[AeroConfig::num_modes()];

  Real const small = 1e-20;     /*FIXME: BAD CONSTANT*/
  Real const mid = 1e5;         /*FIXME: BAD CONSTANT*/
  Real const big = 1.0 / small; /*FIXME: BAD CONSTANT*/
  Real sum = 0;
  Real g1, g2;
  bool weak_forcing = true; // whether forcing is sufficiently weak or not

  for (int m = 0; m < nmode; m++) {
    if (zeta > mid * eta[m] || smc[m] * smc[m] > mid * eta[m]) {
      // weak forcing. essentially none activated
      smax = small;
    } else {
      // significant activation of this mode. calc activation of all modes.
      weak_forcing = false;
      break;
    }
  }

  // if the forcing is weak, return
  if (weak_forcing)
    return;

  for (int m = 0; m < nmode; m++) {
    f1[m] = 0.5 * haero::exp(2.5 * haero::square(haero::log(modes(m).mean_std_dev))); 
    f2[m] = 1.0 + 0.25 * haero::log(modes(m).mean_std_dev);
    if (eta[m] > small) {
      g1 = (zeta / eta[m]) * haero::sqrt(zeta / eta[m]);
      g2 = (smc[m] / haero::sqrt(eta[m] + 3.0 * zeta)) *
           haero::sqrt(smc[m] / haero::sqrt(eta[m] + 3.0 * zeta));
      sum += (f1[m] * g1 + f2[m] * g2) / (smc[m] * smc[m]);
    } else {
      sum = big; 
    }
  }
  smax = 1.0 / haero::sqrt(sum);
  return;
} // end maxsat

} // namespace ndrop
} // namespace mam4
#endif
