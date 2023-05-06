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

//get_aer_num is being refactored by oscar in collab/ndrop

KOKKOS_INLINE_FUNCTION
void explmix(
    const Real qold_km1, // number / mass mixing ratio from previous time step at level k-1 [# or
                     // kg / kg]
    const Real qold_k,  // number / mass mixing ratio from previous time step at level k [# or
                     // kg / kg]
    const Real qold_kp1,  // number / mass mixing ratio from previous time step at level k+1 [# or
                     // kg / kg]
    Real &q,    // OUTPUT, number / mass mixing ratio to be updated [# or kg / kg]
    const Real src,  // source due to activation/nucleation at level k [# or kg / (kg-s)]
    const Real ek_kp1, // zn*zs*density*diffusivity (kg/m3 m2/s) at interface 
                     // [/s]; below layer k  (k,k+1 interface)
    const Real ek_km1, // zn*zs*density*diffusivity (kg/m3 m2/s) at interface
                     // [/s]; above layer k  (k,k+1 interface)
    const Real overlap_kp1, // cloud overlap below [fraction]
    const Real overlap_km1, // cloud overlap above [fraction]
    const Real dt,         // time step [s]
    const bool is_unact,   // true if this is an unactivated species
    const Real 
        qactold_km1 = 1, // optional: number / mass mixing ratio of ACTIVATED species
                // from previous step at level k-1 *** this should only be present if the
                // current species is unactivated number/sfc/mass
    const Real 
        qactold_kp1 = 1 // optional: number / mass mixing ratio of ACTIVATED species
                // from previous step at level k+1 *** this should only be present if the
                // current species is unactivated number/sfc/mass
) {

  // the qactold*(1-overlap) terms are resuspension of activated material

  if (is_unact) {
    q = qold_k +
            (dt * (-src +
                  (ek_kp1 * (qold_kp1 - qold_k +
                              (qactold_kp1 * (1 - overlap_kp1)))) +
                  (ek_km1 * (qold_km1 - qold_k +
                              (qactold_km1 * (1 - overlap_km1))))));
  } else {
    q = qold_k +
            (dt *
            (src + (ek_kp1 * ((overlap_kp1 * qold_kp1) - qold_k)) +
              (ek_km1 * ((overlap_km1 * qold_k) - qold_k))));
  }
  // force to non-negative
  q = max(q, 0);
} // end explmix

} // namespace ndrop
} // namespace mam4
#endif
