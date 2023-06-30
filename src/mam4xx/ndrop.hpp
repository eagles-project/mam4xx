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

// FIXME; surften is defined in ndrop_init
// BAD CONSTANT
const int ncnst_tot = 25;
const int nspec_max = 8;

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
    Real &
        qnew, // OUTPUT, number / mass mixing ratio to be updated [# or kg / kg]
    const Real src, // source due to activation/nucleation at level k [# or kg /
                    // (kg-s)]
    const Real ekkp,     // zn*zs*density*diffusivity (kg/m3 m2/s) at interface
                         // [/s]; below layer k  (k,k+1 interface)
    const Real ekkm,     // zn*zs*density*diffusivity (kg/m3 m2/s) at interface
                         // [/s]; above layer k  (k,k+1 interface)
    const Real overlapp, // cloud overlap below [fraction]
    const Real overlapm, // cloud overlap above [fraction]
    const Real dtmix     // time step [s]
) {

  qnew = qold_k + dtmix * (src + ekkp * (overlapp * qold_kp1 - qold_k) +
                           ekkm * (overlapm * qold_km1 - qold_k));

  // force to non-negative
  qnew = haero::max(qnew, 0);

} // end explmix

KOKKOS_INLINE_FUNCTION
void explmix(
    const Real qold_km1, // number / mass mixing ratio from previous time step
                         // at level k-1 [# or kg / kg]
    const Real qold_k, // number / mass mixing ratio from previous time step at
                       // level k [# or kg / kg]
    const Real qold_kp1, // number / mass mixing ratio from previous time step
                         // at level k+1 [# or kg / kg]
    Real &
        qnew, // OUTPUT, number / mass mixing ratio to be updated [# or kg / kg]
    const Real src, // source due to activation/nucleation at level k [# or kg /
                    // (kg-s)]
    const Real ekkp,     // zn*zs*density*diffusivity (kg/m3 m2/s) at interface
                         // [/s]; below layer k  (k,k+1 interface)
    const Real ekkm,     // zn*zs*density*diffusivity (kg/m3 m2/s) at interface
                         // [/s]; above layer k  (k,k+1 interface)
    const Real overlapp, // cloud overlap below [fraction]
    const Real overlapm, // cloud overlap above [fraction]
    const Real dtmix,    // time step [s]
    const Real qactold_km1,
    // optional: number / mass mixing ratio of ACTIVATED species
    // from previous step at level k-1 *** this should only be present if
    // the current species is unactivated number/sfc/mass
    const Real qactold_kp1
    // optional: number / mass mixing ratio of ACTIVATED species
    // from previous step at level k+1 *** this should only be present if
    // the current species is unactivated number/sfc/mass
) {

  // the qactold*(1-overlap) terms are resuspension of activated material
  const Real one = 1.0;
  qnew = qold_k +
         dtmix * (-src +
                  ekkp * (qold_kp1 - qold_k + qactold_kp1 * (one - overlapp)) +
                  ekkm * (qold_km1 - qold_k + qactold_km1 * (one - overlapm)));

  // force to non-negative
  qnew = haero::max(qnew, 0);
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
    f1[m] = 0.5 *
            haero::exp(2.5 * haero::square(haero::log(modes(m).mean_std_dev)));
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

KOKKOS_INLINE_FUNCTION
void update_from_explmix(
    const Real dtmicro, // time step for microphysics [s]
    int top_lev,        // top level
    int pver,           // number of levels
    ColumnView csbot,   // air density at bottom (interface) of layer [kg/m^3]
    ColumnView cldn,    // cloud fraction [fraction]
    ColumnView zn,      // g/pdel for layer [m^2/kg]
    ColumnView zs,      // inverse of distance between levels [m^-1]
    ColumnView ekd,     // diffusivity for droplets [m^2/s]
    ColumnView nact[AeroConfig::num_modes()], // fractional aero. number
                                              // activation rate [/s]
    ColumnView mact[AeroConfig::num_modes()], // fractional aero. mass
                                              // activation rate [/s]
    ColumnView qcld, // cloud droplet number mixing ratio [#/kg]
    ColumnView raercol[2][ncnst_tot],    // single column of saved aerosol mass,
                                         // number mixing ratios [#/kg or kg/kg]
    ColumnView raercol_cw[2][ncnst_tot], // same as raercol but for cloud-borne
                                         // phase [#/kg or kg/kg]
    int &nsav, // indices for old, new time levels in substepping
    int &nnew, // indices for old, new time levels in substepping
    const int nspec_amode[AeroConfig::num_modes()],
    const int mam_idx[AeroConfig::num_modes()][nspec_max],
    // work vars
    ColumnView overlapp, // cloud overlap involving level kk+1 [fraction]
    ColumnView overlapm, // cloud overlap involving level kk-1 [fraction]
    ColumnView ekk, ColumnView ekkp, ColumnView ekkm, ColumnView qncld,
    ColumnView srcn,  // droplet source rate [/s]
    ColumnView source //  source rate for activated number or species mass [/s]
) {

  // local arguments
  // int mm;          // local array index for MAM number, species
  // int nsubmix;//, nsubmix_bnd;  // number of substeps and bound
  // int ntemp;   // temporary index for substepping
  // int kp1;    // current level + 1
  // int km1;    // current level -1

  const Real overlap_cld_thresh =
      1e-10; //  threshold cloud fraction to compute overlap [fraction]
  const Real zero = 0.0;
  // Real ekk[pver+1];       // density*diffusivity for droplets [kg/m/s]
  // Real dtmin;     // time step to determine subloop time step [s]
  // Real qncld[pver];     // updated cloud droplet number mixing ratio [#/kg]
  // Real ekkp[pver];      // zn*zs*density*diffusivity [/s]
  // Real ekkm[pver];      // zn*zs*density*diffusivity   [/s]
  // Real source[pver];    //  source rate for activated number or species mass
  // [/s] Real tinv;      // inverse timescale of droplet diffusivity [/s] Real
  // dtt;       // timescale of droplet diffusivity [s] Real dtmix;     //
  // timescale for subloop [s]
  Real tmpa = zero; //  temporary aerosol tendency variable [/s]
  // Real srcn[pver];      // droplet source rate [/s]

  const int ntot_amode = AeroConfig::num_modes();

  // load new droplets in layers above, below clouds
  Real dtmin = dtmicro;
  ekk(top_lev - 1) = zero;
  ekk(pver) = zero;
  // rce-comment -- ekd(k) is eddy-diffusivity at k/k+1 interface
  //   want ekk(k) = ekd(k) * (density at k/k+1 interface)
  //   so use pint(i,k+1) as pint is 1:pverp
  //           ekk(k)=ekd(k)*2.*pint(i,k)/(rair*(temp(i,k)+temp(i,k+1)))
  //           ekk(k)=ekd(k)*2.*pint(i,k+1)/(rair*(temp(i,k)+temp(i,k+1)))
  for (int k = top_lev; k < pver - 1; k++) {
    ekk(k) = ekd(k) * csbot(k);
  }

  // start k for loop here. for k = top_lev to pver
  // cldn will be columnviews of length pver,
  // overlaps also to columnview pass as parameter so it is allocated elsewhwere
  for (int k = top_lev; k < pver; k++) {
    const int kp1 = haero::min(k + 1, pver);
    const int km1 = haero::max(k - 1, top_lev);
    // maximum overlap assumption
    if (cldn(kp1) > overlap_cld_thresh) {
      overlapp(k) = haero::min(cldn(k) / cldn(kp1), 1.0);
    } else {
      overlapp(k) = 1.0;
    }

    if (cldn(km1) > overlap_cld_thresh) {
      overlapm(k) = haero::min(cldn(k) / cldn(km1), 1.0);
    } else {
      overlapm(k) = 1.0;
    }

    ekkp(k) = zn(k) * ekk(k) * zs(k);
    ekkm(k) = zn(k) * ekk(km1) * zs(km1);
    const int tinv = ekkp(k) + ekkm(k);

    // rce-comment -- tinv is the sum of all first-order-loss-rates
    //    for the layer.  for most layers, the activation loss rate
    //    (for interstitial particles) is accounted for by the loss by
    //    turb-transfer to the layer above.
    //    k=pver is special, and the loss rate for activation within
    //    the layer must be added to tinv.  if not, the time step
    //    can be too big, and explmix can produce negative values.
    //    the negative values are reset to zero, resulting in an
    //    artificial source.

    // FIXME: BAD CONSTANT
    if (tinv > 1e-6) {
      dtmin = haero::min(dtmin, 1.0 / tinv);
    }
  }
  // TODO
  //    fix dtmin section
  //    pass arrs as columnviews and k len instead of single values
  //    loop over k for calls to explmis
  //    for things like src, qcld, loop over to assign the values
  //    pass overlapp etc as params as work vars so they are allocated elsewhere

  // timescale for subloop [s]
  //  BAD CONSTANT
  Real dtmix = 0.9 * dtmin;
  // number of substeps and bound
  const int nsubmix = dtmicro / dtmix + 1;
  // FIXME: nsubmix_bnd is used in the code. Ask Fortran team.
  //  re: nsubmix_bnd isn't used in the code?
  //  if (nsubmix > 100) {
  //     nsubmix_bnd = 100;
  //  } else {
  //     nsubmix_bnd = nsubmix;
  //  }
  dtmix = dtmicro / nsubmix;

  // rce-comment
  //    the activation source(k) = mact(k,m)*raercol(kp1,lmass)
  //       should not exceed the rate of transfer of unactivated particles
  //       from kp1 to k which = ekkp(k)*raercol(kp1,lmass)
  //    however it might if things are not "just right" in subr activate
  //    the following is a safety measure to avoid negatives in explmix

  for (int k = top_lev; k < pver; k++) {
    for (int imode = 0; imode < ntot_amode; imode++) {
      nact[imode](k) = haero::min(nact[imode](k), ekkp(k));
      mact[imode](k) = haero::min(mact[imode](k), ekkp(k));
    }
  }

  // old_cloud_nsubmix_loop
  //  Note:  each pass in submix loop stores updated aerosol values at index
  //  nnew, current values at index nsav.  At the start of each pass, nnew
  //  values are copied to nsav.  However, this is accomplished by switching the
  //  values of nsav and nnew rather than a physical copying.  At end of loop
  //  nnew stores index of most recent updated values (either 1 or 2).

  for (int isub = 0; isub < nsubmix; isub++) {
    for (int k = top_lev; k < pver; k++) {
      qncld(k) = qcld(k);
      srcn(k) = 0.0;
    }
    // after first pass, switch nsav, nnew so that nsav is the recently updated
    // aerosol
    if (isub > 0) {
      const int ntemp = nsav;
      nsav = nnew;
      nnew = ntemp;
    }

    for (int imode = 0; imode < ntot_amode; imode++) {
      const int mm = mam_idx[imode][0] - 1;

      // update droplet source

      // rce-comment- activation source in layer k involves particles from k+1
      //	       srcn(:)=srcn(:)+nact(:,m)*(raercol(:,mm,nsav))
      for (int k = top_lev; k < pver - 1; k++) {
        const int kp1 = haero::min(k + 1, pver);
        srcn(k) += nact[imode](k) * raercol[nsav][mm](kp1);
      } // kk
      // rce-comment- new formulation for k=pver
      //             srcn(  pver  )=srcn(  pver  )+nact(  pver  ,m)*(raercol(
      //             pver,mm,nsav))
      tmpa = raercol[nsav][mm](pver - 1) * nact[imode](pver - 1) +
             raercol_cw[nsav][mm](pver - 1) * nact[imode](pver - 1);
      srcn(pver - 1) += haero::max(zero, tmpa);
    }
    // qcld == qold
    // qncld == qnew
    for (int k = top_lev; k < pver; k++) {
      const int kp1 = haero::min(k + 1, pver);
      const int km1 = haero::max(k - 1, top_lev);
      explmix(qcld(km1), qcld(k), qcld(kp1), qncld(k), srcn(k), ekkp(k),
              ekkm(k), overlapp(k), overlapm(k), dtmix);
    }
    // update aerosol number
    // rce-comment
    //    the interstitial particle mixratio is different in clear/cloudy
    //    portions of a layer, and generally higher in the clear portion.  (we
    //    have/had a method for diagnosing the the clear/cloudy mixratios.)  the
    //    activation source terms involve clear air (from below) moving into
    //    cloudy air (above). in theory, the clear-portion mixratio should be
    //    used when calculating source terms
    for (int imode = 0; imode < ntot_amode; imode++) {
      const int mm = mam_idx[imode][0] - 1;
      // rce-comment -   activation source in layer k involves particles from
      // k+1
      //	              source(:)= nact(:,m)*(raercol(:,mm,nsav))
      for (int k = top_lev; k < pver - 1; k++) {
        const int kp1 = haero::min(k + 1, pver);
        // const int km1 = haero::max(k-1, top_lev);
        source(k) = nact[imode](k) * raercol[nsav][mm](kp1);
      } // end k

      tmpa = raercol[nsav][mm](pver - 1) * nact[imode](pver - 1) +
             raercol_cw[nsav][mm](pver - 1) * nact[imode](pver - 1);
      source(pver - 1) = haero::max(zero, tmpa);

      // raercol_cw[mm][nnew] == qold
      // raercol_cw[mm][nsav] == qnew
      for (int k = top_lev; k < pver; k++) {
        const int kp1 = haero::min(k + 1, pver);
        const int km1 = haero::max(k - 1, top_lev);
        explmix(raercol_cw[nnew][mm](km1), raercol_cw[nnew][mm](k),
                raercol_cw[nnew][mm](kp1), raercol_cw[nsav][mm](k), source(k),
                ekkp(k), ekkm(k), overlapp(k), overlapm(k), dtmix);
      }

      // raercol[mm][nnew] == qold
      // raercol[mm][nsav] == qnew
      // raercol_cw[mm][nsav] == qactold
      for (int k = top_lev; k < pver; k++) {
        const int kp1 = haero::min(k + 1, pver);
        const int km1 = haero::max(k - 1, top_lev);
        explmix(raercol[nnew][mm](km1), raercol[nnew][mm](k),
                raercol[nnew][mm](kp1), raercol[mm][nsav](k), source(k),
                ekkp(k), ekkm(k), overlapp(k), overlapm(k), dtmix,
                raercol[nsav][mm](km1), raercol[nsav][mm](kp1)); // optional in
      }

      // update aerosol species mass
      for (int lspec = 1; lspec < nspec_amode[imode] + 1; lspec++) {
        const int mm = mam_idx[imode][lspec] - 1;
        // rce-comment -   activation source in layer k involves particles from
        // k+1
        //	          source(:)= mact(:,m)*(raercol(:,mm,nsav))
        for (int k = top_lev; k < pver - 1; k++) {
          const int kp1 = haero::min(k + 1, pver);
          source(k) = mact[imode](k) * raercol[nsav][mm](kp1);

        } // end k

        tmpa = raercol[nsav][mm](pver - 1) * nact[imode](pver - 1) +
               raercol_cw[nsav][mm](pver - 1) * nact[imode](pver - 1);
        source(pver - 1) = haero::max(zero, tmpa);

        // raercol_cw[mm][nnew] == qold
        // raercol_cw[mm][nsav] == qnew
        for (int k = top_lev; k < pver; k++) {
          const int kp1 = haero::min(k + 1, pver);
          const int km1 = haero::max(k - 1, top_lev);
          explmix(raercol_cw[nnew][mm](km1), raercol_cw[nnew][mm](k),
                  raercol_cw[nnew][mm](kp1), raercol_cw[nsav][mm](k), source(k),
                  ekkp(k), ekkm(k), overlapp(k), overlapm(k), dtmix);
        }

        // raercol[mm][nnew] == qold
        // raercol[mm][nsav] == qnew
        // raercol_cw[mm][nsav] == qactold
        for (int k = top_lev; k < pver; k++) {
          const int kp1 = haero::min(k + 1, pver);
          const int km1 = haero::max(k - 1, top_lev);
          explmix(raercol[nnew][mm](km1), raercol[nnew][mm](k),
                  raercol[nnew][mm](kp1), raercol[nsav][mm](k), source(k),
                  ekkp(k), ekkm(k), overlapp(k), overlapm(k), dtmix,
                  raercol_cw[nsav][mm](km1),
                  raercol_cw[nsav][mm](kp1)); // optional in
        }

      } // lspec loop
    }   //  imode loop

  } // old_cloud_nsubmix_loop

  // evaporate particles again if no cloud

  for (int k = top_lev; k < pver; k++) {
    if (cldn(k) == 0) {
      // no cloud
      qcld(k) = 0.0;

      // convert activated aerosol to interstitial in decaying cloud
      for (int imode = 0; imode < ntot_amode; imode++) {
        const int mm = mam_idx[imode][0] - 1;
        raercol[nnew][mm](k) += raercol_cw[nnew][mm](k);
        raercol_cw[nnew][mm](k) = 0.0;

        for (int lspec = 1; lspec < nspec_amode[imode] + 1; lspec++) {
          const int mm = mam_idx[imode][lspec] - 1;
          raercol[nnew][mm](k) += raercol_cw[nnew][mm](k);
          raercol_cw[nnew][mm](k) = 0.0;
        }
      }
    }
  }
} // end update_from_explmix

} // namespace ndrop
} // namespace mam4
#endif
