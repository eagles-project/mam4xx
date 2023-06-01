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
                           ekkm * (overlapm * qold_k - qold_k));

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
  qnew = qold_k +
         dtmix *
             (-src + ekkp * (qold_kp1 - qold_k + qactold_kp1 * (1 - overlapp)) +
              ekkm * (qold_km1 - qold_k + qactold_km1 * (1 - overlapm)));

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

#if 0
KOKKOS_INLINE_FUNCTION
void update_from_explmix(const Real dtmicro,  // time step for microphysics [s]
                         const Real csbot_k,    // air density at bottom (interface) of layer [kg/m^3]
                         const Real csbot_km1,    // air density at bottom (interface) of layer [kg/m^3]
                         const Real cldn_k,     // cloud fraction [fraction]
                         const Real cldn_km1,     // cloud fraction [fraction]
                         const Real cldn_kp1,     // cloud fraction [fraction]
                         const Real zn_k,       // g/pdel for layer [m^2/kg]
                         const Real zs_k,       // inverse of distance between levels [m^-1]
                         const Real zs_km1,       // inverse of distance between levels [m^-1]
                         const Real ekd_k,      //diffusivity for droplets [m^2/s]
                         const Real ekd_km1,      //diffusivity for droplets [m^2/s]
                         Real nact[AeroConfig::num_modes()], // fractional aero. number  activation rate [/s]
                         Real mact[AeroConfig::num_modes()], // fractional aero. mass    activation rate [/s]
                         Real qcld, // cloud droplet number mixing ratio [#/kg]
                         Real raercol_k[ncnst_tot][2], // single column of saved aerosol mass, number mixing ratios [#/kg or kg/kg]
                         Real raercol_cw_k[ncnst_tot][2], // same as raercol but for cloud-borne phase [#/kg or kg/kg]
                         int nsav, // indices for old, new time levels in substepping
                         int nnew  // indices for old, new time levels in substepping
                         ) {
  

    // local arguments
    int kk;           // vertical level index
    int kp1;         // kk+1
    int km1;          // kk-1
    int imode;       // mode counter variable
    int mm;          // local array index for MAM number, species
    int lspec;       // species counter variable
    int nsubmix, nsubmix_bnd;  // number of substeps and bound
    int ntemp;   // temporary index for substepping
    int isub;       // substep index

    Real overlap_cld_thresh = 1e-10;  //  threshold cloud fraction to compute overlap [fraction]

    Real ekk_k;       // density*diffusivity for droplets [kg/m/s]
    Real ekk_km1;       // density*diffusivity for droplets [kg/m/s]
    Real dtmin;     // time step to determine subloop time step [s]
    Real qncld;     // updated cloud droplet number mixing ratio [#/kg]
    Real ekkp;      // zn*zs*density*diffusivity [/s]
    Real ekkm;      // zn*zs*density*diffusivity   [/s]
    Real overlapp;  // cloud overlap involving level kk+1 [fraction]
    Real overlapm;  // cloud overlap involving level kk-1 [fraction]
    Real source_k;    //  source rate for activated number or species mass [/s]
    Real tinv;      // inverse timescale of droplet diffusivity [/s]
    Real dtt;       // timescale of droplet diffusivity [s]
    Real dtmix;     // timescale for subloop [s]
    Real tmpa;      //  temporary aerosol tendency variable [/s]
    Real srcn_k;      // droplet source rate [/s]

    // load new droplets in layers above, below clouds

    dtmin = dtmicro;
    //ekk= 0.0;
  
       // rce-comment -- ekd(k) is eddy-diffusivity at k/k+1 interface
       //   want ekk(k) = ekd(k) * (density at k/k+1 interface)
       //   so use pint(i,k+1) as pint is 1:pverp
       //           ekk(k)=ekd(k)*2.*pint(i,k)/(rair*(temp(i,k)+temp(i,k+1)))
       //           ekk(k)=ekd(k)*2.*pint(i,k+1)/(rair*(temp(i,k)+temp(i,k+1)))
    ekk_k = ekd_k * csbot_k;
    ekk_km1 = ekd_km1 * csbot_km1;

    kp1 = min(kk+1, pver);
    km1 = max(kk-1, top_lev);

    // maximum overlap assumption
    if (cldn_kp1 > overlap_cld_thresh) {
      overlapp = haero::min(cldn_k/cldn_kp1, 1);
    } else {
      overlapp = 1;
    }
    
    if (cldn_km1 > overlap_cld_thresh) {
      overlapm = haero::min(cldn_k/cldn_km1, 1);
    } else {
      overlapm = 1;
    }

    ekkp = zn_k * ekk * zs_k;
    ekkm = zn_k * ekk_km1 * zs_km1;
    tinv = ekkp + ekkm;

       // rce-comment -- tinv is the sum of all first-order-loss-rates
       //    for the layer.  for most layers, the activation loss rate
       //    (for interstitial particles) is accounted for by the loss by
       //    turb-transfer to the layer above.
       //    k=pver is special, and the loss rate for activation within
       //    the layer must be added to tinv.  if not, the time step
       //    can be too big, and explmix can produce negative values.
       //    the negative values are reset to zero, resulting in an
       //    artificial source.

    if (tinv > 1.e-6_r8) {
      dtt   = 1.0 / tinv;
      dtmin = haero::min(dtmin, dtt);
    }
    

    dtmix = 0.9 * dtmin;
    nsubmix = dtmicro / dtmix + 1;
    if (nsubmix > 100) {
       nsubmix_bnd = 100;
    } else {
       nsubmix_bnd = nsubmix;
    }
    dtmix = dtmicro / nsubmix;

    // rce-comment
    //    the activation source(k) = mact(k,m)*raercol(kp1,lmass)
    //       should not exceed the rate of transfer of unactivated particles
    //       from kp1 to k which = ekkp(k)*raercol(kp1,lmass)
    //    however it might if things are not "just right" in subr activate
    //    the following is a safety measure to avoid negatives in explmix

    for(int imode = 0; imode < ntot_amode; imode++) {
      nact[imode] = haero::min( nact[imode], ekkp);
      mact[imode] = haero::min( mact[imode], ekkp);
    }


    // old_cloud_nsubmix_loop

    //  Note:  each pass in submix loop stores updated aerosol values at index nnew,
    //  current values at index nsav.  At the start of each pass, nnew values are
    //  copied to nsav.  However, this is accomplished by switching the values
    //  of nsav and nnew rather than a physical copying.  At end of loop nnew stores
    //  index of most recent updated values (either 1 or 2).

    for(int isub = 0; isub < nsubmix; isub++) {
       qncld = qcld;
       // after first pass, switch nsav, nnew so that nsav is the recently updated aerosol
       if(isub > 0) {
          ntemp = nsav;
          nsav = nnew;
          nnew = ntemp;
       }
       srcn = 0.0;

       for(int imode = 0; imode < ntot_amode; imode++) {
          mm = mam_idx(imode,0); //does this 0 need to change?

          // update droplet source

          // rce-comment- activation source in layer k involves particles from k+1
          //	       srcn(:)=srcn(:)+nact(:,m)*(raercol(:,mm,nsav))
          srcn(top_lev:pver-1) = srcn(top_lev:pver-1) + nact(top_lev:pver-1,imode)*(raercol(top_lev+1:pver,mm,nsav))

          // rce-comment- new formulation for k=pver
          //              srcn(  pver  )=srcn(  pver  )+nact(  pver  ,m)*(raercol(  pver,mm,nsav))
          tmpa = raercol(pver,mm,nsav)*nact(pver,imode) &
               + raercol_cw(pver,mm,nsav)*nact(pver,imode)
          srcn(pver) = srcn(pver) + max(0.0_r8,tmpa)
       } 
       explmix(  qcld, &   // out
            srcn, ekkp, ekkm, overlapp,  &   // in
            overlapm, qncld,  &  // in
            dtmix);     // in
       // update aerosol number

       // rce-comment
       //    the interstitial particle mixratio is different in clear/cloudy portions
       //    of a layer, and generally higher in the clear portion.  (we have/had
       //    a method for diagnosing the the clear/cloudy mixratios.)  the activation
       //    source terms involve clear air (from below) moving into cloudy air (above).
       //    in theory, the clear-portion mixratio should be used when calculating
       //    source terms
       for(int imode = 0; imode < ntot_amode; imode++) {
          mm = mam_idx(imode,0);
          // rce-comment -   activation source in layer k involves particles from k+1
          //	              source(:)= nact(:,m)*(raercol(:,mm,nsav))
          source(top_lev:pver-1) = nact(top_lev:pver-1,imode)*(raercol(top_lev+1:pver,mm,nsav))
          // rce-comment - new formulation for k=pver
          //               source(  pver  )= nact(  pver,  m)*(raercol(  pver,mm,nsav))
          tmpa = raercol(pver,mm,nsav)*nact(pver,imode) &
               + raercol_cw(pver,mm,nsav)*nact(pver,imode)
          source(pver) = max(0.0_r8, tmpa)

          call explmix( raercol_cw(:,mm,nnew), &  // out
               source, ekkp, ekkm, overlapp, &   // in
               overlapm, raercol_cw(:,mm,nsav),   &  // in
               dtmix, .false. )  // in
          call explmix( raercol(:,mm,nnew), &   // out
               source, ekkp, ekkm, overlapp,  &    // in
               overlapm, raercol(:,mm,nsav),  &  // in
               dtmix, .true., &  // in
               raercol_cw(:,mm,nsav))  // optional in

          // update aerosol species mass

          for(int lspec = 0; lspec < nspec_amode[imode]; lspec++) {
             mm = mam_idx(imode,lspec);
             // rce-comment -   activation source in layer k involves particles from k+1
             //	          source(:)= mact(:,m)*(raercol(:,mm,nsav))
             source(top_lev:pver-1) = mact(top_lev:pver-1,imode)*(raercol(top_lev+1:pver,mm,nsav))
             // rce-comment- new formulation for k=pver
             //                 source(  pver  )= mact(  pver  ,m)*(raercol(  pver,mm,nsav))
             tmpa = raercol(pver,mm,nsav)*mact(pver,imode) &
                  + raercol_cw(pver,mm,nsav)*mact(pver,imode)
             source(pver) = max(0.0_r8, tmpa)

             explmix( raercol_cw(:,mm,nnew), &  // out
                  source, ekkp, ekkm, overlapp, &  // in
                  overlapm, raercol_cw(:,mm,nsav),    &  // in
                  dtmix); // in
             explmix( raercol(:,mm,nnew), &  // out
                  source, ekkp, ekkm, overlapp,  &  // in
                  overlapm, raercol(:,mm,nsav),  &   // in
                  dtmix, .true., & // in
                  raercol_cw(:,mm,nsav))  // optional in

          }  // lspec loop
       }  //  imode loop

    } // old_cloud_nsubmix_loop

    // evaporate particles again if no cloud

    if (cldn_k == 0) {
      // no cloud
      qcld = 0.0;

      // convert activated aerosol to interstitial in decaying cloud
      for(int imode = 0; imode < ntot_amode; imode++) {
        mm = mam_idx(imode,0);
        raercol_k[mm][nnew] += raercol_cw_k[mm][nnew];
        raercol_cw_k[mm][nnew] = 0.0;

        for(int lspec = 0; lspec < nspec_amode[imode]; lspec++) {
          mm = mam_idx(imode,lspec);
          raercol_k[mm][nnew] += raercol_cw_k[mm][nnew];
          raercol_cw_k[mm][nnew] = 0.0;
        } 
      }
    }
    

} // end update_from_explmix
#endif

} // namespace ndrop
} // namespace mam4
#endif
