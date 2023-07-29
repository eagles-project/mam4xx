#ifndef MAM4XX_MO_PHOTO_HPP
#define MAM4XX_MO_PHOTO_HPP

#include <haero/math.hpp>
#include <mam4xx/mam4_types.hpp>
namespace mam4 {

namespace mo_photo {

// number of vertical levels
constexpr int pver = 72;
constexpr int pverm = pver - 1;

KOKKOS_INLINE_FUNCTION
void cloud_mod(const Real zen_angle,
               const Real *clouds,
               const Real *lwc,
               const Real *delp,
               const Real srf_alb, //  in
               Real *eff_alb, Real *cld_mult) {
  /*-----------------------------------------------------------------------
  ! 	... cloud alteration factors for photorates and albedo
  !-----------------------------------------------------------------------*/

  // @param[in]   zen_angle         ! zenith angle [deg]
  // @param[in]   srf_alb           ! surface albedo [fraction]
  // @param[in]   clouds(pver)       ! cloud fraction [fraction]
  // @param[in]   lwc(pver)          ! liquid water content [kg/kg]
  // @param[in]   delp(pver)         ! del press about midpoint [Pa]
  // @param[out]   eff_alb(pver)      ! effective albedo [fraction]
  // @param[out]     cld_mult(pver)     ! photolysis mult factor

  /*---------------------------------------------------------
  !	... modify lwc for cloud fraction and form
  !	    liquid water path and tau for each layer
  !---------------------------------------------------------*/
  const Real zero = 0.0;
  const Real thousand = 1000.0;
  const Real one = 1;
  const Real half = 0.5;

  // liquid water path in each layer [g/m2]
  Real del_lwp[pver] = {};
  // cloud optical depth in each layer
  Real del_tau[pver] = {};

  // BAD CONSTANT
  const Real rgrav = one / 9.80616; //  1/g [s^2/m]
  const Real f_lwp2tau =
      .155; // factor converting LWP to tau [unknown source and unit]
  const Real tau_min = 5.0; // tau threshold below which assign cloud as zero

  for (int kk = 0; kk < pver; kk++) {
    if (clouds[kk] != zero) {

      del_lwp[kk] = rgrav * lwc[kk] * delp[kk] * thousand /
                    clouds[kk]; // the unit is (likely) g/m^2
      del_tau[kk] = del_lwp[kk] * f_lwp2tau * haero::pow(clouds[kk], 1.5);
    } else {
      del_lwp[kk] = zero;
      del_tau[kk] = zero;
    } // end if
  }   // end kk
  /*---------------------------------------------------------
              ... form integrated tau and cloud cover from top down
  --------------------------------------------------------- */
  // cloud optical depth above this layer
  Real above_tau[pver] = {zero};
  // cloud cover above this layer
  Real above_cld[pver] = {zero};

  for (int kk = 0; kk < pverm; kk++) {
    above_tau[kk + 1] = del_tau[kk] + above_tau[kk];
    above_cld[kk + 1] = clouds[kk] * del_tau[kk] + above_cld[kk];
  } // end kk

  for (int kk = 1; kk < pver; kk++) {
    if (above_tau[kk] != zero) {
      above_cld[kk] /= above_tau[kk];
    } else {
      above_cld[kk] = above_cld[kk - 1];
    }
  } // end kk

  /*---------------------------------------------------------
              ... form integrated tau and cloud cover from bottom up
  ---------------------------------------------------------*/
  // cloud optical depth below this layer
  Real below_tau[pver] = {};
  below_tau[pver - 1] = zero;
  // cloud cover below this layer
  Real below_cld[pver] = {};
  below_cld[pver - 1] = zero;

  for (int kk = pverm - 1; kk > 0; kk--) {
    below_tau[kk] = del_tau[kk + 1] + below_tau[kk + 1];
    below_cld[kk] = clouds[kk + 1] * del_tau[kk + 1] + below_cld[kk + 1];
  } // end kk

  for (int kk = pverm - 1; kk > 0; kk--) {
    if (below_tau[kk] != zero) {
      below_cld[kk] /= below_tau[kk];
    } else {
      below_cld[kk] = below_cld[kk + 1];
    } // end if
  }   // end kk

  /*---------------------------------------------------------
      ... modify above_tau and below_tau via jfm
  ---------------------------------------------------------*/

  for (int kk = 1; kk < pver; kk++) {
    if (above_cld[kk] != zero) {
      above_tau[kk] /= above_cld[kk];
    } // end if
  }   // end kk

  for (int kk = 0; kk < pverm; kk++) {
    if (below_cld[kk] != zero) {
      above_tau[kk] /= below_cld[kk];
    } // end if
  }   // end kk

  for (int kk = 1; kk < pver; kk++) {
    if (above_tau[kk] < tau_min) {
      above_cld[kk] = zero;
    } // end if
  }   // end kk

  for (int kk = 0; kk < pverm; kk++) {
    if (below_tau[kk] < tau_min) {
      below_cld[kk] = zero;
    } // end if
  }   // end kk

  /*---------------------------------------------------------
      ... form transmission factors
  ---------------------------------------------------------*/

  // BAD CONSTANT
  const Real C1 = 11.905;
  const Real C2 = 9.524;
  /*---------------------------------------------------------
      ... form effective albedo
  ---------------------------------------------------------*/
  for (int kk = 0; kk < pver; kk++) {
    if (below_cld[kk] != zero) {
      // // transmission factor below this layer
      const Real below_tra = C1 / (C2 + below_tau[kk]);
      eff_alb[kk] =
          srf_alb + below_cld[kk] * (one - below_tra) * (one - srf_alb);
    } else {
      eff_alb[kk] = srf_alb;
    } // end if

  } // end kk

  // cos (solar zenith angle)
  const Real coschi = haero::max(haero::cos(zen_angle), half);

  for (int kk = 0; kk < pver; kk++) {
    // factor to calculate cld_mult
    Real fac1 = zero;
    if (del_lwp[kk] * f_lwp2tau >= tau_min) {
      // BAD CONSTANT
      fac1 = 1.4 * coschi - one;
    } // end if

    // transmission factor above this layer
    const Real above_tra = C1 / (C2 + above_tau[kk]);
    // factor to calculate cld_mult
    Real fac2 = haero::min(zero, (1.6 * coschi * above_tra) - one);
    // BAD CONSTANT
    cld_mult[kk] =
        haero::max(.05, one + fac1 * clouds[kk] + fac2 * above_cld[kk]);

  } // end kk

} // end cloud_mod

} // namespace mo_photo
} // end namespace mam4

#endif
