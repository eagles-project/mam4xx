#ifndef MAM4XX_MO_PHOTO_HPP
#define MAM4XX_MO_PHOTO_HPP

#include <haero/math.hpp>
#include <mam4xx/mam4_types.hpp>
#include <mam4xx/utils.hpp>

namespace mam4 {

namespace mo_photo {

// number of vertical levels
constexpr int pver = 72;
constexpr int pverm = pver - 1;

using View5D = Kokkos::View<Real*****>;
using View4D = Kokkos::View<Real****>;
using View2D = DeviceType::view_2d<Real>;


KOKKOS_INLINE_FUNCTION
void cloud_mod(const Real zen_angle, const Real *clouds, const Real *lwc,
               const Real *delp,
               const Real srf_alb, //  in
               Real *eff_alb, Real *cld_mult) {
  /*-----------------------------------------------------------------------
        ... cloud alteration factors for photorates and albedo
  -----------------------------------------------------------------------*/

  // @param[in]   zen_angle         zenith angle [deg]
  // @param[in]   srf_alb           surface albedo [fraction]
  // @param[in]   clouds(pver)      cloud fraction [fraction]
  // @param[in]   lwc(pver)         liquid water content [kg/kg]
  // @param[in]   delp(pver)        del press about midpoint [Pa]
  // @param[out]  eff_alb(pver)    effective albedo [fraction]
  // @param[out]  cld_mult(pver) photolysis mult factor

  /*---------------------------------------------------------
        ... modify lwc for cloud fraction and form
            liquid water path and tau for each layer
  ---------------------------------------------------------*/
  const Real zero = 0.0;
  const Real thousand = 1000.0;
  const Real one = 1;
  const Real half = 0.5;

  // cloud optical depth in each layer
  Real del_tau[pver] = {};
  // cloud optical depth below this layer
  Real below_tau[pver] = {};
  // cloud cover below this layer
  Real below_cld[pver] = {};

  // BAD CONSTANT
  const Real rgrav = one / 9.80616; //  1/g [s^2/m]
  const Real f_lwp2tau =
      .155; // factor converting LWP to tau [unknown source and unit]
  const Real tau_min = 5.0; // tau threshold below which assign cloud as zero

  for (int kk = 0; kk < pver; kk++) {
    if (clouds[kk] != zero) {
      // liquid water path in each layer [g/m2]
      const Real del_lwp = rgrav * lwc[kk] * delp[kk] * thousand /
                           clouds[kk]; // the unit is (likely) g/m^2
      del_tau[kk] = del_lwp * f_lwp2tau * haero::pow(clouds[kk], 1.5);
    } else {
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

  below_tau[pver - 1] = zero;
  below_cld[pver - 1] = zero;

  for (int kk = pverm - 1; kk > -1; kk--) {
    below_tau[kk] = del_tau[kk + 1] + below_tau[kk + 1];
    below_cld[kk] = clouds[kk + 1] * del_tau[kk + 1] + below_cld[kk + 1];
  } // end kk

  for (int kk = pverm - 1; kk > -1; kk--) {
    if (below_tau[kk] != zero) {
      below_cld[kk] /= below_tau[kk];
    } else {
      below_cld[kk] = below_cld[kk + 1];
    } // end if

  } // end kk

  /*---------------------------------------------------------
      ... modify above_tau and below_tau via jfm
  ---------------------------------------------------------*/
  for (int kk = 1; kk < pver; kk++) {
    if (above_cld[kk] != zero) {
      above_tau[kk] /= above_cld[kk];
    } // end if

    if (above_tau[kk] < tau_min) {
      above_cld[kk] = zero;
    } // end if

  } // end kk

  for (int kk = 0; kk < pverm; kk++) {
    if (below_cld[kk] != zero) {
      below_tau[kk] /= below_cld[kk];
    } // end if
    if (below_tau[kk] < tau_min) {
      below_cld[kk] = zero;
    } // end if

  } // end kk

  /*---------------------------------------------------------
      ... form transmission factors
  ---------------------------------------------------------*/

  // BAD CONSTANT
  const Real C1 = 11.905;
  const Real C2 = 9.524;

  // cos (solar zenith angle)
  const Real coschi = haero::max(haero::cos(zen_angle), half);

  for (int kk = 0; kk < pver; kk++) {

    /*---------------------------------------------------------
      ... form effective albedo
      ---------------------------------------------------------*/
    // transmission factor below this layer
    const Real below_tra = C1 / (C2 + below_tau[kk]);
    eff_alb[kk] = srf_alb + below_cld[kk] * (one - below_tra) * (one - srf_alb);

    // factor to calculate cld_mult
    Real fac1 = zero;
    Real del_lwp = zero;
    if (clouds[kk] != zero) {
      // liquid water path in each layer [g/m2]
      del_lwp = rgrav * lwc[kk] * delp[kk] * thousand /
                clouds[kk]; // the unit is (likely) g/m^2
    }
    if (del_lwp * f_lwp2tau >= tau_min) {
      // BAD CONSTANT
      fac1 = 1.4 * coschi - one;
    } // end if
    // transmission factor above this layer
    const Real above_tra = C1 / (C2 + above_tau[kk]);
    // factor to calculate cld_mult
    Real fac2 = haero::min(zero, 1.6 * coschi * above_tra - one);
    // BAD CONSTANT
    cld_mult[kk] =
        haero::max(.05, one + fac1 * clouds[kk] + fac2 * above_cld[kk]);
  } // end kk

} // end cloud_mod

KOKKOS_INLINE_FUNCTION
void find_index(const Real *var_in, const int var_len,
                const Real var_min, //  in
                int &idx_out)       // out
{

  /*------------------------------------------------------------------------------
   find the index of the first element in var_in(1:var_len) where the value is >
  var_min
  ------------------------------------------------------------------------------*/
  // @param[in]  var_len   length of the input variable
  // @param[in]  var_in(var_len)    input variable
  // @param[in]  var_min   variable threshold
  // @param[out]  idx_out   index

  for (int ii = 0; ii < var_len; ii++) {
    if (var_in[ii] > var_min) {
      // Fortran to C++ indexing
      idx_out = haero::max(haero::min(ii, var_len - 1) - 1, 0);
      break;
    } // end if
  }   // end for ii

} // find_index
// FIXME: get values of the following parameters:
// constexpr int nw = 67;      // wavelengths >200nm
// constexpr int nump = 135;     // number of altitudes in rsf
// constexpr int numsza = 100;   // number of zen angles in rsf
// constexpr int numcolo3 = 100; // number of o3 columns in rsf
// constexpr int numalb = 100;   // number of albedos in rsf
// constexpr int numj = 1;     // number of photorates in xsqy, rsf
// constexpr int nt = 1;       // number of temperatures in xsection table
// constexpr int np_xs = 1;    // number of pressure levels in xsection table

KOKKOS_INLINE_FUNCTION
void calc_sum_wght(const Real dels[3], const Real wrk0, // in
                   const int iz, const int is, const int iv,
                   const int ial,                                          // in
                   const View5D& rsf_tab, // in
                   const int nw, 
                   Real *psum) // out
{

  // @param[in]   dels(3)
  // @param[in]   wrk0
  // @param[in]  iz, is, iv, ial
  // @param[in/out] psum(:)
  const int isp1 = is + 1;
  const int ivp1 = iv + 1;
  const int ialp1 = ial + 1;
  const Real one = 1;
  // #if 0
  Real wrk1 = (one - dels[1]) * (one - dels[2]);
  const Real wght_0_0_0 = wrk0 * wrk1;
  const Real wght_1_0_0 = dels[0] * wrk1;
  wrk1 = (one - dels[1]) * dels[2];
  const Real wght_0_0_1 = wrk0 * wrk1;
  const Real wght_1_0_1 = dels[0] * wrk1;
  wrk1 = dels[1] * (one - dels[2]);
  const Real wght_0_1_0 = wrk0 * wrk1;
  const Real wght_1_1_0 = dels[0] * wrk1;
  wrk1 = dels[1] * dels[2];
  const Real wght_0_1_1 = wrk0 * wrk1;
  const Real wght_1_1_1 = dels[0] * wrk1;

  // nw and rsf_tab are module variables
  for (int wn = 0; wn < nw; wn++) {
    psum[wn] = wght_0_0_0 * rsf_tab(wn,iz,is,iv,ial) +
               wght_0_0_1 * rsf_tab(wn,iz,is,iv,ialp1) +
               wght_0_1_0 * rsf_tab(wn,iz,is,ivp1,ial) +
               wght_0_1_1 * rsf_tab(wn,iz,is,ivp1,ialp1) +
               wght_1_0_0 * rsf_tab(wn,iz,isp1,iv,ial) +
               wght_1_0_1 * rsf_tab(wn,iz,isp1,iv,ialp1) +
               wght_1_1_0 * rsf_tab(wn,iz,isp1,ivp1,ial) +
               wght_1_1_1 * rsf_tab(wn,iz,isp1,ivp1,ialp1);
  } // end wn
  // #endif
} // calc_sum_wght

constexpr int nlev = 72;
KOKKOS_INLINE_FUNCTION
void interpolate_rsf(const Real *alb_in, const Real sza_in, const Real *p_in,
                     const Real *colo3_in,
                     const int kbot, //  in
                     const Real *sza, const Real *del_sza, const Real *alb,
                     const Real *press, const Real *del_p, const Real *colo3,
                     const Real *o3rat, const Real *del_alb,
                     const Real *del_o3rat, const Real *etfphot,
                     const View5D& rsf_tab, // in
                     const int nw, 
                     const int nump,
                     const int numsza, 
                     const int numcolo3, 
                     const int numalb, 
                     const View2D& rsf, // out
                     // work array
                     Real *psum_l, Real *psum_u) {
  /*----------------------------------------------------------------------
          ... interpolate table rsf to model variables
  ----------------------------------------------------------------------*/

  // @param[in]  alb_in(:)        albedo [unitless]
  // @param[in]  sza_in           solar zenith angle [degrees]
  // @param[in]  kbot            ! heating levels [level]
  // @param[in]  p_in(:)         ! midpoint pressure [hPa]
  // @param[in]  colo3_in(:)     ! o3 column density [molecules/cm^3]
  // @param[in] rsf(:,:)        ! Radiative Source Function [quanta cm-2 sec-1]

  /*----------------------------------------------------------------------
          ... find the zenith angle index ( same for all levels )
  ----------------------------------------------------------------------*/
  const Real one = 1;
  const Real zero = 0;

  int is = 0;
  find_index(sza, numsza, sza_in, // in
             is);                 // ! out

  Real dels[3] = {};
  dels[0] = utils::min_max_bound(zero, one, (sza_in - sza[is]) * del_sza[is]);
  const Real wrk0 = one - dels[0];
  int izl = 2; //   may change in the level_loop
  for (int kk = kbot - 1; kk > -1; kk--) {
    /*----------------------------------------------------------------------
       ... find albedo indicies
     ----------------------------------------------------------------------*/
    int albind = 0;
    find_index(alb, numalb, alb_in[kk], //  & ! in
               albind);                 // ! out
    /*----------------------------------------------------------------------
         ... find pressure level indicies
    ----------------------------------------------------------------------*/
    int pind = 0;
    Real wght1 = 0;
    if (p_in[kk] > press[0]) {
      pind = 1;
      wght1 = one;
      // Fortran to C++ indexing
    } else if (p_in[kk] <= press[nump-1]) {
      // Fortran to C++ indexing
      pind = nump-1;
      wght1 = zero;
    } else {
      int iz = 0;
      // Fortran to C++ indexing
      for (iz = izl - 1; iz < nump; iz++) {
        if (press[iz] < p_in[kk]) {
          izl = iz;
          break;
        } // end if
      }   // end for iz
      // Fortran to C++ indexing
      pind = haero::max(haero::min(iz, nump-1), 1);
      wght1 = utils::min_max_bound(zero, one,
                                   (p_in[kk] - press[pind]) * del_p[pind - 1]);
    } // end if

    /*----------------------------------------------------------------------
         ... find "o3 ratios"
    ----------------------------------------------------------------------*/

    const Real v3ratu = colo3_in[kk] / colo3[pind - 1];
    int ratindu = 0;
    find_index(o3rat, numcolo3, v3ratu, //  in
               ratindu);                // out

    Real v3ratl = zero;
    int ratindl = 0;
    if (colo3[pind] != zero) {
      v3ratl = colo3_in[kk] / colo3[pind];
      find_index(o3rat, numcolo3, v3ratl, // in
                 ratindl);                // ! out
    } else {
      ratindl = ratindu;
      v3ratl = o3rat[ratindu];
    } // end if colo3[pind] != zero

    /*----------------------------------------------------------------------
            ... compute the weigths
    ----------------------------------------------------------------------*/

    int ial = albind;

    dels[2] =
        utils::min_max_bound(zero, one, (alb_in[kk] - alb[ial]) * del_alb[ial]);

    int iv = ratindl;
    dels[1] =
        utils::min_max_bound(zero, one, (v3ratl - o3rat[iv]) * del_o3rat[iv]);
    // FIXME
    calc_sum_wght(dels, wrk0,        // in
                  pind, is, iv, ial, // in
                  rsf_tab,
                  nw, 
                  psum_l); // out

    iv = ratindu;
    dels[1] =
        utils::min_max_bound(zero, one, (v3ratu - o3rat[iv]) * del_o3rat[iv]);
    calc_sum_wght(dels, wrk0,            // in
                  pind - 1, is, iv, ial, // in
                  rsf_tab,
                  nw,
                  psum_u); //  inout

    for (int wn = 0; wn < nw; wn++) {
      rsf(wn,kk) = psum_l[wn] + wght1 * (psum_u[wn] - psum_l[wn]);
    }

    /*------------------------------------------------------------------------------
        etfphot comes in as photons/cm^2/sec/nm  (rsf includes the wlintv factor
     -- nm)
       ... --> convert to photons/cm^2/s
     ------------------------------------------------------------------------------*/
    for (int wn = 0; wn < nw; wn++) {
      rsf(wn,kk) *= etfphot[wn];
    } // end for wn

  } // end Level_loop

  // #endif

} // interpolate_rsf

//======================================================================================
KOKKOS_INLINE_FUNCTION
void jlong(
    //  const int nlev,
    const Real sza_in, const Real *alb_in, const Real *p_in, const Real *t_in,
    const Real *colo3_in, const View4D& xsqy, const Real *sza,
    const Real *del_sza, const Real *alb, const Real *press, const Real *del_p,
    const Real *colo3, const Real *o3rat, const Real *del_alb,
    const Real *del_o3rat, const Real *etfphot,
    const View5D& rsf_tab,
    const Real *prs, const Real *dprs,
    Real &j_long, // output

    // work arrays
    const View2D& rsf, const View2D& xswk, Real* psum_l,
    Real* psum_u) // out
{
#if 0
  /*==============================================================================
     Purpose:
       To calculate the total J for selective species longward of 200nm.
  ==============================================================================
     Approach:
       1) Reads the Cross Section*QY NetCDF file
       2) Given a temperature profile, derives the appropriate XS*QY

       3) Reads the Radiative Source function (RSF) NetCDF file
          Units = quanta cm-2 sec-1

       4) Indices are supplied to select a RSF that is consistent with
          the reference atmosphere in TUV (for direct comparision of J's).
          This approach will be replaced in the global model. Here colo3, zenith
          angle, and altitude will be inputed and the correct entry in the table
          will be derived.
  ==============================================================================*/

  // @param[in] nlev               ! number vertical levels
  // @param[in] sza_in             ! solar zenith angle [degrees]
  // @param[in] alb_in(nlev)       ! albedo
  // @param[in]  p_in(nlev)         ! midpoint pressure [hPa]
  // @param[in]  t_in(nlev)         ! Temperature profile [K]
  // @param[in]  colo3_in(nlev)     ! o3 column density [molecules/cm^3]
  // @param[out]  j_long(:,:)        ! photo rates [1/s]

  /*----------------------------------------------------------------------
    ... interpolate table rsf to model variables
----------------------------------------------------------------------*/
  const Real zero = 0;
  interpolate_rsf(alb_in, sza_in, p_in, colo3_in, nlev, sza, del_sza, alb,
                  press, del_p, colo3, o3rat, del_alb, del_o3rat, etfphot,
                  rsf_tab,              //  in
                  rsf, psum_l, psum_u); // out
  /*------------------------------------------------------------------------------
  ... calculate total Jlong for wavelengths >200nm
  ------------------------------------------------------------------------------
  ------------------------------------------------------------------------------
   LLNL LUT approach to finding temperature index...
   Calculate the temperature index into the cross section
   data which lists coss sections for temperatures from
   150 to 350 degrees K.  Make sure the index is a value
   between 1 and 201.
  ------------------------------------------------------------------------------*/

  for (int kk = 0; kk < nlev; kk++) {
    /*----------------------------------------------------------------------
      ... get index into xsqy
     ----------------------------------------------------------------------*/
    // BAD CONSTANT
    // Fortran indexing to C++ indexing
    const int t_index = haero::min(201, haero::max(t_in[kk] - 148.5, 1)) - 1;

    /*----------------------------------------------------------------------
               ... find pressure level
     ----------------------------------------------------------------------*/
    const Real ptarget = p_in[kk];
    if (ptarget >= prs[0]) {
      for (int wn = 0; wn < nw; wn++) {
        for (int i = 0; i < numj; i++) {
          xswk(i,wn) = xsqy(i,wn,t_index,0);
        } // end for i
      }   // end for wn
      // Fortran to C++ indexing conversion
    } else if (ptarget <= prs[np_xs - 1]) {
      for (int wn = 0; wn < nw; wn++) {
        for (int i = 0; i < numj; i++) {
          // Fortran to C++ indexing conversion
          xswk(i,wn) = xsqy(i,wn,t_index,np_xs - 1);
        } // end for i
      }   // end for wn

    } else {
      Real delp = zero;
      int pndx = 0;
      // Question: delp is not initialized in fortran code. What if the
      // following code does not satify this if condition: ptarget >= prs[km]
      for (int km = 1; km < np_xs; km++) {
        if (ptarget >= prs[km]) {
          // Fortran to C++ indexing convertion
          pndx = km - 1 - 1;
          delp = (prs[pndx] - ptarget) * dprs[pndx];
          break;
        } // end if

      } // end for km
      for (int wn = 0; wn < nw; wn++) {
        for (int i = 0; i < numj; i++) {
          xswk(i,wn) = xsqy(i,wn,t_index,pndx) +
                        delp * (xsqy(i,wn,t_index,pndx + 1) -
                                xsqy(i,wn,t_index,pndx));

        } // end for i
      }   // end for wn
    }     // end if

    // FIXME: Does haero have matrix multiplication ?
    // j_long(:,kk) = matmul( xswk(:,:),rsf(:,kk) )

  } // end kk
#endif
} // jlong

} // namespace mo_photo
} // end namespace mam4

#endif
