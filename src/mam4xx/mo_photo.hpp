#ifndef MAM4XX_MO_PHOTO_HPP
#define MAM4XX_MO_PHOTO_HPP

#include <haero/math.hpp>
#include <mam4xx/aero_config.hpp>
#include <mam4xx/gas_chem_mechanism.hpp>
#include <mam4xx/mam4_types.hpp>
#include <mam4xx/utils.hpp>

namespace mam4 {

namespace mo_photo {

// number of vertical levels
constexpr int pver = mam4::nlev;
constexpr int pverm = pver - 1;

// number of photolysis reactions
constexpr int phtcnt = 1;

using View5D = DeviceType::view_ND<Real, 5>;
using View4D = DeviceType::view_ND<Real, 4>;
using View2D = DeviceType::view_2d<Real>;
using View1D = DeviceType::view_1d<Real>;
using ViewInt1D = DeviceType::view_1d<int>;

// photolysis table data (common to all columns)
struct PhotoTableData {
  View4D xsqy;
  View1D sza;
  View1D del_sza;
  View1D alb;
  View1D press;
  View1D del_p;
  View1D colo3;
  View1D o3rat;
  View1D del_alb;
  View1D del_o3rat;
  View1D etfphot;
  View5D rsf_tab;
  View1D prs;
  View1D dprs;
  View1D pht_alias_mult_1;
  ViewInt1D lng_indexer;

  int nw;       // number of wavelengths >200nm
  int nt;       // number of temperatures in xsection table
  int np_xs;    // number of pressure levels in xsection table
  int numj;     // number of photorates in xsqy, rsf
  int nump;     // number of altitudes in rsf
  int numsza;   // number of zenith angles in rsf
  int numcolo3; // number of o3 columns in rsf
  int numalb;   // number of albedos in rsf
};

// this host-only function creates a new PhotoTableData instance given its
// various dimensions, and initializes the necessary device Views
inline PhotoTableData create_photo_table_data(int nw, int nt, int np_xs,
                                              int numj, int nump, int numsza,
                                              int numcolo3, int numalb) {
  PhotoTableData table_data{};

  // set dimensions
  table_data.nw = nw;
  table_data.nump = nump;
  table_data.numsza = numsza;
  table_data.numcolo3 = numcolo3;
  table_data.numalb = numalb;
  table_data.numj = numj;
  table_data.nt = nt;
  table_data.np_xs = np_xs;

  // create views
  table_data.rsf_tab =
      View5D("photo_table_data.rsf_tab", table_data.nw, table_data.numalb,
             table_data.numcolo3, table_data.numsza, table_data.nump);
  table_data.xsqy = View4D("photo_table_data.xsqy", table_data.numj,
                           table_data.nw, table_data.nt, table_data.np_xs);
  table_data.sza = View1D("photo_table_data.sza", table_data.numsza);
  table_data.del_sza =
      View1D("photo_table_data.del_sza", table_data.numsza - 1);
  table_data.alb = View1D("photo_table_data.alb", table_data.numalb);
  table_data.press = View1D("photo_table_data.press", table_data.nump);
  table_data.del_p = View1D("photo_table_data.del_p", table_data.nump - 1);
  table_data.colo3 = View1D("photo_table_data.colo3", table_data.nump);
  table_data.o3rat = View1D("photo_table_data.o3rat", table_data.numcolo3);
  table_data.del_alb =
      View1D("photo_table_data.del_alb", table_data.numalb - 1);
  table_data.del_o3rat =
      View1D("photo_table_data.del_o3rat", table_data.numcolo3 - 1);
  table_data.etfphot = View1D("photo_table_data.etfphot", table_data.nw);
  table_data.prs = View1D("photo_table_data.prs", table_data.np_xs);
  table_data.dprs = View1D("photo_table_data.dprs", table_data.np_xs - 1);
  table_data.pht_alias_mult_1 = View1D("photo_table_data.pht_alias_mult_1", 2);
  table_data.lng_indexer = ViewInt1D("photo_table_data.lng_indexer", phtcnt);

  return table_data;
}

// column-specific photolysis work arrays
struct PhotoTableWorkArrays {
  View2D lng_prates;
  View2D rsf;
  View2D xswk;
  View1D psum_l;
  View1D psum_u;
};

KOKKOS_INLINE_FUNCTION
void cloud_mod(const Real zen_angle, const Real *clouds, const Real *lwc,
               const Real *delp,
               const Real srf_alb, //  in
               Real *eff_alb, Real *cld_mult) {
  /*-----------------------------------------------------------------------
        ... cloud alteration factors for photorates and albedo
  -----------------------------------------------------------------------*/

  // @param[in]   zen_angle         zenith angle [deg]
  // @param[in]   clouds(pver)      cloud fraction [fraction]
  // @param[in]   lwc(pver)         liquid water content [kg/kg]
  // @param[in]   srf_alb           surface albedo [fraction]
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
    // BAD CONSTANT
    Real fac2 = haero::min(zero, 1.6 * coschi * above_tra - one);
    // BAD CONSTANT
    cld_mult[kk] =
        haero::max(.05, one + fac1 * clouds[kk] + fac2 * above_cld[kk]);
  } // end kk

} // end cloud_mod

// NOTE: set_ub_col and setcol compute only the density of o3 in the atmosphere
// NOTE: (o2 is not computed--if we want to add it back to the calculation,
// NOTE: we should properly reorganize the code before we do it)
KOKKOS_INLINE_FUNCTION
void set_ub_col(Real &o3_col_delta,
                const Real vmr[mam4::gas_chemistry::gas_pcnst],
                const Real invariants[mam4::gas_chemistry::nfs],
                const Real pdel) {
  // NOTE: chemical characteristics of our current mechanism are generated in
  // NOTE: eam/src/chemistry/pp_linoz_mam4_resus_mom_soag/mo_sim_dat.F90

  // NOTE: o3_exo_col is 0 because O2 does not appear in the solsym array.
  // NOTE: Further, it's only used to initialize the 0th vertical level, so
  // NOTE: we ignore it in this single-level calculation.

  // the following logic was extracted from the calc_col_delta subroutine under
  // the above conditions that o3_ndx == 0, o3_inv_ndx == -1, and O2 is not
  // a species or an invariant of interest
  constexpr Real xfactor = 2.8704e21 / (9.80616 * 1.38044); // BAD_CONSTANT!
  constexpr int o3_ndx = 0;
  o3_col_delta = xfactor * pdel * vmr[o3_ndx];
}

KOKKOS_INLINE_FUNCTION
void setcol(const Real o3_col_deltas[mam4::nlev + 1], ColumnView &o3_col_dens) {
  // we can probably accelerate this with a parallel_scan, but let's just do
  // a simple loop for now
  constexpr int nlev = mam4::nlev;
  o3_col_dens(0) = 0.5 * (o3_col_deltas[0] + o3_col_deltas[1]);
  for (int k = 1; k < nlev; ++k) {
    o3_col_dens(k) =
        o3_col_dens(k - 1) + 0.5 * (o3_col_deltas[k] + o3_col_deltas[k + 1]);
  }
}

KOKKOS_INLINE_FUNCTION
void find_index(const Real *var_in, const int var_len,
                const Real var_min, //  in
                int &idx_out)       // out
{

  /*------------------------------------------------------------------------------
   find the index of the first element in var_in(1:var_len) where the value is >
  var_min
  ------------------------------------------------------------------------------*/
  // @param[in]  var_in(var_len)    input variable
  // @param[in]  var_len   length of the input variable
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

KOKKOS_INLINE_FUNCTION
void calc_sum_wght(const Real dels[3], const Real wrk0, // in
                   const int iz, const int is, const int iv,
                   const int ial,         // in
                   const View5D &rsf_tab, // in
                   const int nw,
                   Real *psum) // out
{

  // @param[in]   dels(3)
  // @param[in]   wrk0
  // @param[in]  iz, is, iv, ial
  // @param[in]   nw// wavelengths >200nm
  // @param[in/out] psum(:)
  const int isp1 = is + 1;
  const int ivp1 = iv + 1;
  const int ialp1 = ial + 1;
  const Real one = 1;

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

  for (int wn = 0; wn < nw; wn++) {
    psum[wn] = wght_0_0_0 * rsf_tab(wn, iz, is, iv, ial) +
               wght_0_0_1 * rsf_tab(wn, iz, is, iv, ialp1) +
               wght_0_1_0 * rsf_tab(wn, iz, is, ivp1, ial) +
               wght_0_1_1 * rsf_tab(wn, iz, is, ivp1, ialp1) +
               wght_1_0_0 * rsf_tab(wn, iz, isp1, iv, ial) +
               wght_1_0_1 * rsf_tab(wn, iz, isp1, iv, ialp1) +
               wght_1_1_0 * rsf_tab(wn, iz, isp1, ivp1, ial) +
               wght_1_1_1 * rsf_tab(wn, iz, isp1, ivp1, ialp1);
  } // end wn
} // calc_sum_wght

KOKKOS_INLINE_FUNCTION
void interpolate_rsf(const Real *alb_in, const Real sza_in, const Real *p_in,
                     const Real *colo3_in,
                     const int kbot, //  in
                     const Real *sza, const Real *del_sza, const Real *alb,
                     const Real *press, const Real *del_p, const Real *colo3,
                     const Real *o3rat, const Real *del_alb,
                     const Real *del_o3rat, const Real *etfphot,
                     const View5D &rsf_tab, // in
                     const int nw, const int nump, const int numsza,
                     const int numcolo3, const int numalb,
                     const View2D &rsf, // out
                     // work array
                     Real *psum_l, Real *psum_u) {
  /*----------------------------------------------------------------------
          ... interpolate table rsf to model variables
  ----------------------------------------------------------------------*/

  // @param[in]  alb_in(:)        albedo [unitless]
  // @param[in]  sza_in           solar zenith angle [degrees]
  // @param[in]  p_in(:)          midpoint pressure [hPa]
  // @param[in]  colo3_in(:)      o3 column density [molecules/cm^3]
  // @param[in]  kbot             heating levels [level]
  // @param[in]  sza
  // @param[in]  del_sza
  // @param[in]  alb
  // @param[in]  press
  // @param[in]  del_p
  // @param[in]  colo3
  // @param[in]  o3rat
  // @param[in]  del_alb
  // @param[in]  del_o3rat
  // @param[in]  etfphot
  // @param[in]  rsf_tab
  // @param[in]  nw             wavelengths >200nm
  // @param[in]  nump           number of altitudes in rsf
  // @param[in]  numsza         number of zen angles in rsf
  // @param[in]  numcolo3       number of o3 columns in rsf
  // @param[in]  numalb         number of albedos in rsf
  // @param[out] rsf(:,:)       Radiative Source Function [quanta cm-2 sec-1]

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
    } else if (p_in[kk] <= press[nump - 1]) {
      // Fortran to C++ indexing
      pind = nump - 1;
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
      pind = haero::max(haero::min(iz, nump - 1), 1);
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
    calc_sum_wght(dels, wrk0,        // in
                  pind, is, iv, ial, // in
                  rsf_tab, nw,
                  psum_l); // out

    iv = ratindu;
    dels[1] =
        utils::min_max_bound(zero, one, (v3ratu - o3rat[iv]) * del_o3rat[iv]);
    calc_sum_wght(dels, wrk0,            // in
                  pind - 1, is, iv, ial, // in
                  rsf_tab, nw,
                  psum_u); //  inout

    for (int wn = 0; wn < nw; wn++) {
      rsf(wn, kk) = psum_l[wn] + wght1 * (psum_u[wn] - psum_l[wn]);
    }

    /*------------------------------------------------------------------------------
        etfphot comes in as photons/cm^2/sec/nm  (rsf includes the wlintv factor
     -- nm)
       ... --> convert to photons/cm^2/s
     ------------------------------------------------------------------------------*/
    for (int wn = 0; wn < nw; wn++) {
      rsf(wn, kk) *= etfphot[wn];
    } // end for wn

  } // end Level_loop

} // interpolate_rsf

//======================================================================================
KOKKOS_INLINE_FUNCTION
void jlong(const Real sza_in, const Real *alb_in, const Real *p_in,
           const Real *t_in, const Real *colo3_in, const View4D &xsqy,
           const Real *sza, const Real *del_sza, const Real *alb,
           const Real *press, const Real *del_p, const Real *colo3,
           const Real *o3rat, const Real *del_alb, const Real *del_o3rat,
           const Real *etfphot, const View5D &rsf_tab, const Real *prs,
           const Real *dprs, const int nw, const int nump, const int numsza,
           const int numcolo3, const int numalb, const int np_xs,
           const int numj,
           const View2D &j_long, // output
           // work arrays
           const View2D &rsf, const View2D &xswk, Real *psum_l,
           Real *psum_u) // out
{
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

  // @param[in] sza_in             ! solar zenith angle [degrees]
  // @param[in] alb_in(pver)       ! albedo
  // @param[in]  p_in(pver)         ! midpoint pressure [hPa]
  // @param[in]  t_in(pver)         ! Temperature profile [K]
  // @param[in]  colo3_in(pver)     ! o3 column density [molecules/cm^3]
  // @param[in]  xsqy
  // @param[in]  sza
  // @param[in]  del_sza
  // @param[in]  alb
  // @param[in]  press
  // @param[in]  del_p
  // @param[in]  colo3
  // @param[in]  o3rat
  // @param[in]  del_alb
  // @param[in]  del_o3rat
  // @param[in]  etfphot
  // @param[in]  rsf_tab
  // @param[in]  prs
  // @param[in]  dprs
  // @param[in]  nw             wavelengths >200nm
  // @param[in]  nump           number of altitudes in rsf
  // @param[in]  numsza         number of zen angles in rsf
  // @param[in]  numcolo3       number of o3 columns in rsf
  // @param[in]  numalb         number of albedos in rsf
  // @param[in]  np_xs          number of pressure levels in xsection table
  // @param[in]  numj           number of photorates in xsqy, rsf
  // @param[out]  j_long(:,:)   photo rates [1/s]

  /*----------------------------------------------------------------------
    ... interpolate table rsf to model variables
----------------------------------------------------------------------*/
  const Real zero = 0;
  interpolate_rsf(alb_in, sza_in, p_in, colo3_in, pver, sza, del_sza, alb,
                  press, del_p, colo3, o3rat, del_alb, del_o3rat, etfphot,
                  rsf_tab, //  in
                  nw, nump, numsza, numcolo3, numalb, rsf, psum_l,
                  psum_u); // out

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

  for (int kk = 0; kk < pver; kk++) {
    /*----------------------------------------------------------------------
      ... get index into xsqy
     ----------------------------------------------------------------------*/

    // Fortran indexing to C++ indexing
    // number of temperatures in xsection table
    // BAD CONSTANT for 201 and 148.5
    const int t_index = haero::min(201, haero::max(t_in[kk] - 148.5, 0)) - 1;

    /*----------------------------------------------------------------------
               ... find pressure level
     ----------------------------------------------------------------------*/
    const Real ptarget = p_in[kk];
    if (ptarget >= prs[0]) {
      for (int wn = 0; wn < nw; wn++) {
        for (int i = 0; i < numj; i++) {
          xswk(i, wn) = xsqy(i, wn, t_index, 0);
        } // end for i
      }   // end for wn
      // Fortran to C++ indexing conversion
    } else if (ptarget <= prs[np_xs - 1]) {
      for (int wn = 0; wn < nw; wn++) {
        for (int i = 0; i < numj; i++) {
          // Fortran to C++ indexing conversion
          xswk(i, wn) = xsqy(i, wn, t_index, np_xs - 1);
        } // end for i
      }   // end for wn

    } else {
      Real delp = zero;
      int pndx = 0;
      // Question: delp is not initialized in fortran code. What if the
      // following code does not satify this if condition: ptarget >= prs[km]
      // Conversion indexing from Fortran to C++
      for (int km = 1; km < np_xs; km++) {
        if (ptarget >= prs[km]) {
          pndx = km - 1;
          delp = (prs[pndx] - ptarget) * dprs[pndx];
          break;
        } // end if

      } // end for km
      for (int wn = 0; wn < nw; wn++) {
        for (int i = 0; i < numj; i++) {
          xswk(i, wn) = xsqy(i, wn, t_index, pndx) +
                        delp * (xsqy(i, wn, t_index, pndx + 1) -
                                xsqy(i, wn, t_index, pndx));

        } // end for i
      }   // end for wn
    }     // end if

    // j_long(:,kk) = matmul( xswk(:,:),rsf(:,kk) )
    for (int i = 0; i < numj; ++i) {
      Real suma = zero;
      for (int wn = 0; wn < nw; wn++) {
        suma += xswk(i, wn) * rsf(wn, kk);
      } // wn
      j_long(i, kk) = suma;
    } // i
  }   // end kk

} // jlong

// FIXME: note the use of ConstColumnView for views we get from the
// FIXME: haero::Atmosphere type
KOKKOS_INLINE_FUNCTION
void table_photo(const View2D &photo, // out
                 const ConstColumnView &pmid, const ConstColumnView &pdel,
                 const ConstColumnView &temper, // in
                 const ColumnView &colo3_in, const Real zen_angle,
                 const Real srf_alb, const ColumnView &lwc,
                 const ConstColumnView &clouds, // in
                 const Real esfact, const PhotoTableData &table_data,
                 PhotoTableWorkArrays &work_arrays) {
  /*-----------------------------------------------------------------
      ... table photorates for wavelengths > 200nm
 -----------------------------------------------------------------*/

  //@param[out] photo(pver,phtcnt)  column photodissociation rates [1/s]
  //@param[in]  pmid(pver)          midpoint pressure [Pa]
  //@param[in]  pdel(pver)          pressure delta about midpoint [Pa]
  //@param[in]  temper(pver)        midpoint temperature [K]
  //@param[in]  colo3_in(pver)      column densities [molecules/cm^2]
  //@param[in]  zen_angle(icol)     solar zenith angle [radians]
  //@param[in]  srf_alb(icols)      surface albedo
  //@param[in]  lwc(icol,pver)      liquid water content [kg/kg]
  //@param[in]  clouds(icol,pver)   cloud fraction
  //@param[in]  esfact              earth sun distance factor
  //@param[in]  table_data          column-independent photolysis table data
  //@param[out] work_arrays         column-specific photolysis work arrays

  if (phtcnt < 1) {
    return;
  }

  constexpr Real zero = 0;

  constexpr Real Pa2mb = 1.e-2;                      // pascals to mb
  constexpr Real r2d = 180.0 / haero::Constants::pi; // degrees to radians
  // BAD CONSTANT
  constexpr Real max_zen_angle = 88.85; //  degrees

  // vertical pressure array [hPa]
  Real parg[pver] = {};
  Real eff_alb[pver] = {};
  Real cld_mult[pver] = {};

  /*-----------------------------------------------------------------
    ... zero all photorates
    -----------------------------------------------------------------*/
  const Real sza_in = zen_angle * r2d;
  // daylight
  if (sza_in >= zero && sza_in < max_zen_angle) {
    /*-----------------------------------------------------------------
         ... compute eff_alb and cld_mult -- needs to be before jlong
    -----------------------------------------------------------------*/
    cloud_mod(zen_angle, clouds.data(), lwc.data(), pdel.data(),
              srf_alb, //  in
              eff_alb, cld_mult);

    for (int kk = 0; kk < pver; ++kk) {
      parg[kk] = pmid(kk) * Pa2mb;
      cld_mult[kk] *= esfact;
    } // kk
    /*-----------------------------------------------------------------
     ... long wave length component
    -----------------------------------------------------------------*/

    jlong(sza_in, eff_alb, parg, temper.data(), colo3_in.data(),
          table_data.xsqy, table_data.sza.data(), table_data.del_sza.data(),
          table_data.alb.data(), table_data.press.data(),
          table_data.del_p.data(), table_data.colo3.data(),
          table_data.o3rat.data(), table_data.del_alb.data(),
          table_data.del_o3rat.data(), table_data.etfphot.data(),
          table_data.rsf_tab, // in
          table_data.prs.data(), table_data.dprs.data(), table_data.nw,
          table_data.nump, table_data.numsza, table_data.numcolo3,
          table_data.numalb, table_data.np_xs, table_data.numj,
          work_arrays.lng_prates, // output
          // work arrays
          work_arrays.rsf, work_arrays.xswk, work_arrays.psum_l.data(),
          work_arrays.psum_u.data());

    for (int mm = 0; mm < phtcnt; ++mm) {
      if (table_data.lng_indexer(mm) > -1) {
        for (int kk = 0; kk < pver; ++kk) {
          photo(kk, mm) =
              cld_mult[kk] *
              (photo(kk, mm) +
               table_data.pht_alias_mult_1(mm) *
                   work_arrays.lng_prates(table_data.lng_indexer(mm), kk));
        } // end kk
      }   // end if
    }     // end mm
  }
  // } // end col_loop
}

} // namespace mo_photo
} // end namespace mam4

#endif
