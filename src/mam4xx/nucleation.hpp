#ifndef MAM4XX_NUCLEATION_HPP
#define MAM4XX_NUCLEATION_HPP

#include <mam4xx/aero_config.hpp>
#include <mam4xx/conversions.hpp>
#include <mam4xx/merikanto2007.hpp>
#include <mam4xx/vehkamaki2002.hpp>
#include <mam4xx/wang2008.hpp>

#include <haero/atmosphere.hpp>
#include <haero/math.hpp>

namespace mam4 {

using Pack = PackType;

using haero::max;

//-----------------------------------------------------------------------------
// The following functions were ported from aero_newnuc_utils.F90 in the MAM4
// box model.
//-----------------------------------------------------------------------------

namespace nucleation {

//--------------------------------------------------------
// calculates boundary nucleation nucleation rate
// using the linear or quadratic parameterization in
// Wang, M. and J.E. Penner, 2008,
// Aerosol indirect forcing in a global model with particle nucleation,
// Atmos. Chem. Phys. Discuss., 8, 13943-13998
// Atmos. Chem. Phys.  9, 239–260, 2009
//--------------------------------------------------------
KOKKOS_INLINE_FUNCTION
void pbl_nuc_wang2008(const Pack& so4vol, Real pi,
                      int pbl_nuc_wang2008_user_choice,
                      Real adjust_factor_pbl_ratenucl,
                      IntPack& pbl_nuc_wang2008_actual, Pack& ratenucl,
                      Pack& rateloge, Pack& cnum_tot, Pack& cnum_h2so4,
                      Pack& cnum_nh3, Pack& radius_cluster_nm) {
  // subr arguments (in)
  // real(wp), intent(in) :: pi                           ! pi
  // real(wp), intent(in) :: so4vol                       ! concentration of
  // h2so4 (molecules cm-3) integer,  intent(in) :: pbl_nuc_wang2008_user_choice
  // ! 1 = first-order, 2 = second-order scheme real(wp), intent(in) ::
  // adjust_factor_pbl_ratenucl   ! tuning parameter

  // subr arguments (inout or out)
  // integer,  intent(out)   :: pbl_nuc_wang2008_actual   ! 1 = first-order, 2 =
  // second-order scheme, 0 = none real(wp), intent(inout) :: ratenucl ! binary
  // nucleation rate, j (# cm-3 s-1) real(wp), intent(inout) :: rateloge ! log(
  // ratenucl ) real(wp), intent(inout) :: cnum_tot                  ! total
  // number of molecules in one cluster real(wp), intent(inout) :: cnum_h2so4 !
  // number of h2so4 molecules in one cluster real(wp), intent(inout) ::
  // cnum_nh3                  ! number of nh3 molecules   in one cluster
  // real(wp), intent(inout) :: radius_cluster_nm         ! the radius of a
  // cluster in nm

  Pack tmp_diam, tmp_mass, tmp_volu;
  Pack tmp_rateloge, tmp_ratenucl;

  constexpr Real mw_h2so4_gmol = 98.0;
  constexpr Real avogadro_mol = 6.023e23;
  constexpr Real density_sulfate_gcm3 = 1.8;

  //-----------------------------------------------------------------
  // Initialize the pbl_nuc_wang2008_actual flag. Assumed default is
  // no PBL nucleation.
  //-----------------------------------------------------------------
  pbl_nuc_wang2008_actual = 0;

  //-------------------------------------------------------------
  // Calculate nucleation rate using incoming so4 concentration.
  //-------------------------------------------------------------
  if (pbl_nuc_wang2008_user_choice == 1) {
    tmp_ratenucl = wang2008::first_order_pbl_nucleation_rate(so4vol);
  } else if (pbl_nuc_wang2008_user_choice == 2) {
    tmp_ratenucl = wang2008::second_order_pbl_nucleation_rate(so4vol);
  } else {
    return;
  }

  // Scale the calculated PBL nuc rate by user-specificed tuning factor
  tmp_ratenucl = tmp_ratenucl * adjust_factor_pbl_ratenucl;
  tmp_rateloge = log(max(1.0e-38, tmp_ratenucl));

  //------------------------------------------------------------------
  // If PBL nuc rate is lower than the incoming ternary/binary rate,
  // discard the PBL nuc rate (i.e, do not touch any incoming value).
  // Otherwise, use the PBL nuc rate.
  //------------------------------------------------------------------
  auto pbl_rate_sufficient = (tmp_rateloge > rateloge);
  pbl_nuc_wang2008_actual.set(pbl_rate_sufficient,
                              pbl_nuc_wang2008_user_choice);
  rateloge.set(pbl_rate_sufficient, tmp_rateloge);
  ratenucl.set(pbl_rate_sufficient, tmp_ratenucl);

  // following wang 2002, assume fresh nuclei are 1 nm diameter
  // subsequent code will "grow" them to aitken mode size
  radius_cluster_nm.set(pbl_rate_sufficient, 0.5);

  // assume fresh nuclei are pure h2so4
  //    since aitken size >> initial size, the initial composition
  //    has very little impact on the results

  tmp_diam = radius_cluster_nm * 2.0e-7;       // diameter in cm
  tmp_volu = cube(tmp_diam) * (pi / 6.0);      // volume in cm^3
  tmp_mass = tmp_volu * density_sulfate_gcm3;  // mass in g

  // no. of h2so4 molec per cluster assuming pure h2so4
  cnum_h2so4.set(pbl_rate_sufficient,
                 (tmp_mass / mw_h2so4_gmol) * avogadro_mol);
  cnum_nh3.set(pbl_rate_sufficient, 0.0);
  cnum_tot.set(pbl_rate_sufficient, cnum_h2so4);
}

//-----------------------------------------------------------------
// calculates binary nucleation rate and critical cluster size
// using the parameterization in
//     vehkamäki, h., m. kulmala, i. napari, k.e.j. lehtinen,
//        c. timmreck, m. noppel and a. laaksonen, 2002,
//        an improved parameterization for sulfuric acid-water nucleation
//        rates for tropospheric and stratospheric conditions,
//        j. geophys. res., 107, 4622, doi:10.1029/2002jd002184
//-----------------------------------------------------------------
KOKKOS_INLINE_FUNCTION
void binary_nuc_vehk2002(const Pack& temp, const Pack& rh, const Pack& so4vol,
                         Pack& ratenucl, Pack& rateloge, Pack& cnum_h2so4,
                         Pack& cnum_tot, Pack& radius_cluster) {
  // arguments (in)
  // real(wp), intent(in) :: temp              ! temperature (k)
  // real(wp), intent(in) :: rh                ! relative humidity (0-1)
  // real(wp), intent(in) :: so4vol            ! concentration of h2so4
  // (molecules cm-3)

  // arguments (out)
  // real(wp), intent(out) :: ratenucl         ! binary nucleation rate, j (#
  // cm-3 s-1) real(wp), intent(out) :: rateloge         ! log( ratenucl )
  //
  // real(wp), intent(out) :: cnum_h2so4       ! number of h2so4 molecules
  //                                           ! in the critical nucleus
  // real(wp), intent(out) :: cnum_tot         ! total number of molecules
  //                                           ! in the critical nucleus
  // real(wp), intent(out) :: radius_cluster   ! the radius of cluster (nm)

  // calc sulfuric acid mole fraction in critical cluster
  // following eq. (11) in Vehkamäki et al. (2002)
  Pack x_crit =
      vehkamaki2002::h2so4_critical_mole_fraction(so4vol, temp, rh);

  // calc nucleation rate
  // following eq. (12) in Vehkamäki et al. (2002)
  rateloge = vehkamaki2002::nucleation_rate(so4vol, temp, rh, x_crit);
  ratenucl = exp(min(rateloge, log(1e38)));

  // calc number of molecules in critical cluster
  // following eq. (13) in Vehkamäki et al. (2002)
  cnum_tot =
      vehkamaki2002::num_critical_molecules(so4vol, temp, rh, x_crit);

  cnum_h2so4 = cnum_tot * x_crit;

  // calc radius (nm) of critical cluster
  // following eq. (14) in Vehkamäki et al. (2002)
  radius_cluster = vehkamaki2002::critical_radius(x_crit, cnum_tot);
}

//-----------------------------------------------------------------------------------------
// calculates the parameterized composition and nucleation rate of critical
// clusters in h2o-h2so4-nh3 vapor
// warning: the fit should not be used outside its limits of validity
// (limits indicated below)
//
// in:
// t:     temperature (k), limits 235-295 k
// rh:    relative humidity as fraction (eg. 0.5=50%) limits 0.05-0.95
// c2:    sulfuric acid concentration (molecules/cm3) limits 5x10^4 - 10^9
// molecules/cm3 c3:    ammonia mixing ratio (ppt) limits 0.1 - 1000 ppt
//
// out:
// j_log: logarithm of nucleation rate (1/(s cm3))
// ntot:  total number of molecules in the critical cluster
// nacid: number of sulfuric acid molecules in the critical cluster
// namm:  number of ammonia molecules in the critical cluster
// r:     radius of the critical cluster (nm)
//-----------------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION
void ternary_nuc_merik2007(const Pack& t, const Pack& rh, const Pack& c2,
                           const Pack& c3, Pack& j_log, Pack& ntot, Pack& nacid,
                           Pack& namm, Pack& r) {
  Pack t_onset = merikanto2007::onset_temperature(rh, c2, c3);

  // Set log(J) assuming no nucleation.
  j_log = -300.;

  // If t_onset > t, nucleation occurs.
  auto nuc_occurs = (t_onset > t);

  j_log.set(nuc_occurs, merikanto2007::log_nucleation_rate(t, rh, c2, c3));
  ntot.set(nuc_occurs, merikanto2007::num_critical_molecules(j_log, t, c2, c3));
  r.set(nuc_occurs, merikanto2007::critical_radius(j_log, t, c2, c3));
  nacid.set(nuc_occurs, merikanto2007::num_h2so4_molecules(j_log, t, c2, c3));
  namm.set(nuc_occurs, merikanto2007::num_nh3_molecules(j_log, t, c2, c3));
}

//-----------------------------------------------------------------------------
// Calculates new particle production from homogeneous nucleation
// using nucleation rates from either
// Merikanto et al. (2007) h2so4-nh3-h2o ternary parameterization
// Vehkamaki et al. (2002) h2so4-h2o binary parameterization
//
// References:
// * merikanto, j., i. napari, h. vehkamaki, t. anttila,
//   and m. kulmala, 2007, new parameterization of
//   sulfuric acid-ammonia-water ternary nucleation
//   rates at tropospheric conditions,
//   j. geophys. res., 112, d15207, doi:10.1029/2006jd0027977
//
// * vehkamäki, h., m. kulmala, i. napari, k.e.j. lehtinen,
//   c. timmreck, m. noppel and a. laaksonen, 2002,
//   an improved parameterization for sulfuric acid-water nucleation
//   rates for tropospheric and stratospheric conditions,
//   j. geophys. res., 107, 4622, doi:10.1029/2002jd002184
//
// * Wang, M. and J.E. Penner, 2008,
//   Aerosol indirect forcing in a global model with particle nucleation,
//   Atmos. Chem. Phys. Discuss., 8, 13943-13998
//   Atmos. Chem. Phys.  9, 239–260, 2009
KOKKOS_INLINE_FUNCTION
void mer07_veh02_wang08_nuc_1box(
    int newnuc_method_user_choice, IntPack& newnuc_method_actual,  // in, out
    int pbl_nuc_wang2008_user_choice,
    IntPack& pbl_nuc_wang2008_actual,                           // in, out
    Real ln_nuc_rate_cutoff,                                    // in
    Real adjust_factor_bin_tern_ratenucl,                       // in
    Real adjust_factor_pbl_ratenucl,                            // in
    Real pi, const Pack& so4vol_in, const Pack& nh3ppt_in,      // in
    const Pack& temp_in, const Pack& rh_in, const Pack& zm_in,  // in
    Real pblh_in,                                               // in
    Pack& dnclusterdt, Pack& rateloge, Pack& cnum_h2so4,        // out
    Pack& cnum_nh3, Pack& radius_cluster) {                     // out

  Pack rh_bb;      // bounded value of rh_in
  Pack so4vol_bb;  // bounded value of so4vol_in (molecules per cm3)
  Pack temp_bb;    // bounded value of temp_in (K)
  Pack nh3ppt_bb;  // bounded nh3 (ppt)

  Pack cnum_tot;   // total number of molecules in a cluster
  Pack ratenuclt;  // J: nucleation rate from parameterization.
                   // # of clusters/nuclei per cm3 per s

  //---------------------------------------------------------------
  // Set "effective zero"
  //---------------------------------------------------------------
  ratenuclt = 1.0e-38;
  rateloge = log(ratenuclt);

  //---------------------------------------------------------------
  // Make call to merikanto ternary parameterization routine
  // if nitrate aerosol is considered in the aerosol population
  // and ammonia concentration is non-negligible
  //---------------------------------------------------------------
  auto nh3_present = (nh3ppt_in >= 0.1);
  newnuc_method_actual = 0;
  if ((newnuc_method_user_choice == 3) && nh3_present.any()) {
    auto enough_so4 = (so4vol_in >= 5.0e4);
    temp_bb = max(235.0, min(295.0, temp_in));
    rh_bb = max(0.05, min(0.95, rh_in));
    so4vol_bb = max(5.0e4, min(1.0e9, so4vol_in));
    nh3ppt_bb = max(0.1, min(1.0e3, nh3ppt_in));
    Pack rateloge_bb, cnum_tot_bb, cnum_h2so4_bb, cnum_nh3_bb,
        radius_cluster_bb;
    ternary_nuc_merik2007(temp_bb, rh_bb, so4vol_bb, nh3ppt_bb, rateloge_bb,
                          cnum_tot_bb, cnum_h2so4_bb, cnum_nh3_bb,
                          radius_cluster_bb);
    rateloge.set(enough_so4, rateloge_bb);
    cnum_tot.set(enough_so4, cnum_tot_bb);
    cnum_h2so4.set(enough_so4, cnum_h2so4_bb);
    cnum_nh3.set(enough_so4, cnum_nh3_bb);
    radius_cluster.set(enough_so4, radius_cluster_bb);
    newnuc_method_actual.set(enough_so4, 3);

    //---------------------------------------------------------------------
    // Otherwise, make call to vehkamaki binary parameterization routine
    //---------------------------------------------------------------------
  } else {
    auto enough_so4 = (so4vol_in >= 1.0e4);
    temp_bb = max(230.15, min(305.15, temp_in));
    rh_bb = max(1.0e-4, min(1.0, rh_in));
    so4vol_bb = max(1.0e4, min(1.0e11, so4vol_in));
    Pack ratenuclt_bb, rateloge_bb, cnum_tot_bb, cnum_h2so4_bb,
        radius_cluster_bb;
    binary_nuc_vehk2002(temp_bb, rh_bb, so4vol_bb, ratenuclt_bb, rateloge_bb,
                        cnum_h2so4_bb, cnum_tot_bb, radius_cluster_bb);
    rateloge.set(enough_so4, rateloge_bb);
    cnum_tot.set(enough_so4, cnum_tot_bb);
    cnum_h2so4.set(enough_so4, cnum_h2so4_bb);
    cnum_nh3 = 0.0;
    radius_cluster.set(enough_so4, radius_cluster_bb);
    newnuc_method_actual.set(enough_so4, 2);
  }

  rateloge += log(max(1.0e-38, adjust_factor_pbl_ratenucl));

  //---------------------------------------------------------------------
  // Do boundary layer nuc
  //---------------------------------------------------------------------
  pbl_nuc_wang2008_actual = 0;
  auto inside_pbl = (zm_in <= max(pblh_in, 100.0));
  if ((pbl_nuc_wang2008_user_choice != 0) && inside_pbl.any()) {
    so4vol_bb = so4vol_in;
    IntPack pbl_nuc_wang2008_actual_bb;
    Pack ratenuclt_bb = ratenuclt;
    Pack rateloge_bb = rateloge;
    Pack cnum_tot_bb = cnum_tot;
    Pack cnum_h2so4_bb = cnum_h2so4;
    Pack cnum_nh3_bb = cnum_nh3;
    Pack radius_cluster_bb = radius_cluster;
    pbl_nuc_wang2008(so4vol_bb, pi, pbl_nuc_wang2008_user_choice,
                     adjust_factor_pbl_ratenucl, pbl_nuc_wang2008_actual_bb,
                     ratenuclt_bb, rateloge_bb, cnum_tot_bb, cnum_h2so4_bb,
                     cnum_nh3_bb, radius_cluster_bb);
    pbl_nuc_wang2008_actual.set(inside_pbl, pbl_nuc_wang2008_actual_bb);
    ratenuclt.set(inside_pbl, ratenuclt_bb);
    rateloge.set(inside_pbl, rateloge_bb);
    cnum_tot.set(inside_pbl, cnum_tot_bb);
    cnum_h2so4.set(inside_pbl, cnum_h2so4_bb);
    cnum_nh3.set(inside_pbl, cnum_nh3_bb);
    radius_cluster.set(inside_pbl, radius_cluster_bb);
  }

  //---------------------------------------------------------------------
  // if nucleation rate is less than 1e-6 #/cm3/s ~= 0.1 #/cm3/day,
  // exit with new particle formation = 0. Otherwise, calculate the
  // nucleation rate in #/m3/s
  //---------------------------------------------------------------------
  dnclusterdt.set((rateloge > ln_nuc_rate_cutoff), exp(rateloge) * 1.0e6);
  // ratenuclt is #/cm3/s; dnclusterdt is #/m3/s
}

KOKKOS_INLINE_FUNCTION
void newnuc_cluster_growth(
    const Pack& ratenuclt_bb, const Pack& cnum_h2so4, const Pack& cnum_nh3,
    const Pack& radius_cluster, const Pack* dplom_sect, const Pack* dphim_sect,
    int nsize, Real dtnuc, const Pack& temp_in, const Pack& rh_in,
    const Pack& cair, Real accom_coef_h2so4, Real mw_so4a, Real mw_so4a_host,
    Real mw_nh4a, Real avogad, Real pi, const Pack& qnh3_cur,
    const Pack& qh2so4_cur, const Pack& so4vol_in, const Pack& h2so4_uptkrate,
    IntPack& isize_nuc, Pack& dens_nh4so4a, Pack& qh2so4_del, Pack& qnh3_del,
    Pack& qso4a_del, Pack& qnh4a_del, Pack& qnuma_del) {
  Pack tmpa;
  Pack tmpb, tmpe;
  Pack voldry_clus;       // critical-cluster dry volume (m3)
  Pack voldry_part;       // "grown" single-particle dry volume (m3)
  Pack wetvol_dryvol;     // grown particle (wet-volume)/(dry-volume)
  Pack wet_volfrac_so4a;  // grown particle (dry-volume-from-so4)/(wet-volume)
  Pack dpdry_part;        // "grown" single-particle dry diameter (m)
  Pack dpdry_clus;        // critical cluster diameter (m)

  Pack cs_prime_kk;       // kk2002 "cs_prime" parameter (1/m2)
  Pack cs_kk;             // kk2002 "cs" parameter (1/s)
  Pack dfin_kk, dnuc_kk;  // kk2002 final/initial new particle wet diameter (nm)
  Pack tmpa1, tmpb1;
  Pack tmp_spd;           // h2so4 vapor molecular speed (m/s)
  Pack qmolnh4a_del_max;  // max production of aerosol nh4 over dtnuc
                          // (mol/mol-air)
  Pack qmolso4a_del_max;  // max production of aerosol so4 over dtnuc
                          // (mol/mol-air)
  Pack ratenuclt_kk;      // nucleation rate after kk2002 adjustment (#/m3/s)

  IntPack igrow;

  Pack tmp_n1, tmp_n2, tmp_n3;
  Pack tmp_m1, tmp_m2, tmp_m3;
  Pack dens_part;              // "grown" single-particle dry density (kg/m3)
  Pack mass_part;              // "grown" single-particle dry mass (kg)
  Pack molenh4a_per_moleso4a;  // (mol aerosol nh4)/(mol aerosol so4)
  Pack kgaero_per_moleso4a;    // (kg dry aerosol)/(mol aerosol so4)
  Pack factor_kk;
  Pack freduce;  // reduction factor applied to nucleation rate
                 // due to limited availability of h2so4 & nh3 gases
  Pack freducea, freduceb;
  Pack gamma_kk;  // kk2002 "gamma" parameter (nm2*m2/h)
  Pack gr_kk;     // kk2002 "gr" parameter (nm/h)
  Pack nu_kk;     // kk2002 "nu" parameter (nm)

  constexpr Real onethird = 1.0 / 3.0;

  // dry densities (kg/m3) molecular weights of aerosol
  // ammsulf, ammbisulf, and sulfacid (from mosaic  dens_electrolyte values)
  //  Real dens_ammsulf   = 1.769e3
  //  Real dens_ammbisulf = 1.78e3
  //  Real dens_sulfacid  = 1.841e3
  // use following to match cam3 modal_aero densities
  constexpr Real dens_ammsulf = 1.770e3;
  constexpr Real dens_ammbisulf = 1.770e3;
  constexpr Real dens_sulfacid = 1.770e3;

  // molecular weights (g/mol) of aerosol ammsulf, ammbisulf, and sulfacid
  // for ammbisulf and sulfacid, use 114 & 96 here rather than 115 & 98
  // because we don't keep track of aerosol hion mass
  constexpr Real mw_ammsulf = 132.0;
  constexpr Real mw_ammbisulf = 114.0;
  constexpr Real mw_sulfacid = 96.0;

  // wet/dry volume ratio - use simple kohler approx for ammsulf/ammbisulf
  tmpa = max(0.10, min(0.95, rh_in));
  wetvol_dryvol = 1.0 - 0.56 / log(tmpa);

  // determine size bin into which the new particles go
  // (probably it will always be bin #1, but ...)
  voldry_clus = (max(cnum_h2so4, 1.0) * mw_so4a + cnum_nh3 * mw_nh4a) /
                (1.0e3 * dens_sulfacid * avogad);

  // correction when host code sulfate is really ammonium bisulfate/sulfate
  voldry_clus = voldry_clus * (mw_so4a_host / mw_so4a);
  dpdry_clus = pow(voldry_clus * 6.0 / pi, onethird);

  isize_nuc = 1;
  dpdry_part = dplom_sect[0];
  igrow = 0;
  auto dpdry_clus_lo = (dpdry_clus <= dplom_sect[0]);
  igrow.set(dpdry_clus_lo, 1);
  auto dpdry_clus_hi = (dpdry_clus >= dphim_sect[nsize-1]);
  isize_nuc.set(dpdry_clus_hi, nsize);
  dpdry_part.set(dpdry_clus_hi, dphim_sect[nsize-1]);
  for (int i = 0; i < nsize; ++i) {
    auto dpdry_clus_i = (dpdry_clus < dphim_sect[i]);
    isize_nuc.set(dpdry_clus_i, i);
    dpdry_part.set(dpdry_clus_i, dpdry_clus);
    dpdry_part.set(dpdry_clus_i, min(dpdry_part, dphim_sect[i]));
    dpdry_part.set(dpdry_clus_i, max(dpdry_part, dplom_sect[i]));
  }
  voldry_part = (pi / 6.0) * cube(dpdry_part);

  // determine composition and density of the "grown particles"
  // the grown particles are assumed to be liquid
  //    (since critical clusters contain water)
  //    so any (nh4/so4) molar ratio between 0 and 2 is allowed
  // assume that the grown particles will have
  //    (nh4/so4 molar ratio) = min( 2, (nh3/h2so4 gas molar ratio) )
  auto no_growth = (igrow <= 0);
  tmp_n1.set(no_growth, 0.0);
  tmp_n2.set(no_growth, 0.0);
  tmp_n3.set(no_growth, 0.0);

  auto more_nh3 = (qnh3_cur >= qh2so4_cur);
  // combination of ammonium sulfate and ammonium bisulfate
  // tmp_n1 & tmp_n2 = mole fractions of the ammsulf & ammbisulf
  tmp_n1.set(more_nh3, (qnh3_cur / qh2so4_cur) - 1.0);
  tmp_n1.set(more_nh3, max(0, min(0, tmp_n1)));
  tmp_n2.set(more_nh3, 1.0 - tmp_n1);
  tmp_n3.set(more_nh3, 0.0);

  auto more_h2so4 = !no_growth && !more_nh3;
  // combination of ammonium bisulfate and sulfuric acid
  // tmp_n2 & tmp_n3 = mole fractions of the ammbisulf & sulfacid
  tmp_n1.set(more_h2so4, 0.0);
  tmp_n2.set(more_h2so4, qnh3_cur / qh2so4_cur);
  tmp_n2.set(more_h2so4, max(0.0, min(1.0, tmp_n2)));
  tmp_n3.set(more_h2so4, 1.0 - tmp_n2);

  tmp_m1 = tmp_n1 * mw_ammsulf;
  tmp_m2 = tmp_n2 * mw_ammbisulf;
  tmp_m3 = tmp_n3 * mw_sulfacid;
  dens_part = (tmp_m1 + tmp_m2 + tmp_m3) /
              ((tmp_m1 / dens_ammsulf) + (tmp_m2 / dens_ammbisulf) +
               (tmp_m3 / dens_sulfacid));
  dens_nh4so4a = dens_part;
  mass_part = voldry_part * dens_part;

  // (mol aerosol nh4)/(mol aerosol so4)
  molenh4a_per_moleso4a = 2.0 * tmp_n1 + tmp_n2;

  // (kg dry aerosol)/(mol aerosol so4)
  kgaero_per_moleso4a = 1.0e-3 * (tmp_m1 + tmp_m2 + tmp_m3);

  // correction when host code sulfate is really ammonium bisulfate/sulfate
  kgaero_per_moleso4a = kgaero_per_moleso4a * (mw_so4a_host / mw_so4a);

  // fraction of wet volume due to so4a
  tmpb = 1.0 + molenh4a_per_moleso4a * 17.0 / 98.0;
  wet_volfrac_so4a = 1.0 / (wetvol_dryvol * tmpb);

  // calc kerminen & kulmala (2002) correction
  factor_kk.set(no_growth, 1.0);
  auto growth = !no_growth;

  // "gr" parameter (nm/h) = condensation growth rate of new particles
  // use kk2002 eqn 21 for h2so4 uptake, and correct for nh3 & h2o uptake
  tmp_spd.set(growth, 14.7 * sqrt(temp_in));  // h2so4 molecular speed (m/s)
  gr_kk.set(growth, 3.0e-9 * tmp_spd * mw_sulfacid * so4vol_in /
                        (dens_part * wet_volfrac_so4a));

  // "gamma" parameter (nm2/m2/h)
  // use kk2002 eqn 22
  // dfin_kk = wet diam (nm) of grown particle having dry dia = dpdry_part (m)
  dfin_kk.set(growth, 1.0e9 * dpdry_part * pow(wetvol_dryvol, onethird));

  // dnuc_kk = wet diam (nm) of cluster
  dnuc_kk.set(growth, 2.0 * radius_cluster);
  dnuc_kk.set(growth, max(dnuc_kk, 1.0));

  // neglect (dmean/150)**0.048 factor,
  // which should be very close to 1.0 because of small exponent
  gamma_kk.set(growth, 0.23 * pow(dnuc_kk, 0.2) * pow(dfin_kk / 3.0, 0.075) *
                           pow(dens_part * 1.0e-3, -0.33) *
                           pow(temp_in / 293.0, -0.75));

  // "cs_prime parameter" (1/m2)
  // instead kk2002 eqn 3, use
  //     cs_prime ~= tmpa / (4*pi*tmpb * h2so4_accom_coef)
  // where
  //     tmpa = -d(ln(h2so4))/dt by conden to particles   (1/h units)
  //     tmpb = h2so4 vapor diffusivity (m2/h units)
  // this approx is generally within a few percent of the cs_prime
  //     calculated directly from eqn 2,
  //     which is acceptable, given overall uncertainties
  // tmpa = -d(ln(h2so4))/dt by conden to particles   (1/h units)
  tmpa.set(growth, h2so4_uptkrate * 3600.0);
  tmpa1.set(growth, max(tmpa, 0.0));

  // tmpb = h2so4 gas diffusivity (m2/s, then m2/h)
  tmpb.set(growth, 6.7037e-6 * (pow(temp_in, 0.75) / cair));
  tmpb1.set(growth, tmpb * 3600.0);  // m2/s -> m2/h
  cs_prime_kk.set(growth, tmpa / (4.0 * pi * tmpb * accom_coef_h2so4));
  cs_kk.set(growth, cs_prime_kk * 4.0 * pi * tmpb1);

  // "nu" parameter (nm) -- kk2002 eqn 11
  nu_kk.set(growth, gamma_kk * cs_prime_kk / gr_kk);

  // nucleation rate adjustment factor (--) -- kk2002 eqn 13
  factor_kk.set(growth, exp((nu_kk / dfin_kk) - (nu_kk / dnuc_kk)));

  ratenuclt_kk = ratenuclt_bb * factor_kk;

  // max production of aerosol dry mass (kg-aero/m3-air)
  tmpa = max(0.0, (ratenuclt_kk * dtnuc * mass_part));
  // max production of aerosol so4 (mol-so4a/mol-air)
  tmpe = tmpa / (kgaero_per_moleso4a * cair);
  // max production of aerosol so4 (mol/mol-air)
  // based on ratenuclt_kk and mass_part
  qmolso4a_del_max = tmpe;

  // check if max production exceeds available h2so4 vapor
  freducea = 1.0;
  auto not_enough_h2so4 = (qmolso4a_del_max > qh2so4_cur);
  freducea.set(not_enough_h2so4, qh2so4_cur / qmolso4a_del_max);

  // check if max production exceeds available nh3 vapor
  freduceb = 1.0;
  auto not_enough_nh3 = (molenh4a_per_moleso4a >= 1.0e-10);
  // max production of aerosol nh4 (ppm) based on ratenuclt_kk and mass_part
  qmolnh4a_del_max.set(not_enough_nh3,
                       qmolso4a_del_max * molenh4a_per_moleso4a);
  freduceb.set(not_enough_nh3 && (qmolnh4a_del_max > qnh3_cur),
               qnh3_cur / qmolnh4a_del_max);

  // get the final reduction factor for nucleation rate
  freduce = min(freducea, freduceb);

  //---------------------------------------------------------------------------
  // Derive aerosol and gas mass mass increments and aerosol number increments
  // for output to calling routine
  //---------------------------------------------------------------------------
  // if adjusted nucleation rate is less than 1e-12 #/m3/s ~= 0.1 #/cm3/day,
  // exit with new particle formation = 0
  auto nuc_too_small = (freduce * ratenuclt_kk <= 1.0e-12);
  qh2so4_del.set(nuc_too_small, 0.0);
  qnh3_del.set(nuc_too_small, 0.0);
  qso4a_del.set(nuc_too_small, 0.0);
  qnh4a_del.set(nuc_too_small, 0.0);
  qnuma_del.set(nuc_too_small, 0.0);

  //-------------------------------------------------------------------------
  // note:  suppose that at this point, freduce < 1.0 (no gas-available
  //    constraints) and molenh4a_per_moleso4a < 2.0
  // if the gas-available constraints is do to h2so4 availability,
  //    then it would be possible to condense "additional" nh3 and have
  //    (nh3/h2so4 gas molar ratio) < (nh4/so4 aerosol molar ratio) <= 2
  // one could do some additional calculations of
  //    dens_part & molenh4a_per_moleso4a to realize this
  // however, the particle "growing" is a crude approximate way to get
  //    the new particles to the host code's minimum particle size,
  // are such refinements worth the effort?
  //-------------------------------------------------------------------------
  // changes to h2so4 & nh3 gas (in mol/mol-air), limited by amounts available
  auto enough_nuc = !nuc_too_small;
  tmpa.set(enough_nuc, 0.9999);
  qh2so4_del.set(enough_nuc,
                 min(tmpa * qh2so4_cur, freduce * qmolso4a_del_max));
  qnh3_del.set(enough_nuc,
               min(tmpa * qnh3_cur, qh2so4_del * molenh4a_per_moleso4a));
  qh2so4_del.set(enough_nuc, -qh2so4_del);
  qnh3_del.set(enough_nuc, -qnh3_del);

  // changes to so4 & nh4 aerosol (in mol/mol-air)
  qso4a_del.set(enough_nuc, -qh2so4_del);
  qnh4a_del.set(enough_nuc, -qnh3_del);

  // change to aerosol number (in #/mol-air)
  qnuma_del.set(
      enough_nuc,
      1.0e-3 * (qso4a_del * mw_so4a + qnh4a_del * mw_nh4a) / mass_part);
}

}  // namespace nucleation

/// @class Nucleation
/// This class implements MAM4's nucleation parameterization.
class Nucleation {
 public:
  // nucleation-specific configuration
  struct Config {
    // "host" parameters
    Real dens_so4a_host, mw_nh4a_host, mw_so4a_host;

    // Nucleation parameters
    int newnuc_method_user_choice;
    int pbl_nuc_wang2008_user_choice;
    Real adjust_factor_bin_tern_ratenucl;
    Real adjust_factor_pbl_ratenucl;
    Real accom_coef_h2so4;
    Real newnuc_adjust_factor_dnaitdt;

    // default constructor -- sets default values for parameters
    Config()
        : dens_so4a_host(0),
          mw_nh4a_host(mw_nh4a),
          mw_so4a_host(mw_so4a),
          newnuc_method_user_choice(2),
          pbl_nuc_wang2008_user_choice(1),
          adjust_factor_bin_tern_ratenucl(1.0),
          adjust_factor_pbl_ratenucl(1.0),
          accom_coef_h2so4(1.0),
          newnuc_adjust_factor_dnaitdt(1.0) {}

    Config(const Config&) = default;
    ~Config() = default;
    Config& operator=(const Config&) = default;
  };

 private:
  static const int nait = static_cast<int>(ModeIndex::Aitken);
  static const int igas_h2so4 = static_cast<int>(GasId::H2SO4);
  static const int igas_nh3 = static_cast<int>(GasId::NH3);

  static constexpr Real mw_h2so4 = Constants::molec_weight_h2so4;
  static constexpr Real mw_so4a = Constants::molec_weight_so4;
  static constexpr Real mw_nh4a = Constants::molec_weight_nh4;
  static constexpr Real pi = Constants::pi;

  // Nucleation-specific configuration
  Config config_;

  // Mode parameters
  Real dgnum_aer[4],   // mean geometric number diameter
      dgnumhi_aer[4],  // max geometric number diameter
      dgnumlo_aer[4];  // min geometric number diameter

 public:
  // name -- unique name of the process implemented by this class
  const char* name() const { return "MAM4 nucleation"; }

  // init -- initializes the implementation with MAM4's configuration and with
  // a process-specific configuration.
  void init(const AeroConfig& aero_config,
            const Config& nucl_config = Config()) {
    // Set nucleation-specific config parameters.
    config_ = nucl_config;

    // Set mode parameters.
    for (int m = 0; m < 4; ++m) {
      // FIXME: There is no mean geometric number diameter in a mode.
      // FIXME: Assume "nominal" diameter for now?
      // FIXME: There is a comment in modal_aero_newnuc.F90 that Dick Easter
      // FIXME: thinks that dgnum_aer isn't used in MAM4, but it is actually
      // FIXME: used in this nucleation parameterization. So we will have to
      // FIXME: figure this out.
      dgnum_aer[m] = modes[m].nom_diameter;
      dgnumlo_aer[m] = modes[m].min_diameter;
      dgnumhi_aer[m] = modes[m].max_diameter;
    }
  }

  // validate -- validates the given atmospheric state and prognostics against
  // assumptions made by this implementation, returning true if the states are
  // valid, false if not
  KOKKOS_INLINE_FUNCTION
  bool validate(const AeroConfig& config, const ThreadTeam& team,
                const Atmosphere& atm, const Prognostics& progs) const {
    return atm.quantities_nonnegative(team) &&
           progs.quantities_nonnegative(team);
  }

  // compute_tendencies -- computes tendencies and updates diagnostics
  // NOTE: that both diags and tends are const below--this means their views
  // NOTE: are fixed, but the data in those views is allowed to vary.
  KOKKOS_INLINE_FUNCTION
  void compute_tendencies(const AeroConfig& config, const ThreadTeam& team,
                          Real t, Real dt, const Atmosphere& atm,
                          const Prognostics& progs, const Diagnostics& diags,
                          const Tendencies& tends) const {
    int iaer_so4 = aerosol_index_for_mode(ModeIndex::Aitken, AeroId::SO4);
    constexpr Real r_universal = Constants::r_gas;
    const int nk = PackInfo::num_packs(atm.num_levels());
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, nk), KOKKOS_CLASS_LAMBDA(int k) {
          // extract atmospheric state
          Pack temp = atm.temperature(k);
          Pack pmid = atm.pressure(k);
          Pack aircon = pmid / (r_universal * temp);
          Pack zmid = atm.height(k);
          Real pblh = atm.planetary_boundary_height;
          Pack qv = atm.vapor_mixing_ratio(k);
          Pack relhum = conversions::relative_humidity_from_vapor_mixing_ratio(
              qv, pmid, temp);
          Pack uptkrate_so4 = 0;
          Pack del_h2so4_gasprod = 0;
          Pack del_h2so4_aeruptk = 0;

          // extract gas mixing ratios
          Pack qgas_cur[13], qgas_avg[13];
          for (int g = 0; g < 13; ++g) {
            qgas_cur[g] = progs.q_gas[g](k);
            qgas_avg[g] = progs.q_gas[g](k);  // FIXME: what should we do here??
          }

          // extract aerosol mixing ratios
          Pack qnum_cur[4], qaer_cur[4][7];
          for (int m = 0; m < 4; ++m) {  // modes
            qnum_cur[m] = progs.n_mode[m](k);
            for (int a = 0; a < 7; ++a) {  // aerosols
              qaer_cur[m][a] = progs.q_aero_i[m][a](k);
            }
          }

          Pack qwtr_cur[4] = {0, 0, 0, 0};  // FIXME: what is this?

          // compute tendencies at this level
          Pack dndt_ait, dmdt_ait, dso4dt_ait, dnh4dt_ait, dnclusterdt;
          compute_tendencies_(dt, temp, pmid, aircon, zmid, pblh, relhum,
                              uptkrate_so4, del_h2so4_gasprod,
                              del_h2so4_aeruptk, qgas_cur, qgas_avg, qnum_cur,
                              qaer_cur, qwtr_cur, dndt_ait, dmdt_ait,
                              dso4dt_ait, dnh4dt_ait, dnclusterdt);

          // Store the computed tendencies.
          tends.n_mode[nait](k) = dndt_ait;
          tends.q_aero_i[nait][iaer_so4](k) = dso4dt_ait;
          tends.q_gas[igas_h2so4](k) = -dso4dt_ait;
          // FIXME: what about dmdt_ait?
        });
  }

 private:
  // This function computes relevant tendencies at a single vertical level. It
  // was ported directly from the compute_tendencies subroutine in the
  // modal_aero_newnuc module from the MAM4 box model.
  KOKKOS_INLINE_FUNCTION
  void compute_tendencies_(Real deltat, const Pack& temp, const Pack& pmid,
                           const Pack& aircon, const Pack& zmid, Real pblh,
                           const Pack& relhum, const Pack& uptkrate_h2so4,
                           const Pack& del_h2so4_gasprod,
                           const Pack& del_h2so4_aeruptk,
                           const Pack qgas_cur[13], const Pack qgas_avg[13],
                           const Pack qnum_cur[4], const Pack qaer_cur[4][7],
                           const Pack qwtr_cur[4], Pack& dndt_ait,
                           Pack& dmdt_ait, Pack& dso4dt_ait, Pack& dnh4dt_ait,
                           Pack& dnclusterdt) const {
    static constexpr Real avogadro = Constants::avogadro;
    static constexpr Real rgas = Constants::r_gas;
    static constexpr Real ln_nuc_rate_cutoff = -13.82;

    // min h2so4 vapor for nuc calcs = 4.0e-16 mol/mol-air ~= 1.0e4
    // molecules/cm3
    static constexpr Real qh2so4_cutoff = 4.0e-16;

    IntPack newnuc_method_actual, pbl_nuc_wang2008_actual;

    constexpr int nsize = 1;
    Pack dplom_mode[nsize], dphim_mode[nsize];
    IntPack isize_group;

    Pack cair;  // air density
    Pack so4vol, nh3ppt;

    Pack radius_cluster;  // radius of newly formed cluster, in nm
    Pack rateloge;        // ln(J)
    Pack cnum_h2so4;
    Pack cnum_nh3;

    Pack mass1p;
    Pack mass1p_aithi, mass1p_aitlo;
    Pack qh2so4_cur, qh2so4_avg, qh2so4_del;
    Pack qnh3_cur, qnh3_del, qnh4a_del;
    Pack qnuma_del;
    Pack qso4a_del;
    Pack relhumnn;
    Pack tmpa, tmpb, tmpc;
    Pack tmp_q2, tmp_q3;
    Pack tmp_frso4, tmp_uptkrate;
    Pack dens_nh4so4a;

    // process-specific configuration data
    int newnuc_method_user_choice = config_.newnuc_method_user_choice;
    int pbl_nuc_wang2008_user_choice = config_.pbl_nuc_wang2008_user_choice;
    Real adjust_factor_bin_tern_ratenucl =
        config_.adjust_factor_bin_tern_ratenucl;
    Real adjust_factor_pbl_ratenucl = config_.adjust_factor_pbl_ratenucl;
    Real accom_coef_h2so4 = config_.accom_coef_h2so4;
    Real newnuc_adjust_factor_dnaitdt = config_.newnuc_adjust_factor_dnaitdt;
    Real dens_so4a_host = config_.dens_so4a_host;
    Real mw_so4a_host = config_.mw_so4a_host;
    Real mw_nh4a_host = config_.mw_nh4a_host;

    dndt_ait = 0;
    dmdt_ait = 0;
    dnh4dt_ait = 0;
    dso4dt_ait = 0;
    dnclusterdt = 0;

    //==================================
    // limit RH to between 0.1% and 99%
    //==================================
    relhumnn = max(0.01, min(0.99, relhum));

    //=========================================================================
    // prepare h2so4 mixing ratio and condensation rate that will be passed to
    // the nucleation parameterization
    //=========================================================================
    qh2so4_cur = qgas_cur[igas_h2so4];

    // E3SM: use qh2so4_avg and first-order loss rate calculated in
    // mam_gasaerexch_1subarea
    qh2so4_avg = qgas_avg[igas_h2so4];
    tmp_uptkrate = uptkrate_h2so4;

    if (igas_nh3 > 0) {
      qnh3_cur = max(0.0, qgas_cur[igas_nh3]);
    } else {
      qnh3_cur = 0.0;
    }

    // unit conversion for gas concentrations:
    // calculate h2so4 in molecules/cm3 and nh3 in ppt
    cair = pmid / (temp * rgas);
    so4vol = qh2so4_avg * cair * avogadro * 1.0e-6;
    nh3ppt = qnh3_cur * 1.0e12;

    //=======================================================================
    // call routine to get nucleation rate in terms of new cluster formation
    // rate (#/m3/s)
    //=======================================================================
    if (newnuc_method_user_choice != 0) {
      // Hui Wan's note from code refactoring in July 2021:
      // Subroutine mer07_veh02_wang08_nuc_1box provides
      //  - dnclusterdt (unit: #/m3/s): new cluster formation rate
      //  - rateloge (unit: ln(#/cm3/s)): logarithm of new cluster formation
      //  rate
      //  - cnum_h2so4, cnum_nh3: number of of h2so4 or nh3 molecules per
      //  cluster
      //  - radius_cluster (unit: nm): radius of new cluster
      // Output variables rateloge, cnum_h2so4, cnum_nh3, and radius_cluster
      // are used below in the calculation of cluster "growth". I chose to keep
      // these variable names the same as in the old subroutine
      // mer07_veh02_nuc_mosaic_1box to facilitate comparison.

      nucleation::mer07_veh02_wang08_nuc_1box(
          newnuc_method_user_choice, newnuc_method_actual,        // in, out
          pbl_nuc_wang2008_user_choice, pbl_nuc_wang2008_actual,  // in, out
          ln_nuc_rate_cutoff,                                     // in
          adjust_factor_bin_tern_ratenucl, adjust_factor_pbl_ratenucl,   // in
          pi, so4vol, nh3ppt, temp, relhumnn, zmid, pblh,                // in
          dnclusterdt, rateloge, cnum_h2so4, cnum_nh3, radius_cluster);  // out

    } else {
      rateloge = ln_nuc_rate_cutoff;
      dnclusterdt = 0.;
      newnuc_method_actual = 0;
    }

    //======================================================================
    // "Grow" the newly formed clusters to size in the smallest bin/mode of
    // the host model
    //======================================================================
    qnuma_del = 0.0;
    qso4a_del = 0.0;
    qnh4a_del = 0.0;
    qh2so4_del = 0.0;
    qnh3_del = 0.0;

    // dry-diameter limits for "grown" new particles
    dplom_mode[0] =
        exp(0.67 * log(dgnumlo_aer[nait]) + 0.33 * log(dgnum_aer[nait]));
    dphim_mode[0] = dgnumhi_aer[nait];

    //----------------------------------------------------------------
    // Only do the cluster growth calculation when nucleation rate is
    // appreciable
    //----------------------------------------------------------------
    auto sufficient_nuc = (rateloge > ln_nuc_rate_cutoff);

    // mass1p_... = mass (kg) of so4 & nh4 in a single particle of diameter ...
    // (assuming same dry density for so4 & nh4)
    // mass1p_aitlo - dp = dplom_mode(1);
    // mass1p_aithi - dp = dphim_mode(1);

    tmpa.set(sufficient_nuc, dens_so4a_host * pi / 6.0);
    mass1p_aitlo.set(sufficient_nuc, tmpa * cube(dplom_mode[0]));
    mass1p_aithi.set(sufficient_nuc, tmpa * cube(dphim_mode[0]));

    // Cluster growth
    IntPack isize_group_bb;
    Pack dens_nh4so4a_bb, qh2so4_del_bb, qnh3_del_bb, qso4a_del_bb,
        qnh4a_del_bb, qnuma_del_bb;
    nucleation::newnuc_cluster_growth(
        dnclusterdt, cnum_h2so4, cnum_nh3, radius_cluster, dplom_mode,
        dphim_mode, nsize, deltat, temp, relhumnn, cair, accom_coef_h2so4,
        mw_so4a, mw_so4a_host, mw_nh4a, avogadro, pi, qnh3_cur, qh2so4_cur,
        so4vol, tmp_uptkrate, isize_group_bb, dens_nh4so4a_bb, qh2so4_del_bb,
        qnh3_del_bb, qso4a_del_bb, qnh4a_del_bb, qnuma_del_bb);
    isize_group.set(sufficient_nuc, isize_group_bb);
    dens_nh4so4a.set(sufficient_nuc, dens_nh4so4a_bb);
    qh2so4_del.set(sufficient_nuc, qh2so4_del_bb);
    qnh3_del.set(sufficient_nuc, qnh3_del_bb);
    qso4a_del.set(sufficient_nuc, qso4a_del_bb);
    qnh4a_del.set(sufficient_nuc, qnh4a_del_bb);
    qnuma_del.set(sufficient_nuc, qnuma_del_bb);

    //=====================================
    // Deriving mass mixing ratio tendency
    //=====================================

    // convert qnuma_del from (#/mol-air) to (#/kmol-air)
    qnuma_del *= 1.0e3;

    // number nuc rate (#/kmol-air/s) from number nuc amt
    dndt_ait = qnuma_del / deltat;

    // fraction of mass nuc going to so4
    tmpa = qso4a_del * mw_so4a_host;
    if (igas_nh3 > 0) {
      tmpb = tmpa + qnh4a_del * mw_nh4a_host;
      tmp_frso4 = max(tmpa, 1.0e-35) / max(tmpb, 1.0e-35);
    } else {
      tmpb = tmpa;
      tmp_frso4 = 1.0;
    }

    // mass nuc rate (kg/kmol-air/s) from mass nuc amts
    dmdt_ait = max(0.0, (tmpb / deltat));

    //=====================================================
    // Various adjustments to keep the solution reasonable
    //=====================================================

    // ignore newnuc if number rate < 100 #/kmol-air/s ~= 0.3 #/mg-air/d
    auto dndt_small = (dndt_ait < 1.0e2);
    dndt_ait.set(dndt_small, 0.0);
    dmdt_ait.set(dndt_small, 0.0);

    // mirage2 code checked for complete h2so4 depletion here,
    // but this is now done in mer07_veh02_nuc_mosaic_1box
    mass1p.set(!dndt_small, dmdt_ait / dndt_ait);

    // apply particle size constraints
    auto too_small = (mass1p < mass1p_aitlo);
    // reduce dndt to increase new particle size
    dndt_ait.set(too_small, dmdt_ait / mass1p_aitlo);
    auto too_big = (mass1p > mass1p_aithi);
    // reduce dmdt to decrease new particle size
    dmdt_ait.set(too_big, dndt_ait * mass1p_aithi);

    // If qh2so4_avg is very low. assume no nucleation will happen
    auto enough_h2so4 = (qh2so4_avg > qh2so4_cutoff);

    // *** apply adjustment factor to avoid unrealistically high
    // aitken number concentrations in mid and upper troposphere
    dndt_ait.set(enough_h2so4, dndt_ait * newnuc_adjust_factor_dnaitdt);
    dmdt_ait.set(enough_h2so4, dmdt_ait * newnuc_adjust_factor_dnaitdt);

    //=================================
    // Diagnose so4 and nh4 tendencies
    //=================================
    // dso4dt_ait, dnh4dt_ait are (kmol/kmol-air/s)
    dso4dt_ait = dmdt_ait * tmp_frso4 / mw_so4a_host;
    dnh4dt_ait = dmdt_ait * (1.0 - tmp_frso4) / mw_nh4a_host;
  }
};

}  // namespace mam4

#endif
