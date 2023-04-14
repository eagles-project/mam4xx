// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

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

using haero::cube;
using haero::max;
using haero::min;

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
// Atmos. Chem. Phys.  9, 239-260, 2009
//--------------------------------------------------------
KOKKOS_INLINE_FUNCTION
void pbl_nuc_wang2008(Real so4vol, Real pi, int pbl_nuc_wang2008_user_choice,
                      Real adjust_factor_pbl_ratenucl,
                      int &pbl_nuc_wang2008_actual, Real &ratenucl,
                      Real &rateloge, Real &cnum_tot, Real &cnum_h2so4,
                      Real &cnum_nh3, Real &radius_cluster_nm) {
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

  constexpr Real mw_h2so4_gmol = 98.0;       // BAD_CONSTANT
  constexpr Real avogadro_mol = 6.023e23;    // BAD_CONSTANT
  constexpr Real density_sulfate_gcm3 = 1.8; // BAD_CONSTANT

  //-----------------------------------------------------------------
  // Initialize the pbl_nuc_wang2008_actual flag. Assumed default is
  // no PBL nucleation.
  //-----------------------------------------------------------------
  pbl_nuc_wang2008_actual = 0;

  //-------------------------------------------------------------
  // Calculate nucleation rate using incoming so4 concentration.
  //-------------------------------------------------------------
  Real tmp_ratenucl;
  if (pbl_nuc_wang2008_user_choice == 1) {
    tmp_ratenucl = wang2008::first_order_pbl_nucleation_rate(so4vol);
  } else if (pbl_nuc_wang2008_user_choice == 2) {
    tmp_ratenucl = wang2008::second_order_pbl_nucleation_rate(so4vol);
  } else {
    return;
  }

  // Scale the calculated PBL nuc rate by user-specificed tuning factor
  tmp_ratenucl = tmp_ratenucl * adjust_factor_pbl_ratenucl;
  Real tmp_rateloge = log(max(1.0e-38, tmp_ratenucl));

  //------------------------------------------------------------------
  // If PBL nuc rate is lower than the incoming ternary/binary rate,
  // discard the PBL nuc rate (i.e, do not touch any incoming value).
  //------------------------------------------------------------------
  if (tmp_rateloge <= rateloge)
    return;

  //------------------------------------------------------------------
  // Otherwise, use the PBL nuc rate.
  //------------------------------------------------------------------
  pbl_nuc_wang2008_actual = pbl_nuc_wang2008_user_choice;
  rateloge = tmp_rateloge;
  ratenucl = tmp_ratenucl;

  // following wang 2002, assume fresh nuclei are 1 nm diameter
  // subsequent code will "grow" them to aitken mode size
  radius_cluster_nm = 0.5;

  // assume fresh nuclei are pure h2so4
  //    since aitken size >> initial size, the initial composition
  //    has very little impact on the results

  Real tmp_diam = radius_cluster_nm * 2.0e-7;      // diameter in cm
  Real tmp_volu = cube(tmp_diam) * (pi / 6.0);     // volume in cm^3
  Real tmp_mass = tmp_volu * density_sulfate_gcm3; // mass in g

  // no. of h2so4 molec per cluster assuming pure h2so4
  cnum_h2so4 = (tmp_mass / mw_h2so4_gmol) * avogadro_mol;
  cnum_nh3 = 0.0;
  cnum_tot = cnum_h2so4;
}

//-----------------------------------------------------------------
// calculates binary nucleation rate and critical cluster size
// using the parameterization in
//     vehkam\"aki, h., m. kulmala, i. napari, k.e.j. lehtinen,
//        c. timmreck, m. noppel and a. laaksonen, 2002,
//        an improved parameterization for sulfuric acid-water nucleation
//        rates for tropospheric and stratospheric conditions,
//        j. geophys. res., 107, 4622, doi:10.1029/2002jd002184
//-----------------------------------------------------------------
KOKKOS_INLINE_FUNCTION
void binary_nuc_vehk2002(Real temp, Real rh, Real so4vol, Real &ratenucl,
                         Real &rateloge, Real &cnum_h2so4, Real &cnum_tot,
                         Real &radius_cluster) {
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
  // following eq. (11) in Vehkam\"aki et al. (2002)
  Real x_crit = vehkamaki2002::h2so4_critical_mole_fraction(so4vol, temp, rh);

  // calc nucleation rate
  // following eq. (12) in Vehkam\"aki et al. (2002)
  rateloge = log(vehkamaki2002::nucleation_rate(so4vol, temp, rh, x_crit));
  ratenucl = exp(min(rateloge, log(1e38)));

  // calc number of molecules in critical cluster
  // following eq. (13) in Vehkam\"aki et al. (2002)
  cnum_tot = vehkamaki2002::num_critical_molecules(so4vol, temp, rh, x_crit);

  cnum_h2so4 = cnum_tot * x_crit;

  // calc radius (nm) of critical cluster
  // following eq. (14) in Vehkam\"aki et al. (2002)
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
void ternary_nuc_merik2007(Real t, Real rh, Real c2, Real c3, Real &j_log,
                           Real &ntot, Real &nacid, Real &namm, Real &r) {
  Real t_onset = merikanto2007::onset_temperature(rh, c2, c3);

  // Set log(J) assuming no nucleation.

  // If t_onset > t, nucleation occurs.
  if (t_onset > t) {
    j_log = merikanto2007::log_nucleation_rate(t, rh, c2, c3);
    ntot = merikanto2007::num_critical_molecules(j_log, t, c2, c3);
    r = merikanto2007::critical_radius(j_log, t, c2, c3);
    nacid = merikanto2007::num_h2so4_molecules(j_log, t, c2, c3);
    namm = merikanto2007::num_nh3_molecules(j_log, t, c2, c3);
  } else {
    // nucleation rate less that 5e-6, setting j_log arbitrarily small
    j_log = -300.;
  }
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
// * vehkam\"aki, h., m. kulmala, i. napari, k.e.j. lehtinen,
//   c. timmreck, m. noppel and a. laaksonen, 2002,
//   an improved parameterization for sulfuric acid-water nucleation
//   rates for tropospheric and stratospheric conditions,
//   j. geophys. res., 107, 4622, doi:10.1029/2002jd002184
//
// * Wang, M. and J.E. Penner, 2008,
//   Aerosol indirect forcing in a global model with particle nucleation,
//   Atmos. Chem. Phys. Discuss., 8, 13943-13998
//   Atmos. Chem. Phys.  9, 239-260, 2009
KOKKOS_INLINE_FUNCTION
void mer07_veh02_wang08_nuc_1box(int newnuc_method_user_choice,
                                 int &newnuc_method_actual,        // in, out
                                 int pbl_nuc_wang2008_user_choice, // in
                                 int &pbl_nuc_wang2008_actual,     // in, out
                                 Real ln_nuc_rate_cutoff,          // in
                                 Real adjust_factor_bin_tern_ratenucl,    // in
                                 Real adjust_factor_pbl_ratenucl,         // in
                                 Real pi, Real so4vol_in, Real nh3ppt_in, // in
                                 Real temp_in, Real rh_in, Real zm_in,
                                 Real pblh_in, // in
                                 Real &dnclusterdt, Real &rateloge,
                                 Real &cnum_h2so4,                       // out
                                 Real &cnum_nh3, Real &radius_cluster) { // out

  Real rh_bb;     // bounded value of rh_in
  Real so4vol_bb; // bounded value of so4vol_in (molecules per cm3)
  Real temp_bb;   // bounded value of temp_in (K)
  Real nh3ppt_bb; // bounded nh3 (ppt)

  Real cnum_tot;  // total number of molecules in a cluster
  Real ratenuclt; // J: nucleation rate from parameterization.
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
  if ((newnuc_method_user_choice == 3) && (nh3ppt_in >= 0.1)) {
    if (so4vol_in >= 5.0e4) {
      temp_bb = max(235.0, min(295.0, temp_in));
      rh_bb = max(0.05, min(0.95, rh_in));
      so4vol_bb = max(5.0e4, min(1.0e9, so4vol_in));
      nh3ppt_bb = max(0.1, min(1.0e3, nh3ppt_in));
      ternary_nuc_merik2007(temp_bb, rh_bb, so4vol_bb, nh3ppt_bb, rateloge,
                            cnum_tot, cnum_h2so4, cnum_nh3, radius_cluster);
    }
    newnuc_method_actual = 3;
  } else {
    //---------------------------------------------------------------------
    // Otherwise, make call to vehkamaki binary parameterization routine
    //---------------------------------------------------------------------
    if (so4vol_in >= 1.0e4) {
      temp_bb = max(230.15, min(305.15, temp_in));
      rh_bb = max(1.0e-4, min(1.0, rh_in));
      so4vol_bb = max(1.0e4, min(1.0e11, so4vol_in));
      binary_nuc_vehk2002(temp_bb, rh_bb, so4vol_bb, ratenuclt, rateloge,
                          cnum_h2so4, cnum_tot, radius_cluster);
    }
    cnum_nh3 = 0.0;
    newnuc_method_actual = 2;
  }

  rateloge += log(max(1.0e-38, adjust_factor_pbl_ratenucl));

  //---------------------------------------------------------------------
  // Do boundary layer nuc
  //---------------------------------------------------------------------
  pbl_nuc_wang2008_actual = 0;
  if ((pbl_nuc_wang2008_user_choice != 0) && (zm_in <= max(pblh_in, 100.0))) {
    so4vol_bb = so4vol_in;
    pbl_nuc_wang2008(so4vol_bb, pi, pbl_nuc_wang2008_user_choice,
                     adjust_factor_pbl_ratenucl, pbl_nuc_wang2008_actual,
                     ratenuclt, rateloge, cnum_tot, cnum_h2so4, cnum_nh3,
                     radius_cluster);
  } else {
    pbl_nuc_wang2008_actual = 0;
  }

  //---------------------------------------------------------------------
  // if nucleation rate is less than 1e-6 #/cm3/s ~= 0.1 #/cm3/day,
  // exit with new particle formation = 0. Otherwise, calculate the
  // nucleation rate in #/m3/s
  //---------------------------------------------------------------------
  if (rateloge <= ln_nuc_rate_cutoff) {
    dnclusterdt = 0.0;
  } else {
    // ratenuclt is #/cm3/s; dnclusterdt is #/m3/s
    dnclusterdt = exp(rateloge) * 1.0e6;
  }
}

KOKKOS_INLINE_FUNCTION
void newnuc_cluster_growth(Real ratenuclt_bb, Real cnum_h2so4, Real cnum_nh3,
                           Real radius_cluster, const Real *dplom_sect,
                           const Real *dphim_sect, int nsize, Real dtnuc,
                           Real temp_in, Real rh_in, Real cair,
                           Real accom_coef_h2so4, Real mw_so4a,
                           Real mw_so4a_host, Real mw_nh4a, Real avogad,
                           Real pi, Real qnh3_cur, Real qh2so4_cur,
                           Real so4vol_in, Real h2so4_uptkrate, int &isize_nuc,
                           Real &dens_nh4so4a, Real &qh2so4_del, Real &qnh3_del,
                           Real &qso4a_del, Real &qnh4a_del, Real &qnuma_del) {
  Real tmpa, tmpb, tmpe;
  Real voldry_clus;      // critical-cluster dry volume [m3]
  Real voldry_part;      // "grown" single-particle dry volume [m3]
  Real wetvol_dryvol;    // grown particle [(wet-volume)/(dry-volume)]
  Real wet_volfrac_so4a; // grown particle [(dry-volume-from-so4)/(wet-volume)]
  Real dpdry_part;       // "grown" single-particle dry diameter [m]
  Real dpdry_clus;       // critical cluster diameter [m]

  Real cs_prime_kk;      // kk2002 "cs_prime" parameter [1/m2]
  Real dfin_kk, dnuc_kk; // kk2002 final/initial new particle wet diameter [nm]
  Real tmp_spd;          // h2so4 vapor molecular speed [m/s]
  Real qmolnh4a_del_max; // max production of aerosol nh4 over dtnuc
                         // [mol/mol-air]
  Real qmolso4a_del_max; // max production of aerosol so4 over dtnuc
                         // [mol/mol-air]
  Real ratenuclt_kk;     // nucleation rate after kk2002 adjustment [#/m3/s]

  int igrow;

  Real tmp_n1, tmp_n2, tmp_n3;
  Real tmp_m1, tmp_m2, tmp_m3;
  Real dens_part;             // "grown" single-particle dry density [kg/m3]
  Real mass_part;             // "grown" single-particle dry mass [kg]
  Real molenh4a_per_moleso4a; // [mol aerosol nh4]/[mol aerosol so4]
  Real kgaero_per_moleso4a;   // [kg dry aerosol/mol aerosol so4]
  Real factor_kk;
  Real freduce; // reduction factor applied to nucleation rate
                // due to limited availability of h2so4 & nh3 gases
  Real freducea, freduceb;
  Real gamma_kk; // kk2002 "gamma" parameter [nm2*m2/h]
  Real gr_kk;    // kk2002 "gr" parameter [nm/h]
  Real nu_kk;    // kk2002 "nu" parameter [nm]

  constexpr Real onethird = 1.0 / 3.0;

  // dry densities [kg/m3] molecular weights of aerosol
  // ammsulf, ammbisulf, and sulfacid (from mosaic  dens_electrolyte values)
  //  Real dens_ammsulf   = 1.769e3
  //  Real dens_ammbisulf = 1.78e3
  //  Real dens_sulfacid  = 1.841e3
  // use following to match cam3 modal_aero densities
  constexpr Real dens_ammsulf = 1.770e3;   // BAD_CONSTANT
  constexpr Real dens_ammbisulf = 1.770e3; // BAD_CONSTANT
  constexpr Real dens_sulfacid = 1.770e3;  // BAD_CONSTANT

  // molecular weights [g/mol] of aerosol ammsulf, ammbisulf, and sulfacid
  // for ammbisulf and sulfacid, use 114 & 96 here rather than 115 & 98
  // because we don't keep track of aerosol hion mass
  constexpr Real mw_ammsulf = 132.0;   // BAD_CONSTANT
  constexpr Real mw_ammbisulf = 114.0; // BAD_CONSTANT
  constexpr Real mw_sulfacid = 96.0;   // BAD_CONSTANT

  // wet/dry volume ratio - use simple kohler approx for ammsulf/ammbisulf
  tmpa = max(0.10, min(0.95, rh_in));
  wetvol_dryvol = 1.0 - 0.56 / log(tmpa);

  // determine size bin into which the new particles go
  // (probably it will always be bin #1, but ...)
  voldry_clus = (max(cnum_h2so4, 1.0) * mw_so4a + cnum_nh3 * mw_nh4a) /
                (1.0e3 * dens_sulfacid * avogad);

  // correction when host code sulfate is really ammonium bisulfate/sulfate
  voldry_clus *= (mw_so4a_host / mw_so4a);
  dpdry_clus = pow(voldry_clus * 6.0 / pi, onethird);

  isize_nuc = 1;
  dpdry_part = dplom_sect[0];
  if (dpdry_clus <= dplom_sect[0]) {
    igrow = 1; // need to grow clusters to larger size
  } else if (dpdry_clus >= dphim_sect[nsize - 1]) {
    igrow = 0;
    isize_nuc = nsize;
    dpdry_part = dphim_sect[nsize - 1];
  } else {
    igrow = 0;
    for (int i = 0; i < nsize; ++i) {
      if (dpdry_clus < dphim_sect[i]) {
        isize_nuc = i;
        dpdry_part = dpdry_clus;
        dpdry_part = min(dpdry_part, dphim_sect[i]);
        dpdry_part = max(dpdry_part, dplom_sect[i]);
        break;
      }
    }
  }
  voldry_part = (pi / 6.0) * cube(dpdry_part);

  // determine composition and density of the "grown particles"
  // the grown particles are assumed to be liquid
  //    (since critical clusters contain water)
  //    so any (nh4/so4) molar ratio between 0 and 2 is allowed
  // assume that the grown particles will have
  //    (nh4/so4 molar ratio) = min( 2, (nh3/h2so4 gas molar ratio) )
  if (igrow <= 0) {
    // no "growing" so pure sulfuric acid
    tmp_n1 = 0.0;
    tmp_n2 = 0.0;
    tmp_n3 = 1.0;
  } else if (qnh3_cur >= qh2so4_cur) {
    // combination of ammonium sulfate and ammonium bisulfate
    // tmp_n1 & tmp_n2 = mole fractions of the ammsulf & ammbisulf
    tmp_n1 = (qnh3_cur / qh2so4_cur) - 1.0;
    tmp_n1 = max(0.0, min(1.0, tmp_n1));
    tmp_n2 = 1.0 - tmp_n1;
    tmp_n3 = 0.0;
  } else {
    // combination of ammonium bisulfate and sulfuric acid
    // tmp_n2 & tmp_n3 = mole fractions of the ammbisulf & sulfacid
    tmp_n1 = 0.0;
    tmp_n2 = (qnh3_cur / qh2so4_cur);
    tmp_n2 = max(0.0, min(1.0, tmp_n2));
    tmp_n3 = 1.0 - tmp_n2;
  }

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
  kgaero_per_moleso4a = (tmp_m1 + tmp_m2 + tmp_m3);

  // correction when host code sulfate is really ammonium bisulfate/sulfate
  kgaero_per_moleso4a = kgaero_per_moleso4a * (mw_so4a_host / mw_so4a);

  // fraction of wet volume due to so4a
  tmpb = 1.0 + molenh4a_per_moleso4a * 17.0 / 98.0;
  wet_volfrac_so4a = 1.0 / (wetvol_dryvol * tmpb);

  // calc kerminen & kulmala (2002) correction
  if (igrow <= 0) {
    factor_kk = 1.0;
  } else {
    // "gr" parameter (nm/h) = condensation growth rate of new particles
    // use kk2002 eqn 21 for h2so4 uptake, and correct for nh3 & h2o uptake
    tmp_spd = 14.7 * sqrt(temp_in); // h2so4 molecular speed [m/s]
    gr_kk = 3.0e-9 * tmp_spd * mw_sulfacid * so4vol_in /
            (dens_part * wet_volfrac_so4a);

    // "gamma" parameter (nm2/m2/h)
    // use kk2002 eqn 22
    // dfin_kk = wet diam (nm) of grown particle having dry dia = dpdry_part (m)
    dfin_kk = 1.0e9 * dpdry_part * pow(wetvol_dryvol, onethird);

    // dnuc_kk = wet diam (nm) of cluster
    dnuc_kk = 2.0 * radius_cluster;
    dnuc_kk = max(dnuc_kk, 1.0);

    // neglect (dmean/150)**0.048 factor,
    // which should be very close to 1.0 because of small exponent
    gamma_kk = 0.23 * pow(dnuc_kk, 0.2) * pow(dfin_kk / 3.0, 0.075) *
               pow(dens_part * 1.0e-3, -0.33) * pow(temp_in / 293.0, -0.75);

    // "cs_prime parameter" (1/m2)
    // instead of kk2002 eqn 3, use
    //     cs_prime ~= tmpa / (4*pi*tmpb * h2so4_accom_coef)
    // where
    //     tmpa = -d(ln(h2so4))/dt by conden to particles   (1/h units)
    //     tmpb = h2so4 vapor diffusivity (m2/h units)
    // this approx is generally within a few percent of the cs_prime
    //     calculated directly from eqn 2,
    //     which is acceptable, given overall uncertainties
    // tmpa = -d(ln(h2so4))/dt by conden to particles   (1/h units)
    tmpa = h2so4_uptkrate * 3600.0;
    tmpa = max(tmpa, 0.0);

    // tmpb = h2so4 gas diffusivity ([m2/s], then [m2/h])
    tmpb = 6.7037e-9 * pow(temp_in, 0.75) / cair;
    tmpb *= 3600.0; // [m2/h] 3600 = seconds in hour
    cs_prime_kk = tmpa / (4.0 * pi * tmpb * accom_coef_h2so4);

    // "nu" parameter (nm) -- kk2002 eqn 11
    nu_kk = gamma_kk * cs_prime_kk / gr_kk;

    // nucleation rate adjustment factor (--) -- kk2002 eqn 13
    factor_kk = exp((nu_kk / dfin_kk) - (nu_kk / dnuc_kk));
  }
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
  if (qmolso4a_del_max > qh2so4_cur) {
    freducea = qh2so4_cur / qmolso4a_del_max;
  }

  // check if max production exceeds available nh3 vapor
  freduceb = 1.0;
  if (molenh4a_per_moleso4a >= 1.0e-10) {
    // max production of aerosol nh4 (ppm) based on ratenuclt_kk and mass_part
    qmolnh4a_del_max = qmolso4a_del_max * molenh4a_per_moleso4a;
    if (qmolnh4a_del_max > qnh3_cur) {
      freduceb = qnh3_cur / qmolnh4a_del_max;
    }
  }

  // get the final reduction factor for nucleation rate
  freduce = min(freducea, freduceb);

  //---------------------------------------------------------------------------
  // Derive aerosol and gas mass mass increments and aerosol number increments
  // for output to calling routine
  //---------------------------------------------------------------------------
  // if adjusted nucleation rate is less than 1e-12 #/m3/s ~= 0.1 #/cm3/day,
  // exit with new particle formation = 0
  if (freduce * ratenuclt_kk <= 1.0e-12) {
    qh2so4_del = 0.0;
    qnh3_del = 0.0;
    qso4a_del = 0.0;
    qnh4a_del = 0.0;
    qnuma_del = 0.0;
  } else {
    //---------------------------------------------------------------------------
    // note:  suppose that at this point, freduce < 1.0 (no gas-available
    //    constraints) and molenh4a_per_moleso4a < 2.0
    // if the gas-available constraints is do to h2so4 availability,
    // then it would be possible to condense "additional" nh3 and have
    // (nh3/h2so4 gas molar ratio) < (nh4/so4 aerosol molar ratio) <= 2
    // one could do some additional calculations of
    // dens_part & molenh4a_per_moleso4a to realize this
    // however, the particle "growing" is a crude approximate way to get
    // the new particles to the host code's minimum particle size,
    // are such refinements worth the effort?
    //---------------------------------------------------------------------------
    // changes to h2so4 & nh3 gas [mol/mol-air], limited by amounts available
    tmpa = 0.9999;
    qh2so4_del = min(tmpa * qh2so4_cur, freduce * qmolso4a_del_max);
    qnh3_del = min(tmpa * qnh3_cur, qh2so4_del * molenh4a_per_moleso4a);
    qh2so4_del = -qh2so4_del;
    qnh3_del = -qnh3_del;

    // changes to so4 & nh4 aerosol [mol/mol-air]
    qso4a_del = -qh2so4_del;
    qnh4a_del = -qnh3_del;

    // change to aerosol number [#/mol-air]
    qnuma_del =
        1.0e-3 * (qso4a_del * mw_so4a + qnh4a_del * mw_nh4a) / mass_part;
  }
}

} // namespace nucleation

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
        : dens_so4a_host(0), mw_nh4a_host(mw_nh4a), mw_so4a_host(mw_so4a),
          newnuc_method_user_choice(2), pbl_nuc_wang2008_user_choice(1),
          adjust_factor_bin_tern_ratenucl(1.0), adjust_factor_pbl_ratenucl(1.0),
          accom_coef_h2so4(1.0), newnuc_adjust_factor_dnaitdt(1.0) {}

    Config(const Config &) = default;
    ~Config() = default;
    Config &operator=(const Config &) = default;
  };

private:
  static constexpr int num_modes = AeroConfig::num_modes();
  static constexpr int num_gases = AeroConfig::num_gas_ids();
  static constexpr int max_num_mode_species = AeroConfig::num_aerosol_ids();
  static const int nait = static_cast<int>(ModeIndex::Aitken);
  static const int igas_h2so4 = static_cast<int>(GasId::H2SO4);
  // Turn off NH3. Any negative number would turn it off, this is what is
  // used in the mam_refactor code.
  static const int igas_nh3 = -999888777; // static_cast<int>(GasId::NH3);

  static constexpr Real mw_so4a = 96.0;              // BAD_CONSTANT
  static constexpr Real mw_nh4a = 18.0;              // BAD_CONSTANT
  static constexpr Real pi = 3.14159265358979323846; // BAD_CONSTANT

  // Nucleation-specific configuration
  Config config_;

  // Mode parameters
  Real dgnum_aer[num_modes],  // mean geometric number diameter
      dgnumhi_aer[num_modes], // max geometric number diameter
      dgnumlo_aer[num_modes]; // min geometric number diameter

public:
  // name -- unique name of the process implemented by this class
  const char *name() const { return "MAM4 nucleation"; }

  // init -- initializes the implementation with MAM4's configuration and with
  // a process-specific configuration.
  void init(const AeroConfig &aero_config,
            const Config &nucl_config = Config()) {
    // Set nucleation-specific config parameters.
    config_ = nucl_config;

    // Set mode parameters.
    for (int m = 0; m < num_modes; ++m) {
      // FIXME: There is no mean geometric number diameter in a mode.
      // FIXME: Assume "nominal" diameter for now?
      // FIXME: There is a comment in modal_aero_newnuc.F90 that Dick Easter
      // FIXME: thinks that dgnum_aer isn't used in MAM4, but it is actually
      // FIXME: used in this nucleation parameterization. So we will have to
      // FIXME: figure this out.
      dgnum_aer[m] = modes(m).nom_diameter;
      dgnumlo_aer[m] = modes(m).min_diameter;
      dgnumhi_aer[m] = modes(m).max_diameter;
    }
  }

  // validate -- validates the given atmospheric state and prognostics against
  // assumptions made by this implementation, returning true if the states are
  // valid, false if not
  KOKKOS_INLINE_FUNCTION
  bool validate(const AeroConfig &config, const ThreadTeam &team,
                const Atmosphere &atm, const Prognostics &progs) const {
    return atm.quantities_nonnegative(team) &&
           progs.quantities_nonnegative(team);
  }

  // compute_tendencies -- computes tendencies and updates diagnostics
  // NOTE: that both diags and tends are const below--this means their views
  // NOTE: are fixed, but the data in those views is allowed to vary.
  KOKKOS_INLINE_FUNCTION
  void compute_tendencies(const AeroConfig &config, const ThreadTeam &team,
                          Real t, Real dt, const Atmosphere &atm,
                          const Prognostics &progs, const Diagnostics &diags,
                          const Tendencies &tends) const {
    int iaer_so4 = aerosol_index_for_mode(ModeIndex::Aitken, AeroId::SO4);
    static constexpr Real boltzmann =
        1.38065e-23; // BAD_CONSTANT (Boltzmann's constant ~ J/K/molecule)
    static constexpr Real avogadro =
        6.02214e26; // BAD_CONSTANT (Avogadro's number ~ molecules/kmole)
    static constexpr Real r_universal = boltzmann * avogadro; // BAD_CONSTANT
    const int nk = atm.num_levels();
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, nk), KOKKOS_CLASS_LAMBDA(int k) {
          // extract atmospheric state
          Real temp = atm.temperature(k);
          Real pmid = atm.pressure(k);
          Real aircon = pmid / (r_universal * temp);
          Real zmid = atm.height(k);
          Real pblh = atm.planetary_boundary_layer_height;
          Real qv = atm.vapor_mixing_ratio(k);
          Real relhum = conversions::relative_humidity_from_vapor_mixing_ratio(
              qv, pmid, temp);
          Real uptkrate_so4 = 0;
          Real del_h2so4_gasprod = 0;
          Real del_h2so4_aeruptk = 0;

          // extract gas mixing ratios
          Real qgas_cur[num_gases], qgas_avg[num_gases];
          for (int g = 0; g < num_gases; ++g) {
            qgas_cur[g] = progs.q_gas[g](k);
            qgas_avg[g] = progs.q_gas[g](k); // FIXME: what should we do here??
          }

          // extract aerosol mixing ratios
          Real qnum_cur[num_modes], qaer_cur[num_modes][max_num_mode_species];
          for (int m = 0; m < num_modes; ++m) { // modes
            qnum_cur[m] = progs.n_mode_i[m](k);
            for (int a = 0; a < 7; ++a) { // aerosols
              qaer_cur[m][a] = progs.q_aero_i[m][a](k);
            }
          }

          Real qwtr_cur[num_modes] = {0, 0, 0, 0}; // water vapor mmr?

          // compute tendencies at this level
          Real dndt_ait, dmdt_ait, dso4dt_ait, dnh4dt_ait, dnclusterdt;
          compute_tendencies_(dt, temp, pmid, aircon, zmid, pblh, relhum,
                              uptkrate_so4, del_h2so4_gasprod,
                              del_h2so4_aeruptk, qgas_cur, qgas_avg, qnum_cur,
                              qaer_cur, qwtr_cur, dndt_ait, dmdt_ait,
                              dso4dt_ait, dnh4dt_ait, dnclusterdt);

          // Store the computed tendencies.
          tends.n_mode_i[nait](k) = dndt_ait;
          tends.q_aero_i[nait][iaer_so4](k) = dso4dt_ait;
          tends.q_gas[igas_h2so4](k) = -dso4dt_ait;
          // FIXME: what about dmdt_ait?
        });
  }

  // This function computes relevant tendencies at a single vertical level. It
  // was ported directly from the compute_tendencies subroutine in the
  // modal_aero_newnuc module from the MAM4 box model.
  KOKKOS_INLINE_FUNCTION
  void compute_tendencies_(
      Real deltat, Real temp, Real pmid, Real aircon, Real zmid, Real pblh,
      Real relhum, Real uptkrate_h2so4, Real del_h2so4_gasprod,
      Real del_h2so4_aeruptk, const Real qgas_cur[num_gases],
      const Real qgas_avg[num_gases], const Real qnum_cur[num_modes],
      const Real qaer_cur[num_modes][max_num_mode_species],
      const Real qwtr_cur[num_modes], Real &dndt_ait, Real &dmdt_ait,
      Real &dso4dt_ait, Real &dnh4dt_ait, Real &dnclusterdt) const {
    static constexpr Real avogadro =
        6.02214e26; // BAD_CONSTANT (Avogadro's number ~ molecules/kmole)
    static constexpr Real boltzmann =
        1.38065e-23; // BAD_CONSTANT (Boltzmann's constant ~ J/K/molecule)
    static constexpr Real rgas = boltzmann * avogadro; // BAD_CONSTANT
    static constexpr Real ln_nuc_rate_cutoff = -13.82;

    // min h2so4 vapor for nuc calcs = 4.0e-16 mol/mol-air ~= 1.0e4
    // molecules/cm3
    static constexpr Real qh2so4_cutoff = 4.0e-16;

    int newnuc_method_actual, pbl_nuc_wang2008_actual;

    constexpr int nsize = 1;
    Real dplom_mode[nsize], dphim_mode[nsize];
    int isize_group;

    Real cair; // air density
    Real so4vol, nh3ppt;

    Real radius_cluster = 0.0; // radius of newly formed cluster, in nm
    Real rateloge = 0.0;       // ln(J)
    Real cnum_h2so4 = 0.0;
    Real cnum_nh3 = 0.0;

    Real mass1p;
    Real mass1p_aithi, mass1p_aitlo;
    Real qh2so4_cur, qh2so4_avg, qh2so4_del;
    Real qnh3_cur, qnh3_del, qnh4a_del;
    Real qnuma_del;
    Real qso4a_del;
    Real relhumnn;
    Real tmpa, tmpb;
    Real tmp_frso4, tmp_uptkrate;
    Real dens_nh4so4a;

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

    if (qh2so4_avg <= qh2so4_cutoff) {
      // qh2so4_avg very low. assume no nucleation will happen
      // (diagnose so4 and nn4 tendencies and exit)
      tmp_frso4 = 1.0; // (uninitialized in original MAM code!)
      dso4dt_ait = dmdt_ait * tmp_frso4 / mw_so4a_host;
      dnh4dt_ait = dmdt_ait * (1.0 - tmp_frso4) / mw_nh4a_host;
      return;
    }

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
          newnuc_method_user_choice, newnuc_method_actual,       // in, out
          pbl_nuc_wang2008_user_choice, pbl_nuc_wang2008_actual, // in, out
          ln_nuc_rate_cutoff,                                    // in
          adjust_factor_bin_tern_ratenucl, adjust_factor_pbl_ratenucl,  // in
          pi, so4vol, nh3ppt, temp, relhumnn, zmid, pblh,               // in
          dnclusterdt, rateloge, cnum_h2so4, cnum_nh3, radius_cluster); // out

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

    // mass1p_... = mass (kg) of so4 & nh4 in a single particle of diameter
    // ... (assuming same dry density for so4 & nh4) mass1p_aitlo - dp =
    // dplom_mode(1); mass1p_aithi - dp = dphim_mode(1);
    tmpa = dens_so4a_host * pi / 6.0;
    mass1p_aitlo = tmpa * cube(dplom_mode[0]);
    mass1p_aithi = tmpa * cube(dphim_mode[0]);

    //----------------------------------------------------------------
    // Only do the cluster growth calculation when nucleation rate is
    // appreciable
    //----------------------------------------------------------------
    if (rateloge > ln_nuc_rate_cutoff) {
      nucleation::newnuc_cluster_growth(
          dnclusterdt, cnum_h2so4, cnum_nh3, radius_cluster, dplom_mode,
          dphim_mode, nsize, deltat, temp, relhumnn, cair, accom_coef_h2so4,
          mw_so4a, mw_so4a_host, mw_nh4a, avogadro, pi, qnh3_cur, qh2so4_cur,
          so4vol, tmp_uptkrate, isize_group, dens_nh4so4a, qh2so4_del, qnh3_del,
          qso4a_del, qnh4a_del, qnuma_del);
    } // nucleation rate is appreciable

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
    if (dndt_ait < 1.0e2) {
      dndt_ait = 0.0;
      dmdt_ait = 0.0;
    } else {

      // mirage2 code checked for complete h2so4 depletion here,
      // but this is now done in mer07_veh02_nuc_mosaic_1box
      mass1p = dmdt_ait / dndt_ait;

      // apply particle size constraints
      if (mass1p < mass1p_aitlo) {
        // reduce dndt to increase new particle size
        dndt_ait = dmdt_ait / mass1p_aitlo;
      } else if (mass1p > mass1p_aithi) {
        // reduce dmdt to decrease new particle size
        dmdt_ait = dndt_ait * mass1p_aithi;
      }
    }

    // *** apply adjustment factor to avoid unrealistically high
    // aitken number concentrations in mid and upper troposphere
    dndt_ait *= newnuc_adjust_factor_dnaitdt;
    dmdt_ait *= newnuc_adjust_factor_dnaitdt;

    //=================================
    // Diagnose so4 and nh4 tendencies
    //=================================
    // dso4dt_ait, dnh4dt_ait are (kmol/kmol-air/s)
    dso4dt_ait = dmdt_ait * tmp_frso4 / mw_so4a_host;
    dnh4dt_ait = dmdt_ait * (1.0 - tmp_frso4) / mw_nh4a_host;
  }
};

} // namespace mam4

#endif
