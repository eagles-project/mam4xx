#ifndef MAM4XX_MODALAER_OPT_HPP
#define MAM4XX_MODALAER_OPT_HPP

#include <Kokkos_Complex.hpp>
#include <haero/math.hpp>
#include <mam4xx/aero_config.hpp>
#include <mam4xx/ndrop.hpp>
#include <mam4xx/water_uptake.hpp>

namespace mam4 {
namespace modal_aer_opt {

using View1D = DeviceType::view_1d<Real>;
using View2D = DeviceType::view_2d<Real>;
using View3D = DeviceType::view_3d<Real>;
using ComplexView2D = DeviceType::view_2d<Kokkos::complex<Real>>;
using ComplexView1D = DeviceType::view_1d<Kokkos::complex<Real>>;
using View5D = Kokkos::View<Real *****>;

using ConstColumnView = haero::ConstColumnView;

constexpr int pver = mam4::nlev;
constexpr int ntot_amode = mam4::AeroConfig::num_modes();
constexpr int top_lev = ndrop::top_lev;
constexpr int pcnst = 40;
// ! min, max aerosol surface mode radius treated [m]
constexpr Real rmmin = 0.01e-6;
constexpr Real rmmax = 25.e-6;

// From radconstants
constexpr int nswbands = 14;
constexpr int nlwbands = 16;

// Dimension sizes in coefficient arrays used to parameterize aerosol radiative
// properties in terms of refractive index and wet radius
constexpr int ncoef = 5;
constexpr int prefr = 7;
constexpr int prefi = 10;

// BAD CONSTANT
constexpr Real small_value_40 = 1.e-40;
// BAD CONSTANT
constexpr Real small_value_60 = 1.e-60;

// Density of liquid water (STP)
constexpr Real rhoh2o = haero::Constants::density_h2o;
// ! reciprocal of gravit
constexpr Real rga = 1.0 / haero::Constants::gravity;
// FIXME: this values may cause differences because of number of digits in the
// Boltzmann's constant and Avogadro's number
constexpr Real rair = haero::Constants::r_gas_dry_air;

// ! These are indices to the band for diagnostic output
constexpr int idx_sw_diag = 10; // index to sw visible band
constexpr int idx_nir_diag = 8; // index to sw near infrared (778-1240 nm) band
constexpr int idx_uv_diag = 11; // index to sw uv (345-441 nm) band

constexpr int maxd_aspectype = 14;
// FIXME; is this values set somewhere else?
constexpr int max_nspec = 7;

// dimensions for reading refractive index from modal radioactive properties
// files
constexpr int coef_number = 5;   // number of coefficients
constexpr int refindex_real = 7; // real refractive index
constexpr int refindex_im = 10;  // imaginary refractive index

struct AerosolOpticsDeviceData {
  // devices views
  // FIXME: add description of these tables.
  View1D refitabsw[ntot_amode][nswbands];
  View1D refrtabsw[ntot_amode][nswbands];
  View3D abspsw[ntot_amode][nswbands];
  View3D asmpsw[ntot_amode][nswbands];

  View1D refrtablw[ntot_amode][nlwbands];
  View1D refitablw[ntot_amode][nlwbands];
  View3D absplw[ntot_amode][nlwbands];
  View3D extpsw[ntot_amode][nswbands];

  ComplexView2D specrefndxsw; // specrefndxsw( nswbands, maxd_aspectype )
  ComplexView2D specrefndxlw;
  ComplexView1D crefwlw;
  ComplexView1D crefwsw;
};

inline void set_aerosol_optics_data_for_modal_aero_sw_views(
    AerosolOpticsDeviceData &aersol_optics_data) {

  for (int d1 = 0; d1 < ntot_amode; ++d1) {
    for (int d5 = 0; d5 < nswbands; ++d5) {
      aersol_optics_data.abspsw[d1][d5] =
          View3D("abspsw", coef_number, refindex_real, refindex_im);
      aersol_optics_data.extpsw[d1][d5] =
          View3D("extpsw", coef_number, refindex_real, refindex_im);
      aersol_optics_data.asmpsw[d1][d5] =
          View3D("asmpsw", coef_number, refindex_real, refindex_im);

      aersol_optics_data.refrtabsw[d1][d5] = View1D("refrtabsw", refindex_real);

      aersol_optics_data.refitabsw[d1][d5] = View1D("refitabsw", refindex_im);
    } // d5
  }   // d1

} // configure_aerosol_optics_data

inline void set_aerosol_optics_data_for_modal_aero_lw_views(
    AerosolOpticsDeviceData &aersol_optics_data) {

  for (int d1 = 0; d1 < ntot_amode; ++d1)
    for (int d5 = 0; d5 < nlwbands; ++d5) {
      aersol_optics_data.absplw[d1][d5] =
          View3D("absplw3", coef_number, refindex_real, refindex_im);
      aersol_optics_data.refrtablw[d1][d5] = View1D("refrtablw", refindex_real);
      aersol_optics_data.refitablw[d1][d5] = View1D("refitablw", refindex_im);
    } // d5

} // set_aerosol_optics_data_for_modal_aero_lw_views

inline void
set_complex_views_modal_aero(AerosolOpticsDeviceData &aersol_optics_data) {
  aersol_optics_data.specrefndxsw =
      ComplexView2D("specrefndxsw", nswbands, maxd_aspectype);
  aersol_optics_data.specrefndxlw =
      ComplexView2D("specrefndxlw", nlwbands, maxd_aspectype);

  aersol_optics_data.crefwlw = ComplexView1D("crefwlw", nlwbands);
  aersol_optics_data.crefwsw = ComplexView1D("crefwsw", nswbands);
} // set_complex_views_modal_aero

struct DiagnosticsAerosolOpticsSW {
  // FIXME; add description of these diagnostics variables
  ColumnView extinct; //        ! aerosol extinction [1/m]
  ColumnView absorb;  //         ! aerosol absorption [1/m]
  Real aodnir;
  Real aoduv;
  Real dustaodmode[ntot_amode];
  Real aodmode[ntot_amode];
  Real burdenmode[ntot_amode];
  Real aodabsbc;
  Real aodvis;
  Real aodall;
  Real ssavis;
  Real aodabs;
  Real burdendust;
  Real burdenso4;
  Real burdenbc;
  Real burdenpom;
  Real burdensoa;
  Real burdenseasalt;
  Real burdenmom;
  Real momaod;
  Real dustaod;
  Real so4aod; // total species AOD
  Real pomaod;
  Real soaaod;
  Real bcaod;
  Real seasaltaod;
};

KOKKOS_INLINE_FUNCTION
void modal_size_parameters(const Real sigma_logr_aer,
                           const Real dgnumwet, // in
                           Real &radsurf, Real &logradsurf, Real cheb[ncoef],
                           const bool ismethod2) {

  // ncol
  // sigma_logr_aer  ! geometric standard deviation of number distribution
  // dgnumwet(:,:)   ! aerosol wet number mode diameter [m]
  // radsurf(:,:)    ! aerosol surface mode radius [m]
  // logradsurf(:,:) ! log(aerosol surface mode radius)
  // cheb(:,:,:)     ! chebychev polynomial parameters

  // FORTRAN refactoring: ismethod is tempararily used to ensure BFB test
  /// bool ismethod2

  // integer  :: icol, kk, nc
  // real(r8) :: alnsg_amode      ! log(sigma)
  // real(r8) :: explnsigma
  // real(r8) :: xrad ! normalized aerosol radius
  // !-------------------------------------------------------------------------------
  constexpr Real half = 0.5;
  constexpr Real one = 1.0;
  constexpr Real two = 2.0;
  const Real xrmin = haero::log(rmmin);
  const Real xrmax = haero::log(rmmax);

  const Real alnsg_amode = haero::log(sigma_logr_aer);
  const Real explnsigma = haero::exp(two * alnsg_amode * alnsg_amode);
  // do kk = top_lev, pver
  // do icol = 1, ncol
  // ! convert from number mode diameter to surface area
  radsurf = half * dgnumwet * explnsigma;
  // ! --------------- FORTRAN refactoring -------------------
  // ! here two calculations are used to ensure passing BFB test
  // ! can be simplified (there is only round-off difference
  if (ismethod2) {
    logradsurf = haero::log(half * dgnumwet) + two * alnsg_amode * alnsg_amode;
  } else {
    logradsurf = haero::log(radsurf);

  } // ismethod2
  // ! --------------- FORTRAN refactoring -------------------

  // ! normalize size parameter
  Real xrad = mam4::utils::min_max_bound(xrmin, xrmax, logradsurf);
  xrad = (two * xrad - xrmax - xrmin) / (xrmax - xrmin);
  // chebyshev polynomials
  cheb[0] = one;
  cheb[1] = xrad;
  for (int nc = 2; nc < ncoef; ++nc) {
    cheb[nc] = two * xrad * cheb[nc - 1] - cheb[nc - 2];
  } // nc

} // modal_size_parameters

KOKKOS_INLINE_FUNCTION
void calc_parameterized(const Real coef[ncoef], const Real cheb_k[ncoef],
                        Real &para) {
  // calculate parameterized absorption, extinction or asymmetry factor
  // further calculations are needed. see modal_aero_sw and modal_aero_lw
  // ncol,ncoef
  // coef(pcols,ncoef)
  // cheb_k(ncoef,pcols)
  // para(pcols)
  constexpr Real half = 0.5;
  para = half * coef[0];
  for (int nc = 1; nc < ncoef; ++nc) {
    // printf("cheb_k[nc] %e coef[nc] %e \n", cheb_k[nc],  coef[nc]);
    para += cheb_k[nc] * coef[nc];
  }

} // calc_parameterized

KOKKOS_INLINE_FUNCTION
void update_aod_spec(const Real scath2o,
                     const Real absh2o, // in
                     const Real sumhygro, const Real sumscat,
                     const Real sumabs, // in
                     const Real hygro_s, const Real palb,
                     const Real dopaer, // in
                     Real &scat_s, Real &abs_s, Real &aod_s) {
  // scath2o, absh2o, sumscat, sumabs, sumhygro
  // hygro_s, palb, dopaer
  // scat_s, abs_s, aod_s  ! scatering, absorption and aod for a species
  // update aerosol optical depth from scattering and absorption
  constexpr Real one = 1.0;
  scat_s = (scat_s + scath2o * hygro_s / sumhygro) / sumscat;
  abs_s = (abs_s + absh2o * hygro_s / sumhygro) / sumabs;
  aod_s += (abs_s * (one - palb) + palb * scat_s) * dopaer;

} // update_aod_spec

KOKKOS_INLINE_FUNCTION
void calc_volc_ext(const int trop_level, const ConstColumnView &state_zm,
                   const ColumnView &ext_cmip6_sw, const ColumnView &extinct,
                   Real &tropopause_m) {
  // calculate contributions from volcanic aerosol extinction
  // trop_level(pcols)!tropopause level for each column
  // state_zm(:,:) ! state%zm [m]
  // ext_cmip6_sw(pcols,pver) ! aerosol shortwave extinction [1/m]
  // extinct(pcols,pver) ! aerosol extinction [1/m]
  // tropopause_m(pcols) ! tropopause height [m]
  // kk_tropp = trop_level(icol)
  //

  constexpr Real half = 0.5;

  // diagnose tropopause height
  tropopause_m = state_zm(trop_level); //! in meters
  // update tropopause layer first
  extinct(trop_level) = half * (extinct(trop_level) + ext_cmip6_sw(trop_level));
  // extinction is assigned read in values only for visible band above
  // tropopause
  for (int kk = 0; kk < trop_level; ++kk) {
    extinct(kk) = ext_cmip6_sw(kk);
  }

} // calc_volc_ext

KOKKOS_INLINE_FUNCTION
void calc_diag_spec(const Real specmmr_k, const Real mass_k, const Real vol,
                    const Real specrefr, const Real specrefi,
                    const Real hygro_aer, Real &burden_s, Real &scat_s,
                    Real &abs_s, Real &hygro_s) {
  // calculate some diagnostics for a species
  // specmmr_k(:)   mmr at level kk [kg/kg]
  // mass_k(:)         mass at layer kk [kg/m2]
  // vol(:)  volume concentration of aerosol species [m3/kg]
  // specrefr, specrefi   real and image part of specrefindex
  // hygro_aer        aerosol hygroscopicity [unitless]
  // burden_s(pcols)  aerosol burden of species [kg/m2]
  // scat_s(pcols)    scattering of species [unitless]
  // abs_s(pcols)     absorption of species [unit?]
  // hygro_s(pcols)   hygroscopicity of species [unitless]
  constexpr Real zero = 0;

  burden_s = zero;
  burden_s += specmmr_k * mass_k;
  scat_s = vol * specrefr;
  abs_s = -vol * specrefi;
  hygro_s = vol * hygro_aer;

} // calc_diag_spec

KOKKOS_INLINE_FUNCTION
void calc_refin_complex(const int lwsw, const int ilwsw, const Real qaerwat_kk,
                        const Real *specvol, const ComplexView2D &specrefindex,
                        const int nspec, const ComplexView1D &crefwlw,
                        const ComplexView1D &crefwsw, Real &dryvol,
                        Real &wetvol, Real &watervol,
                        Kokkos::complex<Real> &crefin, Real &refr, Real &refi) {
  /*-------------------------------------------------------------------
calculate complex refractive index
also output wetvol and watervol
-------------------------------------------------------------------*/

  // lwsw   ! indicator if this is lw or sw lw =0 and sw =1
  // ncol, ilwsw
  // qaerwat_kk(:)   ! aerosol water at level kk [g/g]
  // specvol(:,:)    ! volume concentration of aerosol specie [m3/kg]
  // complex(r8), intent(in) :: specrefindex(:,:)     ! species refractive index

  // dryvol(pcols)    ! volume concentration of aerosol mode [m3/kg]
  // wetvol(pcols)    ! volume concentration of wet mode [m3/kg]
  // watervol(pcols)  ! volume concentration of water in each mode [m3/kg]
  // refr(pcols)      ! real part of refractive index
  // refi(pcols)      ! imaginary part of refractive index
  // complex(r8),intent(out) :: crefin(pcols) ! complex refractive index

  // refractive index for water read in read_water_refindex
  // crefwsw(nswbands) ! complex refractive index for water visible
  // crefwlw(nlwbands) ! complex refractive index for water infrared

  // FIXME
  // if ((lwsw /= 'lw') .and. (lwsw /= 'sw')) then
  //     call endrun('calc_refin_complex is called with '// lwsw// ', it should
  //     be called with either lw or sw')
  // endif

  // crefin(:ncol) = (0._r8, 0._r8)
  constexpr Real zero = 0;
  dryvol = zero;

  crefin = {};

  for (int i = 0; i < nspec; ++i) {
    dryvol += specvol[i];
    crefin += specvol[i] * specrefindex(i, ilwsw);
  }
  // printf("qaerwat_kk %e  rhoh2o %e \n", qaerwat_kk,  rhoh2o);

  watervol = qaerwat_kk / rhoh2o;
  wetvol = watervol + dryvol;
  // printf("wetvol %e watervol %e dryvol %e \n ", wetvol, watervol,dryvol);
  if (watervol < zero && lwsw == 0) // lwsw=='lw'
  {
    // BAD CONSTANT
    // FIXME
    // if (haero::abs(watervol > 1.e-1*wetvol))
    // {
    // 	write(iulog,*) 'watervol,wetvol,dryvol=',watervol(icol), &
    //                  wetvol(icol),dryvol(icol)
    // }
    watervol = zero;
    wetvol = dryvol;
  } // end if watervol < zero && lwsw=='lw'

  // ! some different treatments for lw and sw
  if (lwsw == 0) // lwsw=='lw'
  {
    crefin += watervol * crefwlw(ilwsw);
    // BAD CONSTANT
    if (wetvol > small_value_40) {
      crefin /= wetvol;
    } // end if wetvol(icol) > small_value_40

  } else if (lwsw == 1) //  lwsw=='sw
  {

    crefin += watervol * crefwsw(ilwsw);
    // BAD CONTANT
    crefin /= haero::max(wetvol, small_value_60);

  } // lwsw=='lw'
  // FIXME
  refr = crefin.real();
  refi = crefin.imag();

} // calc_refin_complex

KOKKOS_INLINE_FUNCTION
void compute_factors(const int prefri, const Real ref_ind,
                     const Real *ref_table, int &ix, Real &tt) {
  // Compute factors for the real or imaginary parts

  // prefri, ncol
  // ref_table(:) !refractive index table [unitless]
  // ref_ind(:)   !refractive index       [unitless]

  // FORTRAN refactor note: "do_print" is kept to maintain code consistenty with
  // the original code. THis can be removed from the C++ ported code
  //  logical, intent(in), optional :: do_print ! to print log msg or not

  // !intent-inouts/outs
  // ix(:)
  // tt(:)
  // BAD CONSTANT
  constexpr Real threshold = 1.e-20;
  constexpr Real zero = 0;
  ix = 1;
  tt = zero;
  if (prefri > 1) {

    int ii = 0;
    for (ii = 0; ii < prefri; ++ii) {
      if (ref_ind < ref_table[ii]) {
        break;
      }
    } // ii
    // FIXME: check Fortran to C++ indexing conversion
    ix = haero::max(ii - 1, 0);
    const int ip1 = haero::min(ix + 1, prefri - 1);
    const Real dx = ref_table[ip1] - ref_table[ix];
    if (haero::abs(dx) > threshold) {
      tt = (ref_ind - ref_table[ix]) / dx;
      // FIXME: I did not port the following:
      // if (present(do_print) .and. do_print) then
      //       if(tt(ic) < 0._r8 .or. tt(ic) > 1._r8) then
      //          write(iulog,*)
      //          'tt,ref_ind,ix,ref_table,dx=',tt(ic),ref_ind(ic),ix(ic),ref_table(ix(ic)),dx
      //       endif
      //    endif

    } // (abs(dx) > threshold)

  } // if(prefri > 1)

} // compute_factors

KOKKOS_INLINE_FUNCTION
void binterp(const View3D &table, const Real ref_real, const Real ref_img,
             const Real ref_real_tab[prefr], const Real ref_img_tab[prefi],
             int &itab, int &jtab, Real &ttab, Real &utab, Real coef[ncoef],
             const int itab_1) {
  /*------------------------------------------------------------------------------
   Bilinear interpolation along the refractive index dimensions
   of the table to estimate Chebyshev coefficients at an
   intermediate refractive index.

   In short wave, the first call computes itab, jtab, ttab, utab and coef.
   The subsequent calls use itab, jtab, ttab and utab as inputs and compute coef

   In long wave, we have just one call to compute itab,jtab,ttab, utab and coef
  ------------------------------------------------------------------------------*/

  // intent-ins
  // ncol
  // table(ncoef,prefr,prefi)
  // ref_real(pcols), ref_img(pcols) !real and imganinary parts of refractive
  // indices [unitless] ref_real_tab(prefr), ref_img_tab(prefi) !real and
  // imganinary table refractive indices [unitless]

  // intent-inouts/outs
  // itab(pcols), jtab(pcols)
  // ttab(pcols), utab(pcols)
  // coef(pcols,ncoef) !coefficient interpolated bilinearly
  // FIXME; maybe we need to loop over cols
  constexpr Real one = 1.0;
  if (itab_1 <= 0.0) {
    // compute factors for the real part
    compute_factors(prefr, ref_real, ref_real_tab, itab, ttab);

    // !compute factors for the imaginary part
    compute_factors(prefi, ref_img, ref_img_tab, jtab, utab);
  } // itab_1 < -1

  const Real tu = ttab * utab;
  const Real tuc = ttab - tu;
  const Real tcuc = one - tuc - utab;
  const Real tcu = utab - tu;
  // FIXME: check Fortran to C++ indexing
  const int jp1 = haero::min(jtab + 1, prefi - 1);
  const int ip1 = haero::min(itab + 1, prefr - 1);
  for (int icoef = 0; icoef < ncoef; ++icoef) {
    coef[icoef] = tcuc * table(icoef, itab, jtab) +
                  tuc * table(icoef, ip1, jtab) + tu * table(icoef, ip1, jp1) +
                  tcu * table(icoef, itab, jp1);
  } // icoef

} // binterp

KOKKOS_INLINE_FUNCTION
void modal_aero_wateruptake_dr(
    const View2D &state_q, const ConstColumnView &temperature,
    const ConstColumnView &pmid, const ConstColumnView &cldn,
    const View2D &dgnumdry_m, const View2D &dgnumwet_m, const View2D &qaerwat_m,
    // const int list_idx_in,
    int nspec_amode[AeroConfig::num_modes()],
    Real specdens_amode[maxd_aspectype], Real spechygro[maxd_aspectype],
    int lspectype_amode[maxd_aspectype][AeroConfig::num_modes()]) {

  // dgnumdry_m => dgncur_a
  // dgnumwet_m => dgncur_awet
  // qaerwat_m => qaerwat
  // int &nspec_amode2 = nspec_amode;
  // Real &specdens_amode2 = specdens_amode;
  // Real & spechygro2 = spechygro;
  // int &lspectype_amode2 = lspectype_amode;
  for (int kk = top_lev; kk < pver; ++kk) {

    const auto state_q_kk = Kokkos::subview(state_q, kk, Kokkos::ALL());
    const auto dgnumdry_m_kk = Kokkos::subview(dgnumdry_m, kk, Kokkos::ALL());
    const auto dgnumwet_m_kk = Kokkos::subview(dgnumwet_m, kk, Kokkos::ALL());
    const auto qaerwat_m_kk = Kokkos::subview(qaerwat_m, kk, Kokkos::ALL());
    mam4::water_uptake::modal_aero_water_uptake_dr(
        nspec_amode, specdens_amode, spechygro, lspectype_amode,
        state_q_kk.data(), temperature(kk), pmid(kk), cldn(kk),
        dgnumdry_m_kk.data(), dgnumwet_m_kk.data(), qaerwat_m_kk.data());

  } // kk

  // placeholder
} // modal_aero_wateruptake_dr

KOKKOS_INLINE_FUNCTION
void modal_aero_sw(
    const Real dt, const View2D &state_q, const ConstColumnView &state_zm,
    const ConstColumnView &temperature, const ConstColumnView &pmid,
    const ConstColumnView &pdel, const ConstColumnView &pdeldry,
    const ConstColumnView &cldn,
    // const int nnite,
    // idxnite,
    const bool is_cmip6_volc, const ColumnView &ext_cmip6_sw,
    const int trop_level,
    // const ColumnView qqcw_fld[pcnst],
    const View2D &tauxar, const View2D &wa, const View2D &ga, const View2D &fa,
    //
    int nspec_amode[ntot_amode], Real sigmag_amode[ntot_amode],
    int lmassptr_amode[maxd_aspectype][ntot_amode],
    Real spechygro[maxd_aspectype], Real specdens_amode[maxd_aspectype],
    int lspectype_amode[maxd_aspectype][ntot_amode],
    // FIXME
    const mam4::AeroId specname_amode[9],
    const AerosolOpticsDeviceData &aersol_optics_data,
    // diagnostic
    DiagnosticsAerosolOpticsSW &diagnostics_aerosol_optics_sw,
    // work view
    const ColumnView &mass, const ColumnView &air_density, const View2D &cheb,
    const View2D &dgnumwet_m, const View2D &dgnumdry_m,
    const ColumnView &radsurf, const ColumnView &logradsurf,
    const ComplexView2D &specrefindex, const View2D &qaerwat_m) {

  auto extinct = diagnostics_aerosol_optics_sw
                     .extinct; //        ! aerosol extinction [1/m]
  auto absorb = diagnostics_aerosol_optics_sw
                    .absorb; //         ! aerosol absorption [1/m]
  auto &aodnir = diagnostics_aerosol_optics_sw.aodnir;
  auto &aoduv = diagnostics_aerosol_optics_sw.aoduv;
  auto &dustaodmode = diagnostics_aerosol_optics_sw.dustaodmode;
  auto &aodmode = diagnostics_aerosol_optics_sw.aodmode;
  auto &burdenmode = diagnostics_aerosol_optics_sw.burdenmode;
  auto &aodabsbc = diagnostics_aerosol_optics_sw.aodabsbc;
  auto &aodvis = diagnostics_aerosol_optics_sw.aodvis;
  auto &aodall = diagnostics_aerosol_optics_sw.aodall;
  auto &ssavis = diagnostics_aerosol_optics_sw.ssavis;
  auto &aodabs = diagnostics_aerosol_optics_sw.aodabs;
  auto &burdendust = diagnostics_aerosol_optics_sw.burdendust;
  auto &burdenso4 = diagnostics_aerosol_optics_sw.burdenso4;
  auto &burdenbc = diagnostics_aerosol_optics_sw.burdenbc;
  auto &burdenpom = diagnostics_aerosol_optics_sw.burdenpom;
  auto &burdensoa = diagnostics_aerosol_optics_sw.burdensoa;
  auto &burdenseasalt = diagnostics_aerosol_optics_sw.burdenseasalt;
  auto &burdenmom = diagnostics_aerosol_optics_sw.burdenmom;
  auto &momaod = diagnostics_aerosol_optics_sw.momaod;
  auto &dustaod = diagnostics_aerosol_optics_sw.dustaod;
  auto &so4aod = diagnostics_aerosol_optics_sw.so4aod; // total species AOD
  auto &pomaod = diagnostics_aerosol_optics_sw.pomaod;
  auto &soaaod = diagnostics_aerosol_optics_sw.soaaod;
  auto &bcaod = diagnostics_aerosol_optics_sw.bcaod;
  auto &seasaltaod = diagnostics_aerosol_optics_sw.seasaltaod;

  const Real xrmax = haero::log(rmmax);
  // ! calculates aerosol sw radiative properties
  // dt               !timestep [s]
  // lchnk            ! chunk id
  // ncol             ! number of active columns in the chunk
  //  state_q(:,:,:)   ! water and tracers (state%q) in state [kg/kg]
  //  state_zm(:,:)    ! mid-point height (state%zm) [m]
  //  temperature(:,:) ! temperature [K]
  //  pmid(:,:)        ! mid-point pressure [Pa]
  //  pdel(:,:)        ! pressure interval [Pa]
  //  pdeldry(:,:)     ! dry mass pressure interval [Pa]

  //  cldn(:,:)         ! layer cloud fraction [fraction]

  //  nnite          ! number of night columns
  //  idxnite(nnite) ! local column indices of night columns
  //  trop_level(pcols)!tropopause level for each column
  //  ext_cmip6_sw(pcols,pver) ! aerosol shortwave extinction [1/m]
  //  is_cmip6_volc

  //  qqcw(:)               ! Cloud borne aerosols mixing ratios [kg/kg or 1/kg]
  //  tauxar(pcols,0:pver,nswbands) ! layer extinction optical depth [1]
  //  wa(pcols,0:pver,nswbands)     ! layer single-scatter albedo [1]
  //  ga(pcols,0:pver,nswbands)     ! asymmetry factor [1]
  //  fa(pcols,0:pver,nswbands)     ! forward scattered fraction [1]

  // ! Local variables
  // real(r8),    pointer :: specmmr(:,:)        ! species mass mixing ratio
  // [kg/kg] spectype            ! species type hygro_aer           !
  // hygroscopicity [1]

  // sigma_logr_aer         ! geometric standard deviation of number
  // distribution radsurf(pcols,pver)    ! aerosol surface mode radius
  // logradsurf(pcols,pver) ! log(aerosol surface mode radius)
  // cheb(ncoef,pcols,pver) ! chebychev polynomial parameters

  // specvol(:,:)        ! volume concentration of aerosol specie [m3/kg]
  // specdens(:)         ! species density for all species [kg/m3]
  // specrefindex(:,:)     ! species refractive index

  // refr(pcols)     ! real part of refractive index
  // refi(pcols)     ! imaginary part of refractive index
  // crefin(pcols)   ! complex refractive index

  // dryvol(pcols)   ! volume concentration of aerosol mode [m3/kg]
  // watervol(pcols) ! volume concentration of water in each mode [m3/kg]
  // wetvol(pcols)   ! volume concentration of wet mode [m3/kg]

  // pext(pcols)     ! parameterized specific extinction [m2/kg]

  // pabs(pcols)     ! parameterized specific absorption [m2/kg]
  // pasm(pcols)     ! parameterized asymmetry factor [unitless?]
  // palb(pcols)     ! parameterized single scattering albedo [unitless]

  constexpr Real zero = 0.0;
  constexpr Real one = 1.0;

  // FORTRAN refactoring: For prognostic aerosols only, other options are
  // removed
  // const int list_idx = 0; //   ! index of the climate or a diagnostic list

  // FIXME: We need to set these values outside of this subroutine
  //  ! zero'th layer does not contain aerosol
  //  tauxar(1:ncol,0,:)  = 0._r8
  //  wa(1:ncol,0,:)      = 0.925_r8
  //  ga(1:ncol,0,:)      = 0.850_r8
  //  fa(1:ncol,0,:)      = 0.7225_r8

  // diagnostics for visible band summed over modes/output diagnostics
  aodvis = zero;     // extinction optical depth
  aodall = zero;     // extinction optical depth
  aodabs = zero;     // absorption optical depth
  burdendust = zero; // ! burden for each aerosol species [kg/m2]
  burdenso4 = zero;
  burdenbc = zero;
  burdenpom = zero;
  burdensoa = zero;
  burdenseasalt = zero;
  burdenmom = zero; // ! burden for each aerosol species [kg/m2]
  momaod = zero;    // total species AOD
  // FIXME: I need to add code at the end this subroutine. if (list_idx == 0)
  // then
  ssavis = zero; // Aerosol singel-scatter albedo [unitless]

  aodabsbc = zero; //       absorption optical depth of BC
  dustaod = zero;
  so4aod = zero; // total species AOD
  pomaod = zero;
  soaaod = zero;
  bcaod = zero;
  seasaltaod = zero; // total species AOD

  // diags for other bands
  aodnir = zero; //              extinction optical depth in nir
  aoduv = zero;  //               extinction optical depth in uv
  // dustaodmode[ntot_amode] = zero;//         ! dust aod in aerosol mode [1]

  // Local variables
  Real dustvol = zero; //  volume concentration of dust in aerosol mode (m3/kg)
  //  scattering albedo (?) for each aerosol species [unitless]
  Real scatdust = zero;
  Real scatso4 = zero;
  Real scatbc = zero;
  Real scatpom = zero;
  Real scatsoa = zero;
  Real scatseasalt = zero;
  Real scatmom = zero; //

  // absoprtion for each aerosol species [unit?]
  Real absdust = zero;
  Real absso4 = zero;
  Real absbc = zero;
  Real abspom = zero;
  Real abssoa = zero;
  Real absseasalt = zero;
  Real absmom = zero; //

  // hygroscopicity for each aerosol species [unitless]
  Real hygrodust = zero;
  Real hygroso4 = zero;
  Real hygrobc = zero;
  Real hygropom = zero;
  Real hygrosoa = zero;
  Real hygroseasalt = zero;
  Real hygromom = zero;

  Real tropopause_m = zero; // tropopause height [m]

  // FIXME: are these diag variables?
  // Real specrefr, specrefi=zero;// real and imag parts of specref
  Real scath2o, absh2o = zero; // scattering and absorption of h2o
  Real sumscat, sumabs,
      sumhygro = zero; // sum of scattering , absoprtion and hygroscopicity

  for (int kk = 0; kk < pver; ++kk) {
    // layer dry mass [kg/m2]
    mass(kk) = pdeldry(kk) * rga;
    // dry air density [kg/m3]
    air_density(kk) = pmid(kk) / (rair * temperature(kk));

    // diagnostics for visible band summed over modes
    extinct(kk) = zero;
    absorb(kk) = zero;
    // initialize output variables
    for (int i = 0; i < nswbands; ++i) {
      tauxar(kk, i) = zero;
      wa(kk, i) = zero;
      ga(kk, i) = zero;
      fa(kk, i) = zero;
    }
  }

  // Calculate aerosol size distribution parameters and aerosol water uptake
  // For prognostic aerosols
  // We compute dgnumdry_m using calcsize process.
  // modal_aero_calcsize_sub(state_q, pdel, dt, qqcw_fld, list_idx, false,
  //                         dgnumdry_m); // ! out

  modal_aero_wateruptake_dr(state_q, temperature, pmid, cldn, dgnumdry_m,
                            dgnumwet_m, qaerwat_m, nspec_amode, specdens_amode,
                            spechygro, lspectype_amode);
  // ! loop over all aerosol modes

  Real specvol[max_nspec] = {};
  for (int mm = 0; mm < ntot_amode; ++mm) {
    // ! diagnostics for visible band for each mode
    burdenmode[mm] = zero;
    aodmode[mm] = zero;
    dustaodmode[mm] = zero; //  dust aod in aerosol mode [1]
    // ! get mode info
    const int nspec = nspec_amode[mm];
    const Real sigma_logr_aer = sigmag_amode[mm];

    for (int kk = top_lev; kk < pver; ++kk) {

      auto cheb_kk = Kokkos::subview(cheb, Kokkos::ALL(), kk);
      modal_size_parameters(sigma_logr_aer, dgnumwet_m(kk, mm), // in
                            radsurf(kk), logradsurf(kk), cheb_kk.data(), false);
    } // kk

    // FIXME is a complex number
    // auto specrefindex_ll =
    // Kokkos::subview(specrefndxsw,Kokkos::ALL(),spectype_amode[ll][mm]);
    // FIXME: is there a way of avoiding this copy?
    // specrefindex(ll,:) = specrefndxsw(:,lspectype_amode[ll][mm])
    for (int iswbands = 0; iswbands < nswbands; ++iswbands) {
      for (int ll = 0; ll < nspec; ++ll) {
        // Fortran to C++ indexing
        specrefindex(ll, iswbands) = aersol_optics_data.specrefndxsw(
            iswbands, lspectype_amode[ll][mm] - 1);
      }
    }

    for (int isw = 0; isw < nswbands; ++isw) {

      // savaervis ! true if visible wavelength (0.55 micron)
      // savaernir ! true if near ir wavelength (~0.88 micron)
      // savaeruv  ! true if uv wavelength (~0.35 micron)

      const bool savaervis = isw == idx_sw_diag ? true : false;
      const bool savaeruv = isw == idx_uv_diag ? true : false;
      const bool savaernir = isw == idx_nir_diag ? true : false;

      for (int kk = top_lev; kk < pver; ++kk) {

        // FIXME: Note that these variables will be only saved for pver and
        // nswbands.

        dustvol = zero;

        hygrodust = zero;
        hygroso4 = zero;
        hygrobc = zero;
        hygropom = zero;
        hygrosoa = zero;
        hygroseasalt = zero;
        hygromom = zero;

        scatdust = zero;
        scatso4 = zero;
        scatbc = zero;
        scatpom = zero;
        scatsoa = zero;
        scatseasalt = zero;
        scatmom = zero;

        absdust = zero;
        absso4 = zero;
        absbc = zero;
        abspom = zero;
        abssoa = zero;
        absseasalt = zero;
        absmom = zero;

        // ! aerosol species loop

        for (int ll = 0; ll < nspec; ++ll) {

          // get aerosol properties and save for each species
          // Fortran to C++ indexing
          auto specmmr = Kokkos::subview(state_q, Kokkos::ALL(),
                                         lmassptr_amode[ll][mm] - 1);
          // Fortran to C++ indexing
          auto spectype = specname_amode[lspectype_amode[ll][mm] - 1];
          const Real hygro_aer = spechygro[lspectype_amode[ll][mm] - 1];
          const Real specdens = specdens_amode[lspectype_amode[ll][mm] - 1];

          // allocate(specvol(pcols,nspec),stat=istat)
          specvol[ll] = specmmr(kk) / specdens;

          // ! compute some diagnostics for visible band only
          if (savaervis) {
            /// FIXME complex
            const Real specrefr = specrefindex(ll, isw).real();
            const Real specrefi = specrefindex(ll, isw).imag();

            burdenmode[mm] += specmmr(kk) * mass(kk);

            if (spectype == AeroId::DST) {

              calc_diag_spec(specmmr(kk), mass(kk), specvol[ll], specrefr,
                             specrefi, hygro_aer, burdendust, scatdust, absdust,
                             hygrodust);
              dustvol = specvol[ll];
            }

            if (spectype == AeroId::SO4) {
              calc_diag_spec(specmmr(kk), mass(kk), specvol[ll], specrefr,
                             specrefi, hygro_aer, burdenso4, scatso4, absso4,
                             hygroso4);
            }
            if (spectype == AeroId::BC) {
              calc_diag_spec(specmmr(kk), mass(kk), specvol[ll], specrefr,
                             specrefi, hygro_aer, burdenbc, scatbc, absbc,
                             hygrobc);
            }

            if (spectype == AeroId::POM) {
              calc_diag_spec(specmmr(kk), mass(kk), specvol[ll], specrefr,
                             specrefi, hygro_aer, burdenpom, scatpom, abspom,
                             hygropom);
            }

            if (spectype == AeroId::SOA) {
              calc_diag_spec(specmmr(kk), mass(kk), specvol[ll], specrefr,
                             specrefi, hygro_aer, burdensoa, scatsoa, abssoa,
                             hygrosoa);
            }

            if (spectype == AeroId::NaCl) {
              calc_diag_spec(specmmr(kk), mass(kk), specvol[ll], specrefr,
                             specrefi, hygro_aer, burdenseasalt, scatseasalt,
                             absseasalt, hygroseasalt);
            }

            if (spectype == AeroId::MOM) {
              calc_diag_spec(specmmr(kk), mass(kk), specvol[ll], specrefr,
                             specrefi, hygro_aer, burdenmom, scatmom, absmom,
                             hygromom);
            }

          } // savaervis

        } // ll species loop ll

        // lw =0 and sw =1
        Real dryvol, wetvol, watervol = {};
        Kokkos::complex<Real> crefin = {};
        Real refr, refi = {};

        calc_refin_complex(1, isw, qaerwat_m(kk, mm), specvol, specrefindex,
                           nspec, aersol_optics_data.crefwlw,
                           aersol_optics_data.crefwsw, dryvol, wetvol, watervol,
                           crefin, refr, refi);

        // interpolate coefficients linear in refractive index
        // first call calcs itab,jtab,ttab,utab

        const auto sub_extpsw = aersol_optics_data.extpsw[mm][isw];
        // Kokkos::subview(
        // extpsw, mm, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(), isw);
        const auto ref_real_tab = aersol_optics_data.refrtabsw[mm][isw];
        // Kokkos::subview(refrtabsw, mm, Kokkos::ALL(), isw);
        const auto ref_img_tab = aersol_optics_data.refitabsw[mm][isw];
        // Kokkos::subview(refitabsw, mm, Kokkos::ALL(), isw);

        int itab = zero;   // index for Bilinear interpolation
        int itab_1 = zero; // ! index for Bilinear interpolation column 1
        // FIXME: part of binterp only apply for first column
        int jtab = zero;
        Real ttab, utab = {}; // coef for Bilinear interpolation
        Real cext[ncoef], cabs[ncoef],
            casm[ncoef] = {}; //  ! coefficient for extinction, absoption, and
                              //  asymmetry [unitless]

        binterp(sub_extpsw, refr, refi, ref_real_tab.data(), ref_img_tab.data(),
                itab, jtab, ttab, utab, cext, itab_1);

        const auto sub_abspsw = aersol_optics_data.abspsw[mm][isw];
        // Kokkos::subview(
        // abspsw, mm, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(), isw);

        binterp(sub_abspsw, refr, refi, ref_real_tab.data(), ref_img_tab.data(),
                itab, jtab, ttab, utab, cabs, itab_1);

        const auto sub_asmpsw = aersol_optics_data.asmpsw[mm][isw];
        // Kokkos::subview(
        // asmpsw, mm, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(), isw);

        binterp(sub_asmpsw, refr, refi, ref_real_tab.data(), ref_img_tab.data(),
                itab, jtab, ttab, utab, casm, itab_1);

        // parameterized optical properties
        const auto cheb_k = Kokkos::subview(cheb, Kokkos::ALL(), kk);
        Real pext = zero; //    parameterized specific extinction [m2/kg]
        calc_parameterized(cext, cheb_k.data(), pext);
        Real pabs = zero; // parameterized specific absorption [m2/kg]
        calc_parameterized(cabs, cheb_k.data(), pabs);

        Real pasm = zero; // parameterized asymmetry factor [unitless?]
        calc_parameterized(casm, cheb_k.data(), pasm);

        //  do icol=1,ncol

        if (logradsurf(kk) <= xrmax) {
          pext = haero::exp(pext);
        } else {
          // BAD CONSTANT
          pext = 1.5 / (radsurf(kk) * rhoh2o); // ! geometric optics
        }                                      // if logradsurf(kk) <= xrmax

        // convert from m2/kg water to m2/kg aerosol
        // FIXME: specpext is used by check_error_warning, which is not ported
        // yet. const Real specpext = pext;// specific extinction [m2/kg]
        pext *= wetvol * rhoh2o;
        pabs *= wetvol * rhoh2o;
        pabs = mam4::utils::min_max_bound(zero, pext, pabs);
        Real palb =
            one -
            pabs / haero::max(pext,
                              small_value_40); // parameterized single
                                               // scattering albedo [unitless]
        const Real dopaer =
            pext * mass(kk); // aerosol optical depth in layer [1]
        if (savaeruv) {
          aoduv += dopaer;
        } // savaeruv
        if (savaernir) {
          aodnir += dopaer;
        } // savaernir
        // end cols

        // Save aerosol optical depth at longest visible wavelength
        // sum over layers

        if (savaervis) {
          // ! aerosol extinction (/m)
          // do icol = 1, ncol
          extinct(kk) += dopaer * air_density(kk) / mass(kk);
          absorb(kk) += pabs * air_density(kk);
          aodvis += dopaer;
          aodall += dopaer;
          aodabs += pabs * mass(kk);
          aodmode[mm] += dopaer;
          ssavis += dopaer * palb;

          if (wetvol > small_value_40) {
            dustaodmode[mm] += dopaer * dustvol / wetvol;
            // partition optical depth into contributions from each constituent
            // assume contribution is proportional to refractive index X volume

            scath2o = watervol * aersol_optics_data.crefwsw(isw).real();
            absh2o = -watervol * aersol_optics_data.crefwsw(isw).imag();
            sumscat = scatso4 + scatpom + scatsoa + scatbc + scatdust +
                      scatseasalt + scath2o + scatmom;
            sumabs = absso4 + abspom + abssoa + absbc + absdust + absseasalt +
                     absh2o + absmom;
            sumhygro = hygroso4 + hygropom + hygrosoa + hygrobc + hygrodust +
                       hygroseasalt + hygromom;

            //
            update_aod_spec(scath2o,
                            absh2o, // in
                            sumhygro, sumscat,
                            sumabs, // in
                            hygrodust, palb,
                            dopaer, // in
                            scatdust, absdust, dustaod);

            update_aod_spec(scath2o,
                            absh2o, // in
                            sumhygro, sumscat,
                            sumabs, // in
                            hygroso4, palb,
                            dopaer, // in
                            scatso4, absso4, so4aod);

            update_aod_spec(scath2o,
                            absh2o, // in
                            sumhygro, sumscat,
                            sumabs, // in
                            hygropom, palb,
                            dopaer, // in
                            scatpom, abspom, pomaod);

            update_aod_spec(scath2o,
                            absh2o, // in
                            sumhygro, sumscat,
                            sumabs, // in
                            hygrosoa, palb,
                            dopaer, // in
                            scatsoa, abssoa, soaaod);

            update_aod_spec(scath2o,
                            absh2o, // in
                            sumhygro, sumscat,
                            sumabs, // in
                            hygrobc, palb,
                            dopaer, // in
                            scatbc, absbc, bcaod);

            update_aod_spec(scath2o,
                            absh2o, // in
                            sumhygro, sumscat,
                            sumabs, // in
                            hygroseasalt, palb,
                            dopaer, // in
                            scatseasalt, absseasalt, seasaltaod);

            update_aod_spec(scath2o,
                            absh2o, // in
                            sumhygro, sumscat,
                            sumabs, // in
                            hygromom, palb,
                            dopaer, // in
                            scatmom, absmom, momaod);

            aodabsbc += absbc * dopaer * (one - palb);

          } // if wetvol(icol) > small_value_40

        } // savaervis
        // do icol = 1, ncol
        // FIXME: I did not port the following:
        //  call check_error_warning('sw', icol, kk, mm, isw, nspec,list_idx, &
        //  ! in
        //                     dopaer(icol), pabs(icol), dryvol, wetvol,
        //                     watervol, crefin,cabs,& ! in specdens,
        //                     specrefindex, specvol, & ! in nerr_dopaer, & !
        //                     inout pext(icol), specpext(icol) ) ! optional in

        tauxar(kk, isw) += dopaer;
        wa(kk, isw) += dopaer * palb;
        ga(kk, isw) += dopaer * palb * pasm;
        fa(kk, isw) += dopaer * palb * pasm * pasm;
        // enddo
      } // k
    }   // isw

  } // mm

  if (is_cmip6_volc) {
    calc_volc_ext(trop_level, state_zm, ext_cmip6_sw, extinct, tropopause_m);
  }
}

KOKKOS_INLINE_FUNCTION
void modal_aero_lw(const Real dt, const View2D &state_q,
                   const ConstColumnView &temperature,
                   const ConstColumnView &pmid, const ConstColumnView &pdel,
                   const ConstColumnView &pdeldry, const ConstColumnView &cldn,
                   // const ColumnView qqcw_fld[pcnst],
                   const View2D &tauxar,
                   // parameters
                   int nspec_amode[ntot_amode], Real sigmag_amode[ntot_amode],
                   int lmassptr_amode[maxd_aspectype][ntot_amode],
                   Real spechygro[maxd_aspectype],
                   Real specdens_amode[maxd_aspectype],
                   int lspectype_amode[maxd_aspectype][ntot_amode],
                   const AerosolOpticsDeviceData &aersol_optics_data,
                   // work views
                   const ColumnView &mass, const View2D &cheb,
                   const View2D &dgnumwet_m, const View2D &dgnumdry_m,
                   const ColumnView &radsurf, const ColumnView &logradsurf,
                   const ComplexView2D &specrefindex, const View2D &qaerwat_m) {

  //
  // calculates aerosol lw radiative properties

  // dt       ! time step [s]
  // state_q(:,:,:)   ! water and tracers (state%q) in state [kg/kg]
  // temperature(:,:) ! temperature [K]
  // pmid(:,:)        ! mid-point pressure [Pa]
  // pdel(:,:)        ! pressure interval [Pa]
  // pdeldry(:,:)     ! dry mass pressure interval [Pa]
  // cldn(:,:)        ! layer cloud fraction [fraction]

  // qqcw(:)               ! Cloud borne aerosols mixing ratios [kg/kg or 1/kg]
  // tauxar(pcols,pver,nlwbands) ! layer absorption optical depth

  constexpr Real zero = 0.0;
  // dry mass in each cell
  for (int kk = 0; kk < pver; ++kk) {
    // layer dry mass [kg/m2]
    mass(kk) = pdeldry(kk) * rga;

    // initialize output variables
    for (int i = 0; i < nlwbands; ++i) {
      tauxar(kk, i) = zero;
    }
  } // k

  const auto specrefndxlw = aersol_optics_data.specrefndxlw;
  const auto crefwlw = aersol_optics_data.crefwlw;
  const auto crefwsw = aersol_optics_data.crefwsw;

  // FORTRAN refactoring: For prognostic aerosols only, other options are
  // removed
  // const int list_idx = 0; //   index of the climate or a diagnostic list
  // We compute dgnumdry_m with the calcize process.
  // modal_aero_calcsize_sub(state_q, pdel, dt, qqcw_fld, list_idx, false,
  //                         dgnumdry_m); // ! out

  modal_aero_wateruptake_dr(state_q, temperature, pmid, cldn, dgnumdry_m,
                            dgnumwet_m, qaerwat_m, nspec_amode, specdens_amode,
                            spechygro, lspectype_amode);

  Real specvol[max_nspec] = {};
  for (int mm = 0; mm < ntot_amode; ++mm) {

    // get mode info
    const int nspec = nspec_amode[mm];
    const Real sigma_logr_aer = sigmag_amode[mm];
    for (int kk = top_lev; kk < pver; ++kk) {

      // calc size parameter for all columns
      // FORTRAN refactoring: ismethod2 is tempararily used to ensure BFB test.
      // can be removed when porting to C++
      auto cheb_kk = Kokkos::subview(cheb, Kokkos::ALL(), kk);
      modal_size_parameters(sigma_logr_aer, dgnumwet_m(kk, mm), // in
                            radsurf(kk), logradsurf(kk), cheb_kk.data(), true);
    } // kk

    // FIXME: is there a way of avoiding this copy?
    // specrefindex(ll,:) = specrefndxsw(:,lspectype_amode[ll][mm])
    for (int ilwbands = 0; ilwbands < nlwbands; ++ilwbands) {
      // // Fortran to C++ indexing
      for (int ll = 0; ll < nspec; ++ll) {
        specrefindex(ll, ilwbands) =
            specrefndxlw(ilwbands, lspectype_amode[ll][mm] - 1);
      } // ll
    }   // ilwbands

    for (int ilw = 0; ilw < nlwbands; ++ilw) {
      for (int kk = top_lev; kk < pver; ++kk) {
        for (int ll = 0; ll < nspec; ++ll) {
          // Fortran to C++ indexing
          auto specmmr = Kokkos::subview(state_q, Kokkos::ALL(),
                                         lmassptr_amode[ll][mm] - 1);
          // Fortran to C++ indexing
          const Real specdens = specdens_amode[lspectype_amode[ll][mm] - 1];

          specvol[ll] = specmmr(kk) / specdens;
          // printf("lspectype_amode[ll][mm] %d lspectype_amode[ll][mm] %d
          // specmmr(kk) %e specdens %e state_q %e \n
          // ",lspectype_amode[ll][mm],lspectype_amode[ll][mm], specmmr(kk),
          // specdens, state_q(kk,lmassptr_amode[ll][mm]-1));
        } // ll

        // lw =0 and sw =1
        // calculate complex refractive index
        Real dryvol = zero;
        Real wetvol = zero;
        Real watervol = zero;
        Kokkos::complex<Real> crefin = {};
        Real refr, refi = {};
        // printf("wetvol %e nspec %d \n ", wetvol,nspec  );
        // for (int i = 0; i < nspec; ++i)
        //   printf("specvol[%d] %e \n ", i,specvol[i]);
        // FIXME: remove
        // qaerwat_m(kk, mm)=zero;
        calc_refin_complex(0, ilw, qaerwat_m(kk, mm), specvol, specrefindex,
                           nspec, crefwlw, crefwsw, dryvol, wetvol, watervol,
                           crefin, refr, refi);

        // Kokkos::subview(
        // absplw, mm, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(), ilw);

        const auto ref_real_tab = aersol_optics_data.refrtablw[mm][ilw];
        // Kokkos::subview(refrtablw, mm, Kokkos::ALL(), ilw);
        const auto ref_img_tab = aersol_optics_data.refitablw[mm][ilw];
        // Kokkos::subview(refitablw, mm, Kokkos::ALL(), ilw);

        // interpolate coefficients linear in refractive index
        // first call calcs itab,jtab,ttab,utab
        int itab = zero;
        int itab_1 = zero;
        int jtab = zero;
        Real ttab, utab = {};
        Real cabs[ncoef] = {};
        binterp(aersol_optics_data.absplw[mm][ilw], refr, refi,
                ref_real_tab.data(), ref_img_tab.data(), itab, jtab, ttab, utab,
                cabs, itab_1);

        // parameterized optical properties
        Real pabs = zero; //    parameterized specific extinction [m2/kg]
        auto cheb_kk = Kokkos::subview(cheb, Kokkos::ALL(), kk);
        calc_parameterized(cabs, cheb_kk.data(), pabs);

        // printf("pabs %e \n", pabs);
        pabs *= wetvol * rhoh2o;
        // printf("wetvol %e rhoh2o %e \n ", wetvol, rhoh2o );
        pabs = haero::max(zero, pabs);
        Real dopaer = pabs * mass(kk);

        // update absorption optical depth
        // printf("dopaer %e pabs %e mass(kk) %e \n", dopaer, pabs, mass(kk));
        tauxar(kk, ilw) += dopaer;
        // printf("tauxar %e \n", tauxar(kk, ilw));

        // FIXME: specpext is used by check_error_warning, which is not ported
        // yet.
        // FORTRAN refactor: check and writeout error/warning message
        // call check_error_warning('lw', icol, kk,mm, ilw, nspec, list_idx,& !
        // in
        //                 dopaer, pabs(icol), dryvol, wetvol, watervol,
        //                 crefin,cabs,& ! in specdens, specrefindex, specvol, &
        //                 ! in nerr_dopaer) ! inout
      } // kk
    }   // ilw

  } // mm

} // modal_aero_lw

} // namespace modal_aer_opt

} // end namespace mam4

#endif
