#ifndef MAM4XX_AER_RAD_PROPS_HPP
#define MAM4XX_AER_RAD_PROPS_HPP

#include <haero/math.hpp>
#include <mam4xx/aero_config.hpp>

#include <mam4xx/modal_aer_opt.hpp>

namespace mam4 {

namespace aer_rad_props {

using ConstColumnView = haero::ConstColumnView;
// From radconstants
constexpr int nswbands = modal_aer_opt::nswbands;
constexpr int nlwbands = modal_aer_opt::nlwbands;
using View2D = DeviceType::view_2d<Real>;
using namespace mam4::modal_aer_opt;
constexpr Real km_inv_to_m_inv = 0.001; // 1/km to 1/m

constexpr Real shr_const_rgas =
    haero::Constants::r_gas * 1e3; // Universal gas constant ~ J/K/kmole
constexpr Real shr_const_mwdair = haero::Constants::molec_weight_dry_air *
                                  1e3; // molecular weight dry air ~ kg/kmole
constexpr Real shr_const_cpdair =
    haero::Constants::cp_dry_air; // specific heat of dry air   ~ J/kg/K

constexpr Real cnst_kap =
    (shr_const_rgas / shr_const_mwdair) / shr_const_cpdair; //  ! R/Cp

constexpr Real cnst_faktor =
    -haero::Constants::gravity /
    haero::Constants::r_gas_dry_air; // acceleration of gravity ~ m/s^2/Dry air
                                     // gas constant     ~ J/K/kg
constexpr Real cnst_ka1 = cnst_kap - 1.0;
KOKKOS_INLINE_FUNCTION
void volcanic_cmip_sw(const ConstColumnView &zi, const int ilev_tropp,
                      const View2D &ext_cmip6_sw_inv_m,
                      const View2D &ssa_cmip6_sw, const View2D &af_cmip6_sw,
                      const View2D &tau, const View2D &tau_w,
                      const View2D &tau_w_g, const View2D &tau_w_f) {

  // !Intent-in
  // ncol       ! Number of columns
  // zi(:,:)    ! Height above surface at interfaces [m]
  // trop_level(pcols)  ! tropopause level index
  // ext_cmip6_sw_inv_m(pcols,pver,nswbands)  ! short wave extinction [m^{-1}]
  // ssa_cmip6_sw(:,:,:),af_cmip6_sw(:,:,:)

  // !Intent-inout
  // tau    (pcols,0:pver,nswbands) ! aerosol extinction optical depth
  // tau_w  (pcols,0:pver,nswbands) ! aerosol single scattering albedo * tau
  // tau_w_g(pcols,0:pver,nswbands) ! aerosol assymetry parameter * tau * w
  // tau_w_f(pcols,0:pver,nswbands) ! aerosol forward scattered fraction * tau *
  // w

  // !Local variables
  // icol, ipver, ilev_tropp
  // lyr_thk ! thickness between level interfaces [m]
  // ext_unitless(nswbands), asym_unitless(nswbands)
  // ext_ssa(nswbands),ext_ssa_asym(nswbands)

  // !Logic:
  // !Update taus, tau_w, tau_w_g and tau_w_f with the read in volcanic
  // !aerosol extinction (1/km), single scattering albedo and asymmtry factors.

  // !Above the tropopause, the read in values from the file include both the
  // stratospheric !and volcanic aerosols. Therefore, we need to zero out taus
  // above the tropopause !and populate them exclusively from the read in
  // values.

  // !If tropopause is found, update taus with 50% contributuions from the
  // volcanic input !file and 50% from the existing model computed values

  // !First handle the case of tropopause layer itself:
  // do icol = 1, ncol
  // ilev_tropp = trop_level(icol) !tropopause level
  //
  constexpr Real half = 0.5;

  const Real lyr_thk = zi(ilev_tropp) - zi(ilev_tropp + 1);
  for (int i = 0; i < nswbands; ++i) {
    const Real ext_unitless = lyr_thk * ext_cmip6_sw_inv_m(ilev_tropp, i);
    const Real asym_unitless = af_cmip6_sw(ilev_tropp, i);
    const Real ext_ssa = ext_unitless * ssa_cmip6_sw(ilev_tropp, i);
    const Real ext_ssa_asym = ext_ssa * asym_unitless;

    tau(ilev_tropp, i) = half * (tau(ilev_tropp, i) + ext_unitless);
    tau_w(ilev_tropp, i) = half * (tau_w(ilev_tropp, i) + ext_ssa);
    tau_w_g(ilev_tropp, i) = half * (tau_w_g(ilev_tropp, i) + ext_ssa_asym);
    tau_w_f(ilev_tropp, i) =
        half * (tau_w_f(ilev_tropp, i) + ext_ssa_asym * asym_unitless);
  } // end i

  // !As it will be more efficient for FORTRAN to loop over levels and then
  // columns, the following loops !are nested keeping that in mind Note that in
  // C++ ported code, the loop over levels is nested. Thus, the previous comment
  // does not apply.

  // ilev_tropp = trop_level(icol) !tropopause level
  for (int kk = 0; kk < ilev_tropp; ++kk) {

    const Real lyr_thk = zi(kk) - zi(kk + 1);
    for (int i = 0; i < nswbands; ++i) {
      const Real ext_unitless = lyr_thk * ext_cmip6_sw_inv_m(kk, i);
      const Real asym_unitless = af_cmip6_sw(kk, i);
      const Real ext_ssa = ext_unitless * ssa_cmip6_sw(kk, i);
      const Real ext_ssa_asym = ext_ssa * asym_unitless;
      tau(kk, i) = ext_unitless;
      tau_w(kk, i) = ext_ssa;
      tau_w_g(kk, i) = ext_ssa_asym;
      tau_w_f(kk, i) = ext_ssa_asym * asym_unitless;

    } // end nswbands

  } // kk

} // volcanic_cmip_sw
// FIXME; to move compute_odap_volcanic_at_troplayer_lw to a new file,
// aer_rad_props
KOKKOS_INLINE_FUNCTION
void compute_odap_volcanic_at_troplayer_lw(const int ilev_tropp,
                                           const ConstColumnView &zi,
                                           const View2D &ext_cmip6_lw_inv_m,
                                           const View2D &odap_aer) {
  // Update odap_aer with a combination read in volcanic aerosol extinction
  // [1/m] (50%) and module computed values (50%).

  // intent-ins
  //  ncol
  // trop_level(:)
  // zi(:,:) !geopotential height above surface at interfaces [m]
  // ext_cmip6_lw_inv_m(:,:,:) !long wave extinction in the units of [1/m]

  //! intent-inouts
  //  odap_aer(:,:,:)  ! [fraction] absorption optical depth, per layer
  //  [unitless]

  // !local
  // integer :: icol, ilev_tropp
  // real(r8) :: lyr_thk !layer thickness [m]
  // do icol = 1, ncol
  // ilev_tropp = trop_level(icol) !tropopause level
  const Real lyr_thk =
      zi(ilev_tropp) - zi(ilev_tropp + 1); //! compute layer thickness in meters
  constexpr Real half = 0.5;
  //! update taus with 50% contributuions from the volcanic input file
  //! and 50% from the existing model computed values at the tropopause layer
  for (int i = 0; i < nlwbands; ++i) {
    odap_aer(ilev_tropp, i) =
        half * (odap_aer(ilev_tropp, i) +
                (lyr_thk * ext_cmip6_lw_inv_m(ilev_tropp, i)));
  }

} // compute_odap_volcanic_at_troplayer_lw

KOKKOS_INLINE_FUNCTION
void compute_odap_volcanic_above_troplayer_lw(const int ilev_tropp,
                                              const ConstColumnView &zi,
                                              const View2D &ext_cmip6_lw_inv_m,
                                              const View2D &odap_aer) {

  //     !Above the tropopause, the read in values from the file include both
  //     the stratospheric
  // !and volcanic aerosols. Therefore, we need to zero out odap_aer above the
  // tropopause !and populate it exclusively from the read in values.

  // !intent-ins
  // integer, intent(in) :: pver, ncol
  // integer, intent(in) :: trop_level(:)

  // real(r8), intent(in) :: zi(:,:) !geopotential height above surface at
  // interfaces [m] real(r8), intent(in) :: ext_cmip6_lw_inv_m(:,:,:) !long wave
  // extinction in the units of [1/m]

  // !intent-inouts
  // real(r8), intent(inout) :: odap_aer(:,:,:) ! [fraction] absorption optical
  // depth, per layer [unitless]

  // !local
  // integer :: ipver, icol, ilev_tropp
  // real(r8) :: lyr_thk !layer thickness [m]

  // !As it will be more efficient for FORTRAN to loop over levels and then
  // columns, the following loops !are nested keeping that in mind

  for (int kk = 0; kk < ilev_tropp; ++kk) {
    const Real lyr_thk =
        zi(kk) - zi(kk + 1); // ! compute layer thickness in meters
    for (int i = 0; i < nlwbands; ++i) {
      odap_aer(kk, i) = lyr_thk * ext_cmip6_lw_inv_m(kk, i);
    } // end i

  } // end kk

} // compute_odap_volcanic_above_troplayer_lw
KOKKOS_INLINE_FUNCTION
void get_dtdz(const Real pm, const Real pmk, const Real pmid1d_up,
              const Real pmid1d_down, const Real temp1d_up,
              const Real temp1d_down, Real &dtdz, Real &tm) {

  // pm      ! mean pressure [Pa]
  // pmk
  // pmid1d_up     !  midpoint pressure in column at upper level [Pa]
  // pmid1d_down   !  midpoint pressure in column at lower level [Pa]
  // temp1d_up     !  temperature in column at upper level [K]
  // temp1d_down   !  temperature in column at lower level [K]
  // dtdz     ! temperature lapse rate vs. height [K/m]
  // tm       ! mean temperature [K] -- needed to find pressure at trop + 2 km

  const Real a1 =
      (temp1d_up - temp1d_down) /
      (haero::pow(pmid1d_up, cnst_kap) - haero::pow(pmid1d_down, cnst_kap));
  const Real b1 = temp1d_down - a1 * haero::pow(pmid1d_down, cnst_kap);
  tm = a1 * pmk + b1;
  const Real dtdp = a1 * cnst_kap * (haero::pow(pm, cnst_ka1));
  dtdz = cnst_faktor * dtdp * pm / tm;

} // get_dtdz
// FIXME: move to tropopause.hpp
/* This routine is an implementation of Reichler et al. [2003] done by
! Reichler and downloaded from his web site. Minimal modifications were
! made to have the routine work within the CAM framework (i.e. using
! CAM constants and types).
!
! NOTE: I am not a big fan of the goto's and multiple returns in this
! code, but for the moment I have left them to preserve as much of the
! original and presumably well tested code as possible.
! UPDATE: The most "obvious" substitutions have been made to replace
! goto/return statements with cycle/exit. The structure is still
! somewhat tangled.
! UPDATE 2: "gamma" renamed to "gam" in order to avoid confusion
! with the Fortran 2008 intrinsic. "level" argument removed because
! a physics column is not contiguous, so using explicit dimensions
! will cause the data to be needlessly copied.
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
! determination of tropopause height from gridded temperature data
!
! reference: Reichler, T., M. Dameris, and R. Sausen (2003)
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/
KOKKOS_INLINE_FUNCTION
void twmo(const ConstColumnView &temp1d, const ConstColumnView &pmid1d,
          const Real plimu, const Real pliml, const Real gam, Real &trp) {

  // temp1d   !  temperature in column [K]
  // pmid1d   !  midpoint pressure in column [Pa]
  // plimu    ! upper limit of tropopause pressure [Pa]
  // pliml    ! lower limit of tropopause pressure [Pa]
  // gam      ! lapse rate to indicate tropopause [K/m]
  // trp      ! tropopause pressure [Pa]
  // BAD CONSTANT
  constexpr Real deltaz = 2000.0; //   ! [m]
  constexpr Real zero = 0.0;
  constexpr Real one = 1.0;
  constexpr Real half = 0.5;

  trp = -99.0; //                           ! negative means not valid

  // initialize start level
  // dt/dz
  const Real pmk = half * (haero::pow(pmid1d(pver - 2), cnst_kap) +
                           haero::pow(pmid1d(pver - 1), cnst_kap));
  const Real pm = haero::pow(pmk, one / cnst_kap);
  // temperature lapse rate vs. height [K/m]
  Real dtdz = zero;
  // mean temperature [K]
  Real tm = zero;

  get_dtdz(pm, pmk, pmid1d(pver - 2), pmid1d(pver - 1), temp1d(pver - 2),
           temp1d(pver - 1), dtdz, tm);
  for (int kk = pver - 2; kk < 1; --kk) {
    // const Real pm0 = pm;
    const Real pmk0 = pmk;
    const Real dtdz0 = dtdz;

    // dt/dz
    const Real pmk = half * (haero::pow(pmid1d(kk - 1), cnst_kap) +
                             haero::pow(pmid1d(kk), cnst_kap));
    const Real pm = haero::pow(pmk, one / cnst_kap);

    get_dtdz(pm, pmk, pmid1d(kk - 1), pmid1d(kk), temp1d(kk - 1), temp1d(kk),
             dtdz, tm);

    // dt/dz valid?
    if (dtdz <= gam) {
      // nothing here go to next iteration
      // no, dt/dz < -2 K/km
    } else if (pm > plimu) {
      // nothing here go to next iteration
      // no, too low
    } else {

      //  dtdz is valid, calculate tropopause pressure
      Real ag = zero;
      Real bg = zero;
      Real ptph = zero;
      if (dtdz0 < gam) {
        ag = (dtdz - dtdz0) / (pmk - pmk0);
        bg = dtdz0 - (ag * pmk0);
        ptph = haero::exp(haero::log((gam - bg) / ag) / cnst_kap);
      } else {
        ptph = pm;
      } // if dtdz0<gam

      if (ptph < pliml) {
        // cycle main_loop
      } else if (ptph > plimu) {
        // cycle main_loop
      } else {

        //  2nd test: dtdz above 2 km must not exceed gam
        const Real p2km =
            ptph + deltaz * (pm / tm) * cnst_faktor; //     p at ptph + 2km
        Real asum = zero; //                                dtdz above
        int icount =
            0; //                                  number of levels above

        // test until apm < p2km
        for (int jj = kk; jj < 1; --jj) {
          const Real pmk2 =
              half * (haero::pow(pmid1d(jj - 1), cnst_kap) +
                      haero::pow(pmid1d(jj), cnst_kap)); // ! p mean ^kappa
          const Real pm2 = haero::pow(
              pmk2, one / cnst_kap); //                           ! p mean
          if (pm2 > ptph) {
            // cycle in_loop      doesn't happen
          } else if (pm2 < p2km) {
            // exit in_loop      ptropo is valid
            break;
          } else {
            Real dtdz2 = zero;
            Real tm2 = zero;
            get_dtdz(pm2, pmk2, pmid1d(jj - 1), pmid1d(jj), temp1d(jj - 1),
                     temp1d(jj), dtdz2, tm2);
            asum += dtdz2;
            icount++;
            const Real aquer =
                asum / Real(icount); //               ! dt/dz mean
            // ! discard ptropo ?

            if (aquer <= gam) {
              // cycle main_loop      ! dt/dz above < gam
              jj = 1;
            }

          } // pm2>ptph

        } // jj test next level
        trp = ptph;
        break;
      } // ptph<pliml
    }   // dtdz<=gam

  } // kk

} // twmo
// FIXME: move to tropopause.hpp
// This routine uses an implementation of Reichler et al. [2003] done by
// Reichler and downloaded from his web site. This is similar to the WMO
//  routines, but is designed for GCMs with a coarse vertical grid.
KOKKOS_INLINE_FUNCTION
void tropopause_twmo(const ConstColumnView &pmid, const ConstColumnView &pint,
                     const ConstColumnView &temp, const ConstColumnView &zm,
                     const ConstColumnView &zi, int &tropLev) {
  // BAD CONSTANT
  constexpr Real gam =
      -0.002; //       ! lapse rate to indicate tropopause [K/m]
  constexpr Real plimu =
      45000; //       ! upper limit of tropopause pressure [Pa]
  constexpr Real pliml =
      7500; //        !lower limit of tropopause pressure [Pa]

  // Use the routine from Reichler.
  Real tP = 0;
  twmo(temp, pmid, plimu, pliml, gam, tP);

  // if successful, store of the results and find the level and temperature.
  if (tP > 0) {

    // Find the associated level.
    for (int kk = pver - 1; kk < 1; --kk) {
      if (tP >= pint(kk)) {
        tropLev = kk;
        // tropLevValp1 = kk + 1;
        // tropLevValm1 = kk - 1;
        break;
      }
    } // kk

    // tropLev = tropLevVal
  } // tP

} // tropopause_twmo

/* Read the tropopause pressure in from a file containging a climatology. The
  ! data is interpolated to the current dat of year and latitude.
  !
  ! NOTE: The data is read in during tropopause_init and stored in the module
  ! variable trop */

#if 0
  KOKKOS_INLINE_FUNCTION
void tropopause_climate(lchnk,ncol,pmid,pint,temp,zm,zi,    &  ! in
             tropLev,tropP,tropT,tropZ)   {
// TO be ported...
} // tropopause_climate
#endif
KOKKOS_INLINE_FUNCTION
int tropopause_or_quit(const ConstColumnView &pmid, const ConstColumnView &pint,
                       const ConstColumnView &temperature,
                       const ConstColumnView &zm, const ConstColumnView &zi) {
  // Find tropopause or quit the simulation if not found

  // lchnk            ! number of chunks
  // ncol             ! number of columns
  // pmid(:,:)        ! midpoint pressure [Pa]
  // pint(:,:)        ! interface pressure [Pa]
  // temperature(:,:) ! temperature [K]
  // zm(:,:)          ! geopotential height above surface at midpoints [m]
  // zi(:,:)          ! geopotential height above surface at interfaces [m]
  // !return value [out]
  // trop_level(pcols) !return value

  // !trop_level has a value for tropopause for each column
  // call tropopause_find(lchnk, ncol, pmid, pint, temperature, zm, zi, & !in
  //        trop_level) !out
  int trop_level = 0;
  tropopause_twmo(pmid, pint, temperature, zm, zi, trop_level);

  if (trop_level < -1) {
    Kokkos::abort("aer_rad_props: tropopause not found\n");
  }

  // Need to ported default_backup, i.e., tropopause_climate

  return trop_level;
} // tropopause_or_quit

//
KOKKOS_INLINE_FUNCTION
void aer_rad_props_lw(
    const Real dt, const ConstColumnView &pmid, const ConstColumnView &pint,
    const ConstColumnView &temperature, const ConstColumnView &zm,
    const ConstColumnView &zi, const View2D &state_q,
    const ConstColumnView &pdel, const ConstColumnView &pdeldry,
    const ConstColumnView &cldn, const View2D &ext_cmip6_lw,
    // const ColumnView qqcw_fld[pcnst],
    const View2D &odap_aer,
    //
    int nspec_amode[ntot_amode], Real sigmag_amode[ntot_amode],
    int lmassptr_amode[maxd_aspectype][ntot_amode],
    Real spechygro[maxd_aspectype], Real specdens_amode[maxd_aspectype],
    int lspectype_amode[maxd_aspectype][ntot_amode],
    const ComplexView2D &specrefndxlw,
    const Kokkos::complex<Real> crefwlw[nlwbands],
    const Kokkos::complex<Real> crefwsw[nswbands],
    const View3D absplw[ntot_amode][nlwbands],
    const View1D refrtablw[ntot_amode][nlwbands],
    const View1D refitablw[ntot_amode][nlwbands],
    // work views
    const ColumnView &mass, const View2D &cheb, const View2D &dgnumwet_m,
    const View2D &dgnumdry_m, const ColumnView &radsurf,
    const ColumnView &logradsurf, const ComplexView2D &specrefindex,
    const View2D &qaerwat_m, const View2D &ext_cmip6_lw_inv_m

) {

  // Purpose: Compute aerosol transmissions needed in absorptivity/
  // emissivity calculations

  // Intent-ins
  //  is_cmip6_volc    ! flag for using cmip6 style volc emissions
  //  dt               ! time step[s]
  //  lchnk            ! number of chunks
  //  ncol             ! number of columns
  //  pmid(:,:)        ! midpoint pressure [Pa]
  //  pint(:,:)        ! interface pressure [Pa]
  //  temperature(:,:) ! temperature [K]
  //  zm(:,:)          ! geopotential height above surface at midpoints [m]
  //  zi(:,:)          ! geopotential height above surface at interfaces [m]
  //  state_q(:,:,:)
  //  pdel(:,:)
  //  pdeldry(:,:)
  //  cldn(:,:)
  //  ext_cmip6_lw(:,:,:) !long wave extinction in the units of [1/km]
  //  qqcw(:)   ! Cloud borne aerosols mixing ratios [kg/kg or 1/kg]

  // intent-outs
  //  odap_aer(pcols,pver,nlwbands) ! [fraction] absorption optical depth, per
  //  layer [unitless]
  // Compute contributions from the modal aerosols.
  modal_aero_lw(dt, state_q, temperature, pmid, pdel, pdeldry, cldn,
                // qqcw_fld,
                odap_aer,
                // parameters
                nspec_amode, sigmag_amode, lmassptr_amode, spechygro,
                specdens_amode, lspectype_amode, specrefndxlw, crefwlw, crefwsw,
                absplw, refrtablw, refitablw,
                // work views
                mass, cheb, dgnumwet_m, dgnumdry_m, radsurf, logradsurf,
                specrefindex, qaerwat_m);

  // !write out ext from the volcanic input file
  // call outfld('extinct_lw_inp',ext_cmip6_lw(:,:,idx_lw_diag), pcols, lchnk)
  // !convert from 1/km to 1/m
  for (int kk = 0; kk < pver; ++kk) {
    for (int i = 0; i < nlwbands; ++i) {
      ext_cmip6_lw_inv_m(kk, i) = ext_cmip6_lw(kk, i) * km_inv_to_m_inv;
    } /// end i
  }   // end kk

  // FIXME: port tropopause_or_quit
  // !Find tropopause or quit simulation if not found
  // trop_level(1:pcols) = tropopause_or_quit(lchnk, ncol, pmid, pint,
  // temperature, zm, zi)
  const int ilev_tropp = tropopause_or_quit(pmid, pint, temperature, zm, zi);

  // We are here because tropopause is found, update taus with 50%
  // contributuions from the volcanic input file and 50% from the existing model
  // computed values at the tropopause layer
  compute_odap_volcanic_at_troplayer_lw(ilev_tropp, zi, ext_cmip6_lw_inv_m,
                                        odap_aer);
  // Above the tropopause, the read in values from the file include both the
  // stratospheric
  //  and volcanic aerosols. Therefore, we need to zero out odap_aer above the
  //  tropopause and populate it exclusively from the read in values.
  compute_odap_volcanic_above_troplayer_lw(ilev_tropp, zi, ext_cmip6_lw_inv_m,
                                           odap_aer);
  // call outfld('extinct_lw_bnd7',odap_aer(:,:,idx_lw_diag), pcols, lchnk)

} // aer_rad_props_lw
KOKKOS_INLINE_FUNCTION
void aer_rad_props_sw(
    const Real dt, const ConstColumnView &zi, const ConstColumnView &pmid,
    const ConstColumnView &pint, const ConstColumnView &temperature,
    const ConstColumnView &zm, const View2D &state_q,
    const ConstColumnView &pdel, const ConstColumnView &pdeldry,
    const ConstColumnView &cldn, const View2D &ssa_cmip6_sw,
    const View2D &af_cmip6_sw, const View2D &ext_cmip6_sw,
    // nnite, idxnite,
    // is_cmip6_volc,
    // const ColumnView qqcw_fld[pcnst],
    const View2D &tau, const View2D &tau_w, const View2D &tau_w_g,
    const View2D &tau_w_f, int nspec_amode[ntot_amode],
    Real sigmag_amode[ntot_amode],
    int lmassptr_amode[maxd_aspectype][ntot_amode],
    Real spechygro[maxd_aspectype], Real specdens_amode[maxd_aspectype],
    int lspectype_amode[maxd_aspectype][ntot_amode],
    const ComplexView2D
        &specrefndxsw, // specrefndxsw( nswbands, maxd_aspectype )
    const Kokkos::complex<Real> crefwlw[nlwbands],
    const Kokkos::complex<Real> crefwsw[nswbands],
    // FIXME
    const mam4::AeroId specname_amode[9], const View3D extpsw[ntot_amode][nswbands],
    const View3D abspsw[ntot_amode][nswbands],
    const View3D asmpsw[ntot_amode][nswbands],
    const View1D refrtabsw[ntot_amode][nswbands],
    const View1D refitabsw[ntot_amode][nswbands],
    // diagnostic
    const ColumnView &extinct, //        ! aerosol extinction [1/m]
    const ColumnView &absorb,  //         ! aerosol absorption [1/m]
    Real &aodnir, Real &aoduv, Real dustaodmode[ntot_amode],
    Real aodmode[ntot_amode], Real burdenmode[ntot_amode], Real &aodabsbc,
    Real &aodvis, Real &aodall, Real &ssavis, Real &aodabs, Real &burdendust,
    Real &burdenso4, Real &burdenbc, Real &burdenpom, Real &burdensoa,
    Real &burdenseasalt, Real &burdenmom, Real &momaod, Real &dustaod,
    Real &so4aod, // total species AOD
    Real &pomaod, Real &soaaod, Real &bcaod, Real &seasaltaod,
    // work views
    const ColumnView &mass, const ColumnView &air_density, const View2D &cheb,
    const View2D &dgnumwet_m, const View2D &dgnumdry_m,
    const ColumnView &radsurf, const ColumnView &logradsurf,
    const ComplexView2D &specrefindex, const View2D &qaerwat_m,
    const View2D &ext_cmip6_sw_inv_m) {

  // call outfld('extinct_sw_inp',ext_cmip6_sw(:,:,idx_sw_diag), pcols, lchnk)

  // Return bulk layer tau, omega, g, f for all spectral intervals.

  // Arguments
  // pmid(:,:)        ! midpoint pressure [Pa]
  // pint(:,:)        ! interface pressure [Pa]
  // temperature(:,:) ! temperature [K]
  // zm(:,:)          ! geopotential height above surface at midpoints [m]
  // zi(:,:)          ! geopotential height above surface at interfaces [m]
  //  state_q(:,:,:)
  // pdel(:,:)
  // pdeldry(:,:)
  // cldn(:,:)
  // ext_cmip6_sw(:,:,:)
  // ssa_cmip6_sw(:,:,:)
  // af_cmip6_sw(:,:,:)

  // nnite                ! number of night columns
  // idxnite(:)           ! local column indices of night columns
  // is_cmip6_volc        ! true if cmip6 style volcanic file is read otherwise
  // false
  // dt                   ! time step (s)

  // qqcw(:)               ! Cloud borne aerosols mixing ratios [kg/kg or 1/kg]
  // tau    (pcols,0:pver,nswbands) ! aerosol extinction optical depth
  // tau_w  (pcols,0:pver,nswbands) ! aerosol single scattering albedo * tau
  // tau_w_g(pcols,0:pver,nswbands) ! aerosol assymetry parameter * tau * w
  // tau_w_f(pcols,0:pver,nswbands) ! aerosol forward scattered fraction * tau *
  // w

  // FORTRAN REFACTOR: This is done to fill invalid values in columns where
  // pcols>ncol C++ port can ignore this as C++ model is a single column model
  //  initialize to conditions that would cause failure
  //  tau     (:,:,:) = -100._r8
  //  tau_w   (:,:,:) = -100._r8
  //  tau_w_g (:,:,:) = -100._r8
  //  tau_w_f (:,:,:) = -100._r8

  // ! top layer (ilev = 0) has no aerosol (ie tau = 0)
  // ! also initialize rest of layers to accumulate od's
  // tau    (1:ncol,:,:) = 0._r8
  // tau_w  (1:ncol,:,:) = 0._r8
  // tau_w_g(1:ncol,:,:) = 0._r8
  // tau_w_f(1:ncol,:,:) = 0._r8

  // !Converting it from 1/km to 1/m
  constexpr int idx_sw_diag = 10; // index to sw visible band

  for (int kk = 0; kk < pver; ++kk) {
    for (int i = 0; i < nswbands; ++i) {
      ext_cmip6_sw_inv_m(kk, i) = ext_cmip6_sw(kk, i) * km_inv_to_m_inv;
    } /// end i
  }   // end kk

  // Find tropopause (or quit simulation if not found) as extinction should be
  // applied only above tropopause
  const int ilev_tropp = tropopause_or_quit(pmid, pint, temperature, zm, zi);

  auto ext_cmip6_sw_inv_m_idx_sw_diag =
      Kokkos::subview(ext_cmip6_sw_inv_m, Kokkos::ALL(), idx_sw_diag);

  // Special treatment for CMIP6 volcanic aerosols, where extinction, ssa
  // and af are directly read from the prescribed volcanic aerosol file
  modal_aero_sw(dt, state_q, zm, temperature, pmid, pdel, pdeldry, cldn,
                // const int nnite,
                // idxnite,
                true, ext_cmip6_sw_inv_m_idx_sw_diag, ilev_tropp,
                // qqcw_fld,
                tau, tau_w, tau_w_g, tau_w_f,
                //
                nspec_amode, sigmag_amode, lmassptr_amode, spechygro,
                specdens_amode, lspectype_amode,
                specrefndxsw, // specrefndxsw( nswbands, maxd_aspectype )
                crefwlw, crefwsw,
                // FIXME
                specname_amode, extpsw, abspsw, asmpsw, refrtabsw, refitabsw,
                // diagnostic
                extinct, //        ! aerosol extinction [1/m]
                absorb,  //         ! aerosol absorption [1/m]
                aodnir, aoduv, dustaodmode, aodmode, burdenmode, aodabsbc,
                aodvis, aodall, ssavis, aodabs, burdendust, burdenso4, burdenbc,
                burdenpom, burdensoa, burdenseasalt, burdenmom, momaod, dustaod,
                so4aod, // total species AOD
                pomaod, soaaod, bcaod, seasaltaod,
                // work views
                mass, air_density, cheb, dgnumwet_m, dgnumdry_m, radsurf,
                logradsurf, specrefindex, qaerwat_m);

  // Update tau, tau_w, tau_w_g, and tau_w_f with the read in values of
  // extinction, ssa and asymmetry factors
  volcanic_cmip_sw(zi, ilev_tropp, ext_cmip6_sw_inv_m, ssa_cmip6_sw,
                   af_cmip6_sw, tau, tau_w, tau_w_g, tau_w_f);

  //  Diagnostic output of total aerosol optical properties
  //  currently implemented for climate list only
  // FIXME: to be ported
  // call aer_vis_diag_out(lchnk, ncol, nnite, idxnite, tau(:,:,idx_sw_diag))
  // !in

} // aer_rad_props_sw
} // namespace aer_rad_props
} // end namespace mam4

#endif