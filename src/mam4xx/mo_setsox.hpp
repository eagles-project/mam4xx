#ifndef MAM4XX_MO_SETSOX_HPP
#define MAM4XX_MO_SETSOX_HPP

#include <ekat/util/ekat_math_utils.hpp>

#include <haero/atmosphere.hpp>
#include <haero/math.hpp>

#include <mam4xx/aero_config.hpp>
#include <mam4xx/conversions.hpp>
#include <mam4xx/mam4_types.hpp>
#include <mam4xx/utils.hpp>
#include <mam4xx/wv_sat_methods.hpp>

namespace mam4 {

namespace mo_setsox {

  // number of vertical levels
  constexpr int pver = mam4::nlev;
  constexpr int nmodes = AeroConfig::num_modes();
  constexpr int nspec_gas = AeroConfig::num_gas_phase_species();

  // these are the entries/indices in the gas-phase species array
  // from: eam/src/chemistry/pp_linoz_mam4_resus_mom_soag/mo_sim_dat.F90#L23)
  // solsym(: 31) = (/ 'O3', 'H2O2', 'H2SO4', 'SO2', 'DMS',  'SOAG', 'so4_a1',
  //                   'pom_a1', 'soa_a1', 'bc_a1',  'dst_a1', 'ncl_a1',
  //                   'mom_a1', 'num_a1', 'so4_a2',  'soa_a2', 'ncl_a2',
  //                   'mom_a2', 'm_a2', 'dst_a3',  'ncl_a3', 'so4_a3', 'bc_a3',
  //                   'pom_a3', 'soa_a3', 'mom_a3', 'num_a3', 'pom_a4',
  //                   'bc_a4', 'mom_a4', 'num_a4' /)
  // Fortran-indexed:
  // 1: 'O3', 2: 'H2O2', 3: 'H2SO4', 4: 'SO2', 5: 'DMS', 6: 'SOAG', 7: 'so4_a1',
  // 8: 'pom_a1', 9: 'soa_a1', 10: 'bc_a1', 11: 'dst_a1', 12: 'ncl_a1', 13: 'mom_a1',
  // 14: 'num_a1', 15: 'so4_a2', 16: 'soa_a2', 17: 'ncl_a2', 18: 'mom_a2',
  // 19: 'num_a2', 20: 'dst_a3', 21: 'ncl_a3', 22: 'so4_a3', 23: 'bc_a3',
  // 24: 'pom_a3', 25: 'soa_a3', 26: 'mom_a3', 27: 'num_a3', 28: 'pom_a4',
  // 29: 'bc_a4', 30: 'mom_a4', 31: 'num_a4'
  // c++ indexed:
  // 0: 'O3', 1: 'H2O2', 2: 'H2SO4', 3: 'SO2', 4: 'DMS', 5: 'SOAG', 6: 'so4_a1',
  // 7: 'pom_a1', 8: 'soa_a1', 9: 'bc_a1', 10: 'dst_a1', 11: 'ncl_a1',
  // 12: 'mom_a1', 13: 'num_a1', 14: 'so4_a2', 15: 'soa_a2', 16: 'ncl_a2',
  // 17: 'mom_a2', 18: 'num_a2', 19: 'dst_a3', 20: 'ncl_a3', 21: 'so4_a3',
  // 22: 'bc_a3', 23: 'pom_a3', 24: 'soa_a3', 25: 'mom_a3', 26: 'num_a3',
  // 27: 'pom_a4', 28: 'bc_a4', 29: 'mom_a4', 30: 'num_a4'
  constexpr int id_so2 = 3;
  constexpr int id_h2o2 = 1;
  constexpr int id_o3 = 0;
  constexpr int id_h2so4 = 2;

  // Real xso4_init =
  // Real xdelso4hp =

  // ===========================================================================
  //     BAD CONSTANTS (let's just put the global ones in one place for now)
  // ===========================================================================
  constexpr Real lwc = 0.2302286341e-4;
  constexpr int lptr_so4_cw_amode[4] = {15, 23, 30, -1};
  constexpr int loffset = 9;
  constexpr Real small_value_lwc = 1.0e-8;
  constexpr Real small_value_cf = 1.0e-5;
  constexpr Real p0 = 101300.0;
  constexpr bool cloud_borne = false;
  constexpr bool modal_aerosols = false;
  // universal gas constant (sort of)
  // FIXME: TERRIBLE CONSTANT
  constexpr Real Ra = 8314.0 / 101325.0;
  // water acidity
  constexpr Real xkw = 1.0e-14;
  // FIXME: AWFUL CONSTANT
  // [cm3/L / Avogadro constant]
  constexpr Real const0 = 1.0e3 / 6.023e23;
  // 330 ppm = 330.0e-6 atm
  constexpr Real co2g = 330.0e-6;
  constexpr int itermax = 20;
  constexpr Real small_value_20 = 1.0e-20;
  constexpr Real small_value_30 = 1.0e-30;
  // ===========================================================================

  // struct for the sox_cldaero_create_obj()
  // NOTE: maybe ditch this in the future to make life simpler?
  struct Cloudconc {
   Real so4c;
   Real xlwc;
   Real so4_fact;
  };
  // use shr_kind_mod, only : r8 => shr_kind0
  // use cam_logfile,  only : iulog

  // private
  // public :: sox_inti, setsox
  // public :: has_sox

  // save
  // logical            ::  inv_o3
  // integer            ::  id_msa

  // integer :: id_so2, id_nh3, id_hno3, id_h2o2, id_o3, id_ho2
  // integer :: id_so4, id_h2so4

  // logical :: has_sox = .true.
  // logical :: inv_so2, inv_nh3, inv_hno3, inv_h2o2, inv_ox, inv_nh4no3, inv_ho2

  // logical :: cloud_borne = .false.
  // logical :: modal_aerosols = .false.
KOKKOS_INLINE_FUNCTION
void sox_init(Real a) {}
KOKKOS_INLINE_FUNCTION
void setsox() {}



KOKKOS_INLINE_FUNCTION
void sox_init(AeroConfig aero_config) {
    //-----------------------------------------------------------------------
    // ... initialize the hetero sox routine
    //-----------------------------------------------------------------------

    // use mo_chem_utls, only : get_spc_ndx, get_inv_ndx
    // use spmd_utils,   only : masterproc
    // use cam_history,  only : addfld
    // use cam_history,  only : add_default
    // use ppgrid,       only : pver
    // use phys_control, only : phys_getopts
    // use sox_cldaero_mod, only : sox_cldaero_init

    // implicit none

    // logical :: history_aerosol   // Output aerosol diagnostics
    // logical :: history_verbose   // produce verbose history output

    // call phys_getopts( &
    //      history_aerosol_out = history_aerosol, &
    //      history_verbose_out = history_verbose, &
    //      prog_modal_aero_out=modal_aerosols )

    // cloud_borne = modal_aerosols

    // //-----------------------------------------------------------------
    // //       ... get species indices
    // //-----------------------------------------------------------------

    // if (cloud_borne) then
    //    id_h2so4 = get_spc_ndx( 'H2SO4' )
    // else
    //    id_so4 = get_spc_ndx( 'SO4' )
    // endif
    // id_msa = get_spc_ndx( 'MSA' )

    // inv_so2 = .false.
    // id_so2 = get_inv_ndx( 'SO2' )
    // inv_so2 = id_so2 > 0
    // if ( .not. inv_so2 ) then
    //    id_so2 = get_spc_ndx( 'SO2' )
    // endif

    // inv_NH3 = .false.
    // id_NH3 = get_inv_ndx( 'NH3' )
    // inv_NH3 = id_NH3 > 0
    // if ( .not. inv_NH3 ) then
    //    id_NH3 = get_spc_ndx( 'NH3' )
    // endif

    // inv_HNO3 = .false.
    // id_HNO3 = get_inv_ndx( 'HNO3' )
    // inv_HNO3 = id_hno3 > 0
    // if ( .not. inv_HNO3 ) then
    //    id_HNO3 = get_spc_ndx( 'HNO3' )
    // endif

    // inv_H2O2 = .false.
    // id_H2O2 = get_inv_ndx( 'H2O2' )
    // inv_H2O2 = id_H2O2 > 0
    // if ( .not. inv_H2O2 ) then
    //    id_H2O2 = get_spc_ndx( 'H2O2' )
    // endif

    // inv_HO2 = .false.
    // id_HO2 = get_inv_ndx( 'HO2' )
    // inv_HO2 = id_HO2 > 0
    // if ( .not. inv_HO2 ) then
    //    id_HO2 = get_spc_ndx( 'HO2' )
    // endif

    // inv_o3 = get_inv_ndx( 'O3' ) > 0
    // if (inv_o3) then
    //    id_o3 = get_inv_ndx( 'O3' )
    // else
    //    id_o3 = get_spc_ndx( 'O3' )
    // endif
    // inv_ho2 = get_inv_ndx( 'HO2' ) > 0
    // if (inv_ho2) then
    //    id_ho2 = get_inv_ndx( 'HO2' )
    // else
    //    id_ho2 = get_spc_ndx( 'HO2' )
    // endif

    // has_sox = (id_so2>0) .and. (id_h2o2>0) .and. (id_o3>0) .and. (id_ho2>0)
    // if (cloud_borne) then
    //    has_sox = has_sox .and. (id_h2so4>0)
    // else
    //    has_sox = has_sox .and. (id_so4>0) .and. (id_nh3>0)
    // endif

    // if (masterproc) then
    //    write(iulog,*) 'sox_inti: has_sox = ',has_sox
    // endif

    // if( has_sox ) then
    //    if (masterproc) then
    //       write(iulog,*) '-----------------------------------------'
    //       write(iulog,*) 'mozart will do sox aerosols'
    //       write(iulog,*) '-----------------------------------------'
    //    endif
    // else
    //    return
    // end if

    // call addfld( 'XPH_LWC',(/ 'lev' /), 'A','kg/kg', 'pH value multiplied by lwc')
    // if ( history_aerosol .and. history_verbose ) then
    //    call add_default ('XPH_LWC', 1, ' ')
    // endif

    // call sox_cldaero_init()

}

KOKKOS_INLINE_FUNCTION
Cloudconc sox_cldaero_create_obj(Real cldfrc, Real qcw[nspec_gas], Real lwc,
                            Real cfact, int loffset) {
  // input variables
  // cldfrc: cloud fraction [fraction]
  // qcw: cloud-borne aerosol [vmr]
  // lwc: cloud liquid water content [kg/kg]
  // cfact: total atms density total atms density [kg/L]
  // loffset: # of tracers in the host model that are not part of MAM

  Real so4c = 0.0;
  Real xlwc = 0.0;

  // xlwc is in-cloud LWC with the unit of [kg/L]
  if (cldfrc > 0.0) {
    // cloud water L(water)/L(air)
    xlwc = lwc * cfact;
    // liquid water in the cloudy fraction of cell
    xlwc = xlwc / cldfrc;
  } else {
    xlwc = 0.0;
  }

  // BAD ASSUMPTION?
  // lptr_so4_cw_amode is a modal variable, but here it's hard-coded to only
  // look at the first 3. Granted, entry 4 is zero, but will it always be?
  int id_so4_1a = lptr_so4_cw_amode[0] - loffset;
  int id_so4_2a = lptr_so4_cw_amode[1] - loffset;
  int id_so4_3a = lptr_so4_cw_amode[2] - loffset;
  so4c = qcw[id_so4_1a] + qcw[id_so4_2a] + qcw[id_so4_3a];

  // with 3-mode, assume so4 is nh4hso4, and so half-neutralized
  // FIXME: BAD CONSTANT
  Real so4_fact = 1.0;
  Cloudconc cldconc;
  cldconc.so4c = so4c;
  cldconc.xlwc = xlwc;
  cldconc.so4_fact = so4_fact;
  return cldconc;
} // end function sox_cldaero_create_obj

//===========================================================================
KOKKOS_INLINE_FUNCTION
void henry_factor_so2(Real t_factor, Real &xk, Real &xe, Real &x2) {
  //-----------------------------------------------------------------
  // get Henry Law parameters xk, xe and x2 for SO2
  //-----------------------------------------------------------------

    // FIXME: BAD CONSTANTS
    // for so2
    xk = 1.230  * haero::exp(3120.0 * t_factor);
    xe = 1.7e-2 * haero::exp(2090.0 * t_factor);
    x2 = 6.0e-8 * haero::exp(1120.0 * t_factor);
} // end subroutine henry_factor_so2

//===========================================================================
KOKKOS_INLINE_FUNCTION
void henry_factor_co2(Real t_factor, Real &xk, Real &xe) {
  //-----------------------------------------------------------------
  // get Henry Law parameters xk and xe for CO2
  //-----------------------------------------------------------------

  // for co2
  xk = 3.1e-2 * haero::exp(2423.0 * t_factor);
  xe = 4.3e-7 * haero::exp(-913.0 * t_factor);
}// end subroutine henry_factor_co2

//===========================================================================
KOKKOS_INLINE_FUNCTION
void henry_factor_h2o2(Real t_factor, Real &xk, Real &xe) {
  //-----------------------------------------------------------------
  // get Henry Law parameters xk and xe for H2O2
  //-----------------------------------------------------------------

  // for h2o2
  xk = 7.4e4   * haero::exp(6621.0  * t_factor);
  xe = 2.2e-12 * haero::exp(-3730.0 * t_factor);

} // end subroutine henry_factor_h2o2

//===========================================================================
KOKKOS_INLINE_FUNCTION
void henry_factor_o3(Real t_factor, Real &xk) {
  //-----------------------------------------------------------------
  // get Henry Law parameters xk and xe for O3
  //-----------------------------------------------------------------

  // for o3
  xk = 1.15e-2 * haero::exp(2560.0 * t_factor);
}  // end subroutine henry_factor_o3

//===========================================================================
KOKKOS_INLINE_FUNCTION
void calc_ynetpos(Real yph, Real fact1_so2, Real fact2_so2, Real fact3_so2,
                  Real fact4_so2, Real Eco2, Real Eh2o, Real Eso4,
                  Real so4_fact,
                  // out
                  Real &xph, Real &ynetpos) {
  //-----------------------------------------------------------------
  // calculate net positive ions (ynetpos) for iterations in calc_ph_values
  // also calculate H+ concentration (xph) from ph value
  //-----------------------------------------------------------------

  // calc current [H+] from pH
  xph = haero::pow(10.0, -yph);

  //-----------------------------------------------------------------
  //          ... so2
  //-----------------------------------------------------------------
  Real Eso2 = fact1_so2 / (1.0 + fact2_so2 * (1.0 + (fact3_so2 / xph) *
                                              (1.0 + fact4_so2 / xph)));

  Real tmp_hso3 = Eso2 / xph;
  Real tmp_so3  = tmp_hso3 * 2.0 * fact4_so2 / xph;
  Real tmp_hco3 = Eco2 / xph;
  Real tmp_oh   = Eh2o / xph;
  Real tmp_so4  = so4_fact * Eso4;

  // positive ions are H+ only
  Real tmp_pos = xph;
  // all negative ions
  Real tmp_neg = tmp_oh + tmp_hco3 + tmp_hso3 + tmp_so3 + tmp_so4;

  ynetpos = tmp_pos - tmp_neg;
} // end subroutine calc_ynetpos

//===========================================================================
KOKKOS_INLINE_FUNCTION
void calc_ph_values(Real temperature, Real patm, Real xlwc, Real t_factor,
                    Real xso2, Real xso4, Real xhnm, Real so4_fact, Real Ra,
                    Real xkw, Real const0, bool &converged, Real &xph) // out
{
  //---------------------------------------------------------------------------
  // calculate PH value and H+ concentration
  //
  // 21-mar-2011 changes by rce
  // now uses bisection method to solve the electro-neutrality equation
  // 3-mode aerosols (where so4 is assumed to be nh4hso4)
  //       old code set xnh4c = so4c
  //       new code sets xnh4c = 0, then uses a -1 charge (instead of -2)
  //      for so4 when solving the electro-neutrality equation
  //---------------------------------------------------------------------------
  //     implicit none

  //     real(r8),  intent(in) :: temperature        // temperature [K]
  //     real(r8),  intent(in) :: patm               // pressure [atm]
  //     real(r8),  intent(in) :: t_factor           // working variable to
  //     convert to 25 degC (1/T - 1/[298K]) real(r8),  intent(in) :: xso2 //
  //     SO2 [mol/mol] real(r8),  intent(in) :: xso4               // SO4
  //     [mol/mol] real(r8),  intent(in) :: xhnm               // [#/cm3]
  //     real(r8),  intent(in) :: xlwc               // in-cloud LWC [kg/L]
  //     real(r8),  intent(in) :: so4_fact           // factor for SO4
  //     real(r8),  intent(in) :: Ra                 // constant parameter
  //     real(r8),  intent(in) :: xkw                // constant parameter
  //     real(r8),  intent(in) :: const0             // constant parameter

  //     logical,  intent(out) :: converged          // if the method converge
  //     real(r8), intent(out) :: xph                // H+ ions concentration
  //     [mol/L]

  //     // local variables
  //     integer   :: iter  // iteration number
  //     real(r8)  :: yph_lo, yph_hi, yph    // pH values, lower and upper
  //     bounds real(r8)  :: ynetpos_lo, ynetpos_hi // lower and upper bounds of
  //     ynetpos real(r8)  :: xk, xe, x2     // output parameters in Henry's law
  //     real(r8)  :: fact1_so2, fact2_so2, fact3_so2, fact4_so2  // SO2 factors
  //     real(r8)  :: Eh2o, Eco2, Eso4 // effects of species [1/cm3]
  //     real(r8)  :: ynetpos        // net positive ions

  //     integer,  parameter :: itermax = 20  // maximum number of iterations
  //     real(r8), parameter :: co2g = 330.e-6    //330 ppm = 330.e-6 atm
  // #include "../yaml/mo_setsox/f90_yaml/calc_ph_values_beg_yml.f90"

  //----------------------------------------
  // effect of chemical species
  //----------------------------------------

  // -------------- hno3 -------------------
  // FORTRAN refactoring: not incorporated in MAM4

  // -------------- nh3 -------------------
  // FORTRAN refactoring: not incorporated in MAM4

  // -------------- so2 -------------------
  // previous code
  //    heso2(i,k)  = xk*(1.0 + xe/xph(i,k)*(1.0 + x2/xph(i,k)))
  //    px = heso2(i,k) * Ra * tz * xl
  //    so2g =  xso2(i,k)/(1.0+ px)
  //    Eso2 = xk*xe*so2g *patm
  // equivalent new code
  //    heso2 = xk + xk*xe/hplus * xk*xe*x2/hplus**2
  //    so2g = xso2/(1 + px)
  //         = xso2/(1 + heso2*ra*tz*xl)
  //         = xso2/(1 + xk*ra*tz*xl*(1 + (xe/hplus)*(1 + x2/hplus))
  //    eso2 = so2g*xk*xe*patm
  //          = xk*xe*patm*xso2/(1 + xk*ra*tz*xl*(1 + (xe/hplus)*(1 + x2/hplus))
  //          = ( fact1_so2    )/(1 + fact2_so2 *(1 + (fact3_so2/hplus)*(1 +
  //          fact4_so2/hplus)
  //    [hso3-] + 2*[so3--] = (eso2/hplus)*(1 + 2*x2/hplus)

  // output parameters in Henry's law
  Real xk, xe, x2;
  // SO2 factors
  Real fact1_so2, fact2_so2, fact3_so2, fact4_so2;
  henry_factor_so2(t_factor,
                   // out
                   xk, xe, x2);
  fact1_so2 = xk * xe * patm * xso2;
  fact2_so2 = xk * Ra * temperature * xlwc;
  fact3_so2 = xe;
  fact4_so2 = x2;

  // Eh2o, Eco2, Eso4 are "effects of species" [1/cm3]
  // -------------- h2o effects -------------------
  Real Eh2o = xkw;

  // -------------- co2 effects -------------------
  henry_factor_co2(t_factor,
                   // out
                   xk, xe);
  Real Eco2 = xk * xe * co2g * patm;

  // -------------- so4 effects -------------------
  // /cm3(a) (not sure which one this is in reference to)
  Real Eso4 = xso4 * xhnm * const0 / xlwc;

  //-----------------------------------------------------------------
  // now use bisection method to solve electro-neutrality equation
  // to calculate PH value and H+ concentration
  //
  // during the iteration loop,
  //    yph_lo = lower pH value that brackets the root (i.e., correct pH)
  //    yph_hi = upper pH value that brackets the root (i.e., correct pH)
  //    yph    = current pH value
  //    yposnet_lo and yposnet_hi = net positive ions for
  //       yph_lo and yph_hi
  //-----------------------------------------------------------------

  constexpr Real zero = 0.0;
  converged = false;
  // ---------  1st iteration: set lower bound pH value ----------
  // FIXME: BAD CONSTANT
  Real yph_lo = 2.00;
  Real yph_hi = yph_lo;
  Real yph = yph_lo;
  Real ynetpos;
  calc_ynetpos(yph, fact1_so2, fact2_so2, fact3_so2, fact4_so2, Eco2, Eh2o,
               Eso4, so4_fact,
               // out
               xph, ynetpos);
  // FIXME: BAD CONSTANT
  if (ynetpos <= zero) {
    // the lower and upper bound pH values (2.0 and 7.0) do not bracket
    // the correct pH, so use the lower bound
    converged = true;
  }
  Real ynetpos_lo = ynetpos;

  // ---------  2nd iteration: set upper bound pH value ----------
  yph_hi = 7.0;
  yph = yph_hi;
  calc_ynetpos(yph, fact1_so2, fact2_so2, fact3_so2, fact4_so2, Eco2, Eh2o,
               Eso4, so4_fact,
               // out
               xph, ynetpos);
  // FIXME: BAD CONSTANT
  if (ynetpos >= zero) {
    // the lower and upper bound pH values (2.0 and 7.0) do not bracket
    // the correct pH, so use the lower bound
    converged = true;
  }
  Real ynetpos_hi = ynetpos;

  // --------- 3rd iteration and more ------------
  for (int i = 3; i < itermax; ++i) {
    yph = 0.5 * (yph_lo + yph_hi);
    calc_ynetpos(yph, fact1_so2, fact2_so2, fact3_so2, fact4_so2, Eco2, Eh2o,
                 Eso4, so4_fact,
                 // out
                 xph, ynetpos);
    if (ynetpos >= zero) {
      // net positive ions are >= 0 for both yph and yph_lo
      // so replace yph_lo with yph
      yph_lo = yph;
      ynetpos_lo = ynetpos;
    } else {
      // net positive ions are <= 0 for both yph and yph_hi
      // so replace yph_hi with yph
      yph_hi = yph;
      ynetpos_hi = ynetpos;
    }
    // FIXME: better error handling
    // FIXME: BAD CONSTANT
    if (haero::abs(yph_hi - yph_lo) <= 0.005) {
      // |yph_hi - yph_lo| <= convergence criterion, so set
      // final pH to their midpoint and exit
      // (0.005 absolute error in pH gives 0.01 relative error in H+)
      yph = 0.5 * (yph_hi + yph_lo);
      xph = haero::pow(10.0, -yph);
      converged = true;
      return;
    }
  } // end for(iter)
} // end subroutine calc_ph_values

//===========================================================================
void calc_sox_aqueous(bool modal_aerosols, Real rah2o2, Real h2o2g, Real so2g,
                      Real o3g, Real rao3, Real patm, Real dt, Real t_factor,
                      Real xlwc, Real const0, Real xhnm, Real heo3, Real heso2,
                      // inout
                      Real &xso2, Real &xso4, Real &xso4_init, Real &xh2o2,
                      // out
                      Real &xdelso4hp) {
  //-----------------------------------------------------------------
  //       ... Prediction after aqueous phase
  //       so4
  //       When Cloud is present
  //
  //       S(IV) + H2O2 = S(VI)
  //       S(IV) + O3   = S(VI)
  //
  //       reference:
  //           (1) Seinfeld
  //           (2) Benkovitz
  //-----------------------------------------------------------------

  //   logical,  intent(in) :: modal_aerosols      // if using MAM
  //   real(r8), intent(in) :: rah2o2      // reaction rate with h2o2
  //   real(r8), intent(in) :: rao3        // reaction rate with o3
  //   real(r8), intent(in) :: h2o2g, so2g, o3g
  //   real(r8), intent(in) :: patm        // pressure [atm]
  //   real(r8), intent(in) :: dt       // time step [s]
  //   real(r8), intent(in) :: t_factor    // working variables to convert temperature
  //   real(r8), intent(in) :: xlwc        // in-cloud LWC [kg/L]
  //   real(r8), intent(in) :: const0
  //   real(r8), intent(in) :: xhnm
  //   real(r8), intent(in) :: heo3, heso2 // henry law constant
  //   real(r8), intent(inout) :: xso2, xso4, xso4_init, xh2o2 // mixing ratios
  //   real(r8), intent(out) :: xdelso4hp_ik // change of so4 in (i,k)

  //   // local variables
  //   real(r8) :: pso4    // production rate of so4
  //   real(r8) :: delta_s // so4 production in the time step
  //   real(r8) :: xk_so2, xk_h2o2 // for the use of Henry Law subroutines
  //   real(r8) :: xe, x2  // output of henry law subroutines but not used
  //   real(r8), parameter :: small_value_20 = 1.0e-2 // small value
  //   real(r8), parameter :: small_value_30 = 1.0e-3 // small value

  //............................
  //       S(IV) + H2O2 = S(VI)
  //............................

  // FORTRAN refactor: using henry law subroutines here break the BFB test because
  // it changes the order of calculation. The original code is used now but new
  // code is kept but commented out for future implementation.
  //   henry_factor_so2(t_factor, xk_so2, xe, x2)
  //   henry_factor_h2o2(t_factor, xk_h2o2, xe)
  //   pso4 = rah2o2 * xk_h2o2* h2o2g * patm * xk_so2 * so2g * patm

  Real pso4 = rah2o2 * 7.4e4 * haero::exp(6621.0 * t_factor) * h2o2g * patm * 1.23 *
         haero::exp(3120.0 * t_factor) * so2g * patm;
  // [M/s] = [mole/L(w)/s] / [mole/L(a)/s] / [/L(a)/s]
  pso4 = pso4 * xlwc / const0 / xhnm;

  Real delta_s = haero::max(pso4 * dt, small_value_30);

  xso4_init = xso4;

  if ((delta_s <= xso2) && (delta_s <= xh2o2)) {
        xso4  = xso4  + delta_s;
        xh2o2 = xh2o2 - delta_s;
        xso2  = xso2  - delta_s;
      }
  else if (xh2o2 > xso2) {
        xso4 = xso4 + xso2;
        xh2o2 = xh2o2 - xso2;
        xso2 = small_value_20;
      }
  else {
        xso4 = xso4 + xh2o2;
        xso2 = xso2 - xh2o2;
        xh2o2 = small_value_20;
      }

  if (modal_aerosols) {
       xdelso4hp  =  xso4 - xso4_init;
  }
  //...........................
  //       S(IV) + O3 = S(VI)
  //...........................

  // [M/s]
  pso4 = rao3 * heo3 * o3g * patm * heso2 * so2g * patm;
  // [M/s] =[mole/L(w)/s] * [mole/L(a)/s] / [/L(a)/s] / [mixing ratio/s]
  pso4 = pso4 * xlwc / const0 / xhnm;

  delta_s = haero::max(pso4 * dt, small_value_30);

  xso4_init = xso4;

  if (delta_s > xso2) {
       xso4 = xso4 + xso2;
       xso2 = small_value_20;
     }
  else {
       xso4 = xso4 + delta_s;
       xso2 = xso2 - delta_s;
  }
} // end subroutine calc_sox_aqueous

// =============================================================================
KOKKOS_INLINE_FUNCTION
void compute_aer_factor(tmr, loffset,
                        // out
                        faqgain_so4) {
//   -------------------------------------------------------------------------
//    compute factors for partitioning aerosol mass gains among modes
//    the factors are proportional to the activated particle MR for each
//    mode, which is the MR of cloud drops "associated with" the mode
//    thus we are assuming the cloud drop size is independent of the
//    associated aerosol mode properties (i.e., drops associated with
//    Aitken and coarse sea-salt particles are same size)
//    qnum_c(n) = activated particle number MR for mode n (these are just
//    used for partitioning among modes, so don't need to divide by cldfrc)
//   -------------------------------------------------------------------------
//   real(r8), intent(in) :: tmr(:)    tracer mixing ratio [vmr]
//   integer,  intent(in) :: loffset   # of tracers in the host model that are not part of MAM
//   real(r8), intent(out) :: faqgain_so4(ntot_amode)    factor of TMR among modes [fraction]

//    local variables
//   integer  :: imode, ll                index
//   real(r8) :: sumf                     total TMR for all modes [vmr]
//   real(r8) :: qnum_c(ntot_amode)       tracer mixing ratio [vmr]
//   real(r8), parameter :: small_value_10 = 1.e-10_r8

//   -------------------------------------------------------------------------
  constexpr Real zero = 0.0;
  sumf  = zero;
  qnum_c(:)  = zero;
  faqgain_so4(:) = zero;

  // ***mjs: HERE***
  for (int m = 0; m < nmodes; ++m)
  {
    ll = numptrcw_amode(imode) - loffset
    if (ll > 0) qnum_c(imode) = max( zero, tmr(ll) )

     // force qnum_c(n) to be positive for n=modeptr_accum or n=1
    if (imode == modeptr_accum) qnum_c(imode) = max(small_value_10, qnum_c(imode))

     faqgain_so4(n) = fraction of total so4_c gain going to mode n
     // these are proportional to the activated particle MR for each mode
    if (lptr_so4_cw_amode(imode) > 0) then
       faqgain_so4(imode) = qnum_c(imode)
       sumf = sumf + faqgain_so4(imode)
    endif
  }

  // at this point (sumf <= 0.0) only when all the faqgain_so4 are zero
  if (sumf > zero) {
    faqgain_so4(:) = faqgain_so4(:)/sumf;
  }

} // end subroutine compute_aer_factor

//=================================================================================
KOKKOS_INLINE_FUNCTION
void sox_cldaero_update(ncol, lchnk, loffset, dtime, mbar, pdel, press, tfld,
                        cldnum, cldfrc, cfact, xlwc, delso4_hprxn, xh2so4,
                        xso4, xso4_init,
                        // inout
                        qcw, qin ) {
//----------------------------------------------------------------------------------
// Update the mixing ratios
//----------------------------------------------------------------------------------

//     // args
//     integer,  intent(in) :: ncol
//     integer,  intent(in) :: lchnk       // chunk id
//     integer,  intent(in) :: loffset     // # of tracers in the host model that are not part of MAM
//     real(r8), intent(in) :: dtime       // time step [sec]

//     real(r8), intent(in) :: mbar(:,:)   // mean wet atmospheric mass [amu or g/mol]
//     real(r8), intent(in) :: pdel(:,:)   // pressure interval [Pa]
//     real(r8), intent(in) :: press(:,:)  // pressure [Pa]
//     real(r8), intent(in) :: tfld(:,:)   // temperature [K]

//     real(r8), intent(in) :: cldnum(:,:) // droplet number concentration [#/kg]
//     real(r8), intent(in) :: cldfrc(:,:) // cloud fraction [fraction]
//     real(r8), intent(in) :: cfact(:,:)  // total atms density [kg/L]
//     real(r8), intent(in) :: xlwc(:,:)   // liquid water volume [cm^3/cm^3]

//     real(r8), intent(in) :: delso4_hprxn(:,:)   // change of so4 due to H2O2 chemistry [mol/mol]
//     real(r8), intent(in) :: xh2so4(:,:)         // H2SO4 mass mixing ratio [mol/mol]
//     real(r8), intent(in) :: xso4(:,:)           // final SO4 mass mixing ratio [mol/mol]
//     real(r8), intent(in) :: xso4_init(:,:)      // initial SO4 mass mixing ratio [mol/mol]

//     real(r8), intent(inout) :: qcw(:,:,:) // cloud-borne aerosol [vmr]
//     real(r8), intent(inout) :: qin(:,:,:) // xported species [vmr]

//     // local vars ...

//     // FORTRAN refactor note: aqueous chemistry (aq) here reprent two processes:
//     //       S(IV) + H2O2 = S(VI)
//     //       S(IV) + O3   = S(VI)
//     // see the parent subroutine in mo_setsox for reference
//     real(r8) :: delso4_o3rxn    // change of so4 due to O3 chemistry [mol/mol]
//     real(r8) :: dso4dt_aqrxn    // so4_c tendency from aqueous chemistry [mol/mol/s]
//     real(r8) :: dso4dt_hprxn    // so4_c tendency from H2O2 chemistry [mol/mol/s]
//     real(r8) :: dso4dt_gasuptk  // so4_c tendency from h2so4 gas uptake [mol/mol/s]
//     real(r8) :: dqdt_aq         // dqdt due to aqueous chemistry [mol/mol/s]
//     real(r8) :: dqdt_wr         // dqdt due to wet removal, currently set as zero [mol/mol/s]
//     real(r8) :: dqdt_aqso4(ncol,pver,gas_pcnst)    // dqdt due to so4 aqueous chemistry [mol/mol/s]
//     real(r8) :: dqdt_aqh2so4(ncol,pver,gas_pcnst)  // dqdt due to h2so4 uptake [mol/mol/s]
//     real(r8) :: dqdt_aqhprxn(ncol,pver)         // dqdt due to H2O2 chemistry [mol/mol/s]
//     real(r8) :: dqdt_aqo3rxn(ncol,pver)         // dqdt due to O3 chemistry [mol/mol/s]
//     real(r8) :: sflx(ncol)      // integrated surface fluxes [kg/m2/s]
//     real(r8) :: faqgain_so4(ntot_amode) // factor of TMR among modes [fraction]
//     real(r8) :: uptkrate        // uptake rate [1/s]

//     integer :: ll, mm, imode    // aerosol mode index
//     integer :: icol,kk          // column and level index

  constexpr Real small_value_8 = 1.0e-8;
  constexpr Real small_value_5 = 1.0e-5;
  constexpr Real zero = 0.0;

  // make sure dqdt is zero initially, for budgets
  Real dqdt_aqso4[nspec_gas], dqdt_aqh2so4[nspec_gas];
  for (int i = 0; i < nspec_gas; ++i)
  {
    dqdt_aqso4[i] = zero;
    dqdt_aqh2so4[i] = zero;
  }
  Real dqdt_aqhprxn = zero;
  Real dqdt_aqo3rxn = zero;

  if ((cldfrc >= small_value_5) && (xlwc >= small_value_8)) {

  //-------------------------------------------------------------------------
  // compute factors for partitioning aerosol mass gains among modes
  // the factors are proportional to the activated particle MR for each
  // mode, which is the MR of cloud drops "associated with" the mode
  // thus we are assuming the cloud drop size is independent of the
  // associated aerosol mode properties (i.e., drops associated with
  // Aitken and coarse sea-salt particles are same size)
  compute_aer_factor(qcw, loffset,
                     faqgain_so4); // out

    // uptkrate = cldaero_uptakerate(xlwc, cldnum, &
    //             cfact, cldfrc, tfld,  press)
  //   // average uptake rate over dtime
  //   uptkrate = (1.0_r8 - exp(-min(100._r8,dtime*uptkrate))) / dtime
  //   // dso4dt_gasuptk = so4_c tendency from h2so4 gas uptake (mol/mol/s)
  //   dso4dt_gasuptk = xh2so4 * uptkrate

  //   delso4_o3rxn = xso4 - xso4_init
  //   dso4dt_aqrxn = (delso4_o3rxn + delso4_hprxn) / dtime
  //   dso4dt_hprxn = delso4_hprxn / dtime

  //   //-----------------------------------------------------------------------
  //   // now compute TMR tendencies
  //   // this includes the above aqueous so2 chemistry AND
  //   // the uptake of highly soluble aerosol precursor gases (h2so4, ...)
  //   // The wetremoval of dissolved, unreacted so2 and h2o2 are assumed as zero

  //   // compute TMR tendencies for so4 aerosol-in-cloud-water
  //   do imode = 1, ntot_amode
  //      ll = lptr_so4_cw_amode(imode) - loffset
  //      if (ll > 0) then
  //         dqdt_aqso4(icol,kk,ll) = faqgain_so4(imode)*dso4dt_aqrxn*cldfrc
  //         dqdt_aqh2so4(icol,kk,ll) = faqgain_so4(imode)*dso4dt_gasuptk*cldfrc
  //         dqdt_aq = dqdt_aqso4(icol,kk,ll) + dqdt_aqh2so4(icol,kk,ll)
  //         dqdt_wr =  0.0_r8 // don't have wet removal here
  //         call update_tmr ( qcw(icol,kk,ll), dqdt_aq + dqdt_wr, dtime )
  //      endif
  //   enddo

  //   // For gas species, tendency includes reactive uptake to cloud water
  //   // that essentially transforms the gas to a different species.
  //   // Need to multiply both these parts by cldfrc
  //   // Currently it assumes no wet removal here

  //   // h2so4 (g)
  //   qin(icol,kk,id_h2so4) = qin(icol,kk,id_h2so4) - dso4dt_gasuptk * dtime * cldfrc
  // // FORTRAN refactor: The order of multiplying cldfrc makes the following call
  // // failing BFB test, so this calculation is not refactored with new subroutine

  //   // so2 -- the first order loss rate for so2 is frso2_c*clwlrat(i,k)
  //   dqdt_wr =  0.0_r8 // don't have wet removal here
  //   dqdt_aq = -dso4dt_aqrxn*cldfrc
  //   call update_tmr ( qin(icol,kk,id_so2), dqdt_aq + dqdt_wr, dtime )

  //   // h2o2 -- the first order loss rate for h2o2 is frh2o2_c*clwlrat(i,k)
  //   dqdt_wr =  0.0_r8 // don't have wet removal here
  //   dqdt_aq = -dso4dt_hprxn*cldfrc
  //   call update_tmr ( qin(icol,kk,id_h2o2), dqdt_aq + dqdt_wr, dtime )

  //   // for SO4 from H2O2/O3 budgets
  //   dqdt_aqhprxn = dso4dt_hprxn*cldfrc
  //   dqdt_aqo3rxn = (dso4dt_aqrxn - dso4dt_hprxn)*cldfrc

  }

//     //==============================================================
//     // ... Update the mixing ratios
//     //==============================================================
//     do imode = 1, ntot_amode
//        call update_tmr_nonzero ( qcw, (lptr_so4_cw_amode(imode) - loffset) )
//        call update_tmr_nonzero ( qcw, (lptr_nh4_cw_amode(imode) - loffset) )
//     enddo
//     call update_tmr_nonzero ( qin, id_so2 )


//     // diagnostics

//     do imode = 1, ntot_amode
//        mm = lptr_so4_cw_amode(imode)
//        ll = mm - loffset
//        if (ll > 0) then
//           call calc_sfc_flux( dqdt_aqso4(:,:,ll)*adv_mass(ll)/mbar, pdel, sflx)
//           call outfld( trim(cnst_name_cw(mm))//'AQSO4', sflx(:ncol), ncol, lchnk)

//           call calc_sfc_flux( dqdt_aqh2so4(:,:,ll)*adv_mass(ll)/mbar, pdel, sflx)
//           call outfld( trim(cnst_name_cw(mm))//'AQH2SO4', sflx(:ncol), ncol, lchnk)
//        endif
//     enddo

//     call calc_sfc_flux( dqdt_aqhprxn*specmw_so4_amode/mbar, pdel, sflx)
//     call outfld( 'AQSO4_H2O2', sflx(:ncol), ncol, lchnk)

//     call calc_sfc_flux( dqdt_aqo3rxn*specmw_so4_amode/mbar, pdel, sflx)
//     call outfld( 'AQSO4_O3', sflx(:ncol), ncol, lchnk)
// } // end subroutine sox_cldaero_update

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION
void setsox(Real xhnm, Real cldfrc, Real qcw[nspec_gas], Real lwc, Real tfld,
            Real press, Real qin[nspec_gas], Real dt, Real mbar, Real pdel,
            Real cldnum) {
            // out
            // Real &qcw, Real &qin) {
  // don't know dimension of:
    // mbar

  // xnhm := total atms density [#/cm3]

  //-----------------------------------------------------------------------
  //          ... Compute heterogeneous reactions of SOX
  //
  //       (0) using initial PH to calculate PH
  //           (a) HENRYs law constants
  //           (b) PARTIONING
  //           (c) PH values
  //
  //       (1) using new PH to repeat
  //           (a) HENRYs law constants
  //           (b) PARTIONING
  //           (c) REACTION rates
  //           (d) PREDICTION
  //-----------------------------------------------------------------------
  //
  //     use ppgrid,          only : pver
  //     use cam_history,     only : outfld
  //     use sox_cldaero_mod, only : sox_cldaero_update,
  //     sox_cldaero_create_obj, sox_cldaero_destroy_obj use cldaero_mod, only
  //     : cldaero_conc_t use cam_abortutils,  only : endrun

  //
  //     implicit none
  //
  //-----------------------------------------------------------------------
  //      ... Dummy arguments
  //-----------------------------------------------------------------------
  //     integer,          intent(in)    :: ncol              // num of
  //     columns in chunk integer,          intent(in)    :: lchnk // chunk id
  //     integer,          intent(in)    :: loffset           // offset of
  //     chem tracers in the advected tracers array real(r8), intent(in)    ::
  //     dt             // time step [sec] real(r8),         intent(in) ::
  //     press(:,:)        // midpoint pressure [Pa] real(r8), intent(in) ::
  //     pdel(:,:)         // pressure thickness of levels [Pa] real(r8),
  //     intent(in)    :: tfld(:,:)         // temperature [K] real(r8),
  //     intent(in)    :: mbar(:,:)         // mean wet atmospheric mass [amu
  //     or g/mol] real(r8), target, intent(in)    :: lwc(:,:)          //
  //     cloud liquid water content [kg/kg] real(r8), target, intent(in)    ::
  //     cldfrc(:,:)       // cloud fraction [fraction] real(r8), intent(in)
  //     :: cldnum(:,:)       // droplet number concentration [#/kg] real(r8),
  //     intent(in)    :: xhnm(:,:)         // total atms density [#/cm**3]
  //     real(r8), target, intent(inout) :: qcw(:,:,:)        // cloud-borne
  //     aerosol [vmr] real(r8),         intent(inout) :: qin(:,:,:)        //
  //     transported species [vmr]

  //-----------------------------------------------------------------------
  //      ... Local variables
  //
  //      FORTRAN refactoring: the units are a little messy here
  //      my understanding (may not be right) is that, the PH value xph, shown
  //      in [H+] concentration, is (mol H+)/(L water), which can be
  //      transfered to kg/L or kg/kg the variables xso2, xso4, xo3 etc have
  //      units of [mol/mol] (maybe corresponding to kg/kg above?) the
  //      variable xhnm has unit of [#/cm3]. Some units may changes to
  //      different formats across modules
  //      Shuaiqi Tang  4/18/2023
  //-----------------------------------------------------------------------
  //     real(r8), parameter :: const0 = 1.0e30/6.023e230 // [cm3/L *
  //     mol/mole] real(r8), parameter :: Ra = 8314.0/101325.0 // universal
  //     constant   (atm)/(M-K) real(r8), parameter :: xkw = 1.0e-14 // water
  //     acidity real(r8), parameter :: p0 = 101300.0          // sea-level
  //     pressure [Pa] real(r8), parameter :: t298K = 298.0          //
  //     temperature of 25degC [K] real(r8), parameter :: small_value_lwc
  //     = 1.0e-8 // small value of LWC [kg/kg] real(r8), parameter ::
  //     small_value_cf = 1.0e-5  // small value of cloud fraction [fraction]

  //     integer  :: icol,kk
  //     logical  :: converged
  //     real(r8) :: t_factor       // working variables to convert
  //     temperature real(r8) :: cfact(ncol,pver)        // total atms density
  //     [kg/L] real(r8) :: xk, xe, x2     // output parameters in Henry's law
  //     real(r8) :: tz      // temperature at (i,k) [K]
  //     real(r8) :: xlwc    // in-cloud LWC at (i,k) [kg/L]
  //     real(r8) :: px      // temporary variable [unitless]
  //     real(r8) :: patm    // pressure [atm]

  // the concentration values can vary in different forms
  //     real(r8) :: xph0,  xph(ncol,pver)   // pH value in H+ concentration
  //     [mol/L, or kg/L, or kg/kg(w)] real(r8) :: so2g, h2o2g, o3g        //
  //     concentration in gas phase [mol/mol] real(r8) :: rah2o2, rao3 //
  //     reaction rate
  // mass concentration for species
  //     real(r8), dimension(ncol,pver) :: xso2, xso4, xso4_init, xh2so4, xo3,
  //     xh2o2
  //                                       // species concentrations [mol/mol]
  //     real(r8), pointer :: xso4c(:,:)
  //     real(r8) :: xdelso4hp(ncol,pver)    // change of so4 [mol/mol]
  //     real(r8) :: xphlwc(ncol,pver)       // pH value multiplied by lwc
  //     [kg/kg]

  //     real(r8), dimension(ncol,pver) :: heh2o2,heso2,heo3 // henry law
  //     const for species type(cldaero_conc_t), pointer :: cldconc
  // #include "../yaml/mo_setsox/f90_yaml/setsox_beg_yml.f90"

  //==================================================================
  //       ... First set the PH
  //==================================================================
  //      ... Initial values
  //           The values of so2, so4 are after (1) SLT, and CHEM
  //-----------------------------------------------------------------
  constexpr Real one = 1.0;
  constexpr Real zero = 0.0;
  constexpr Real t298K = 298.0;

  constexpr Real ph0 = 5.0; // Initial PH values
  // initial PH value, in H+ concentration
  Real xph0 = haero::pow(10, -ph0);
  // cfact := total atms density [kg/L]
  // FIXME: BAD CONSTANTS
  //           cm-3 * m-3    * kg/m3            * kg/L;
  Real cfact = xhnm * 1.0e6 * 1.38e-23 / 287.0 * 1.0e-3;

  // initialize species concentrations
  Cloudconc cldconc;
  // this name doesn't make sense after porting
  cldconc = sox_cldaero_create_obj(cldfrc, qcw, lwc, cfact, loffset);

  //     if ( inv_so2 .or. id_hno3>0 .or. inv_h2o2 .or. id_nh3>0 .or. inv_o3 &
  //                  .or. (.not. inv_ho2) .or. (.not. cloud_borne) .or.
  //                  id_msa>0) then
  //         call endrun('FORTRAN refactoring: Only keep the code for default
  //         MAM4. &
  //              The following options are removed:  id_nh3>0  id_hno3>0
  //              id_msa>0 & inv_h2o2=.T. inv_so2=.T.  inv_o3=.T. inv_ho2=.F.
  //              cloud_borne=.F. ')
  //     endif

  // species molar mixing ratios(?) [mol/mol]
  Real xso4 = zero;
  // initial PH value
  Real xph = xph0;
  Real xso2 = qin[id_so2];
  Real xh2o2 = qin[id_h2o2];
  Real xo3 = qin[id_o3];
  Real xh2so4 = qin[id_h2so4];

  //-----------------------------------------------------------------
  //       ... Temperature dependent Henry constants
  //-----------------------------------------------------------------
  //     ver_loop0: do kk = 1,pver          //// pver loop for STEP 0
  //        col_loop0: do icol = 1,ncol

  Real t_factor, patm;
  // there doesn't appear to be any reason for doing this
  Real xso4c = cldconc.so4c;
  // in-cloud liquid water content
  Real xlwc = cldconc.xlwc;
  if (xlwc >= small_value_lwc) {
    t_factor = (one / tfld) - (one / t298K);
    // calculate press in atm
    patm = press / p0;

    if ((cloud_borne > zero) && (cldfrc > zero)) {
      xso4 = xso4c / cldfrc;
    }
    bool converged = false;
    calc_ph_values(tfld, patm, xlwc, t_factor, xso2, xso4, xhnm,
                   cldconc.so4_fact, Ra, xkw, const0, converged,
                   // out
                   xph);

    // FIXME: better error handling
    // if (!converged) {
    // write(iulog, *) 'setsox: pH failed to converge @ (', icol, ',', kk,
    // ').'
    // }
  } else {
      // FIXME: BAD CONSTANT
      xph = 1.0e-7;
  }

//==============================================================
//          ... Now use the actual pH
//==============================================================

  t_factor = (one / tfld) - (one / t298K);
  xlwc = cldconc.xlwc;
  // calculate press in atm
  patm = press / p0;

  Real xk, xe, x2;

  //-----------------------------------------------------------------
  //        ... h2o2
  //-----------------------------------------------------------------
  henry_factor_h2o2(t_factor,
                    // out
                    xk, xe);
  Real heh2o2 = xk * (one + xe / xph);

  //-----------------------------------------------------------------
  //         ... so2
  //-----------------------------------------------------------------
  henry_factor_so2(t_factor,
                   // out
                   xk, xe, x2);
  Real heso2 = xk * (one + xe / xph * (one + x2 / xph));

  //-----------------------------------------------------------------
  //        ... o3
  //-----------------------------------------------------------------
  henry_factor_o3(t_factor,
                  // out
                  xk);
  Real heo3 = xk;

  //-----------------------------------------------
  //       ... Partioning
  //-----------------------------------------------
  Real tz = tfld;

  //------------------------------------------------------------------------
  //        ... h2o2
  //------------------------------------------------------------------------
  Real px = heh2o2 * Ra * tz * xlwc;
  Real h2o2g =  xh2o2 / (one + px);

  //------------------------------------------------------------------------
  //         ... so2
  //------------------------------------------------------------------------
  px = heso2 * Ra * tz * xlwc;
  Real so2g = xso2 / (one + px);

  //------------------------------------------------------------------------
  //         ... o3
  //------------------------------------------------------------------------
  px = heo3 * Ra * tz * xlwc;
  Real o3g = xo3 / (one + px);

  //-----------------------------------------------
  //       ... Aqueous phase reaction rates
  //           SO2 + H2O2 -> SO4
  //           SO2 + O3   -> SO4
  //-----------------------------------------------

  //------------------------------------------------------------------------
  //       ... S(IV) (HSO3) + H2O2
  //------------------------------------------------------------------------
  Real rah2o2 = 8.e4 * haero::exp(-3650.0 * t_factor) / (0.1 + xph);

  //------------------------------------------------------------------------
  //        ... S(IV)+ O3
  //------------------------------------------------------------------------
  Real rao3 = 4.39e11 * haero::exp(-4131.0 / tz) + 2.56e3  *
         haero::exp(-996.0 / tz) / xph;

  //-----------------------------------------------------------------
  //       ... Prediction after aqueous phase
  //       so4
  //       When Cloud is present
  //
  //       S(IV) + H2O2 = S(VI)
  //       S(IV) + O3   = S(VI)
  //
  //       reference:
  //           (1) Seinfeld
  //           (2) Benkovitz
  //-----------------------------------------------------------------

  //............................
  //       S(IV) + H2O2 = S(VI)
  //............................

  Real xdelso4hp, xso4_init;
  // WHEN CLOUD IS PRESENTED (present?)
  if (xlwc >= small_value_lwc) {
    calc_sox_aqueous(modal_aerosols, rah2o2, h2o2g, so2g, o3g, rao3, patm,
                     dt, t_factor, xlwc,  const0, xhnm, heo3, heso2,
                     // inout
                     xso2, xso4, xso4_init, xh2o2,
                     // out
                     xdelso4hp);
  } // end WHEN CLOUD IS PRESENTED (present?)

  // mean wet atmospheric mass [amu or g/mol]
  sox_cldaero_update(loffset, dt, mbar, pdel, press, tfld, cldnum,
                     cldfrc, cfact, cldconc.xlwc, xdelso4hp, xh2so4, xso4,
                     xso4_init,
                     // inout
                     qcw, qin);

  // diagnose variable
//     xphlwc(:,:) = 0.0
//     do kk = 1, pver
//        do icol = 1, ncol
//           if (cldfrc>=small_value_cf .and. lwc>=small_value_lwc) then
//              xphlwc = -one*log10(xph) * lwc
//           endif
//        enddo
//     enddo
//     call outfld( 'XPH_LWC', xphlwc(:ncol,:), ncol , lchnk )

//     call sox_cldaero_destroy_obj(cldconc)

}//   end subroutine setsox
} // namespace mo_setsox
} // namespace mam4
#endif
