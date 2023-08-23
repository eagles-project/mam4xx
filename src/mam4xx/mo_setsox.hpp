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

  // use shr_kind_mod, only : r8 => shr_kind_r8
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

// number of vertical levels
constexpr int pver = mam4::nlev;

KOKKOS_INLINE_FUNCTION
void sox_init() {
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

// //-----------------------------------------------------------------------
// //-----------------------------------------------------------------------
//   subroutine setsox(   &
//        ncol,   lchnk,  loffset,   dtime,  & // in
//        press,  pdel,   tfld,      mbar,   & // in
//        lwc,    cldfrc, cldnum,            & // in
//        xhnm,                              & // in
//        qcw,    qin                        ) // inout

//     //-----------------------------------------------------------------------
//     //          ... Compute heterogeneous reactions of SOX
//     //
//     //       (0) using initial PH to calculate PH
//     //           (a) HENRYs law constants
//     //           (b) PARTIONING
//     //           (c) PH values
//     //
//     //       (1) using new PH to repeat
//     //           (a) HENRYs law constants
//     //           (b) PARTIONING
//     //           (c) REACTION rates
//     //           (d) PREDICTION
//     //-----------------------------------------------------------------------
//     //
//     use ppgrid,          only : pver
//     use cam_history,     only : outfld
//     use sox_cldaero_mod, only : sox_cldaero_update, sox_cldaero_create_obj, sox_cldaero_destroy_obj
//     use cldaero_mod,     only : cldaero_conc_t
//     use cam_abortutils,  only : endrun

//     //
//     implicit none
//     //
//     //-----------------------------------------------------------------------
//     //      ... Dummy arguments
//     //-----------------------------------------------------------------------
//     integer,          intent(in)    :: ncol              // num of columns in chunk
//     integer,          intent(in)    :: lchnk             // chunk id
//     integer,          intent(in)    :: loffset           // offset of chem tracers in the advected tracers array
//     real(r8),         intent(in)    :: dtime             // time step [sec]
//     real(r8),         intent(in)    :: press(:,:)        // midpoint pressure [Pa]
//     real(r8),         intent(in)    :: pdel(:,:)         // pressure thickness of levels [Pa]
//     real(r8),         intent(in)    :: tfld(:,:)         // temperature [K]
//     real(r8),         intent(in)    :: mbar(:,:)         // mean wet atmospheric mass [amu or g/mol]
//     real(r8), target, intent(in)    :: lwc(:,:)          // cloud liquid water content [kg/kg]
//     real(r8), target, intent(in)    :: cldfrc(:,:)       // cloud fraction [fraction]
//     real(r8),         intent(in)    :: cldnum(:,:)       // droplet number concentration [#/kg]
//     real(r8),         intent(in)    :: xhnm(:,:)         // total atms density [#/cm**3]
//     real(r8), target, intent(inout) :: qcw(:,:,:)        // cloud-borne aerosol [vmr]
//     real(r8),         intent(inout) :: qin(:,:,:)        // transported species [vmr]

//     //-----------------------------------------------------------------------
//     //      ... Local variables
//     //
//     //      FORTRAN refactoring: the units are a little messy here
//     //      my understanding (may not be right) is that, the PH value xph, shown in [H+] concentration,
//     //      is (mol H+)/(L water), which can be transfered to kg/L or kg/kg
//     //      the variables xso2, xso4, xo3 etc have units of [mol/mol] (maybe
//     //      corresponding to kg/kg above?)
//     //      the variable xhnm has unit of [#/cm3]. Some units may changes to
//     //      different formats across modules
//     //      Shuaiqi Tang  4/18/2023
//     //-----------------------------------------------------------------------
//     real(r8), parameter :: ph0 = 5.0_r8  // Initial PH values
//     real(r8), parameter :: const0 = 1.e3_r8/6.023e23_r8 // [cm3/L * mol/mole]
//     real(r8), parameter :: Ra = 8314._r8/101325._r8 // universal constant   (atm)/(M-K)
//     real(r8), parameter :: xkw = 1.e-14_r8          // water acidity
//     real(r8), parameter :: p0 = 101300._r8          // sea-level pressure [Pa]
//     real(r8), parameter :: t298K = 298._r8          // temperature of 25degC [K]
//     real(r8), parameter :: small_value_lwc = 1.e-8_r8 // small value of LWC [kg/kg]
//     real(r8), parameter :: small_value_cf = 1.e-5_r8  // small value of cloud fraction [fraction]

//     integer  :: icol,kk
//     logical  :: converged
//     real(r8) :: t_factor       // working variables to convert temperature
//     real(r8) :: cfact(ncol,pver)        // total atms density [kg/L]
//     real(r8) :: xk, xe, x2     // output parameters in Henry's law
//     real(r8) :: tz      // temperature at (i,k) [K]
//     real(r8) :: xlwc    // in-cloud LWC at (i,k) [kg/L]
//     real(r8) :: px      // temporary variable [unitless]
//     real(r8) :: patm    // pressure [atm]

//     // the concentration values can vary in different forms
//     real(r8) :: xph0,  xph(ncol,pver)   // pH value in H+ concentration [mol/L, or kg/L, or kg/kg(w)]
//     real(r8) :: so2g, h2o2g, o3g        // concentration in gas phase [mol/mol]
//     real(r8) :: rah2o2, rao3            // reaction rate
//     // mass concentration for species
//     real(r8), dimension(ncol,pver) :: xso2, xso4, xso4_init, xh2so4, xo3, xh2o2
//                                       // species concentrations [mol/mol]
//     real(r8), pointer :: xso4c(:,:)
//     real(r8) :: xdelso4hp(ncol,pver)    // change of so4 [mol/mol]
//     real(r8) :: xphlwc(ncol,pver)       // pH value multiplied by lwc [kg/kg]

//     real(r8), dimension(ncol,pver) :: heh2o2,heso2,heo3 // henry law const for species
//     type(cldaero_conc_t), pointer :: cldconc
// #include "../yaml/mo_setsox/f90_yaml/setsox_beg_yml.f90"


//     //==================================================================
//     //       ... First set the PH
//     //==================================================================
//     //      ... Initial values
//     //           The values of so2, so4 are after (1) SLT, and CHEM
//     //-----------------------------------------------------------------
//     // initial PH value, in H+ concentration
//     xph0 = 10._r8**(-ph0)

//     // calculate total atms density [kg/L]
//     cfact(:,:) = xhnm(:,:)        &          // /cm3(a)
//             * 1.e6_r8             &          // /m3(a)
//             * 1.38e-23_r8/287._r8 &          // Kg(a)/m3(a)
//             * 1.e-3_r8                       // Kg(a)/L(a)

//     if ( inv_so2 .or. id_hno3>0 .or. inv_h2o2 .or. id_nh3>0 .or. inv_o3 &
//                  .or. (.not. inv_ho2) .or. (.not. cloud_borne) .or. id_msa>0) then
//         call endrun('FORTRAN refactoring: Only keep the code for default MAM4. &
//              The following options are removed:  id_nh3>0  id_hno3>0  id_msa>0 &
//              inv_h2o2=.T. inv_so2=.T.  inv_o3=.T. inv_ho2=.F. cloud_borne=.F. ')
//     endif

//     // initialize species concentrations
//     cldconc => sox_cldaero_create_obj( cldfrc,qcw,lwc, cfact, ncol, loffset )
//     xso4c => cldconc%so4c

//     xso4(:,:)   = 0._r8
//     xph(:,:)    = xph0          // initial PH value
//     xso2(:,:)   = qin(:,:,id_so2)
//     xh2o2(:,:)  = qin(:,:,id_h2o2)
//     xo3 (:,:)   = qin(:,:,id_o3)
//     xh2so4(:,:) = qin(:,:,id_h2so4)

//     //-----------------------------------------------------------------
//     //       ... Temperature dependent Henry constants
//     //-----------------------------------------------------------------
//     ver_loop0: do kk = 1,pver          //// pver loop for STEP 0
//        col_loop0: do icol = 1,ncol

//           // in-cloud liquid water content
//           xlwc = cldconc%xlwc(icol,kk)
//           if( xlwc >= small_value_lwc ) then

//              t_factor = (1._r8 / tfld(icol,kk)) - (1._r8 / t298K)
//              patm = press(icol,kk)/p0        // calculate press in atm

//              if (cloud_borne .and. cldfrc(icol,kk)>0._r8) then
//                 xso4(icol,kk) = xso4c(icol,kk) / cldfrc(icol,kk)
//              endif

//              call calc_ph_values(                      &
//                 tfld(icol,kk), patm, xlwc,  t_factor,  & // in
//                 xso2(icol,kk), xso4(icol,kk),          & // in
//                 xhnm(icol,kk), cldconc%so4_fact,       & // in
//                 Ra,            xkw,  const0,           & // in
//                 converged,     xph(icol,kk)            ) // out

//              if( .not. converged ) then
//                 write(iulog,*) 'setsox: pH failed to converge @ (',icol,',',kk,').'
//              endif

//           else
//              xph(icol,kk) =  1.e-7_r8
//           endif

//        enddo col_loop0
//     enddo ver_loop0 // end pver loop for STEP 0

//     //==============================================================
//     //          ... Now use the actual PH
//     //==============================================================
//     ver_loop1: do kk = 1,pver
//        col_loop1: do icol = 1,ncol

//           t_factor = (1._r8 / tfld(icol,kk)) - (1._r8 / t298K)
//           xlwc = cldconc%xlwc(icol,kk)
//           patm = press(icol,kk)/p0        // calculate press in atm

//           //-----------------------------------------------------------------
//           //        ... h2o2
//           //-----------------------------------------------------------------
//           call henry_factor_h2o2(t_factor,      & // in
//                                 xk, xe          ) // out
//           heh2o2(icol,kk)  = xk*(1._r8 + xe/xph(icol,kk))

//           //-----------------------------------------------------------------
//           //         ... so2
//           //-----------------------------------------------------------------
//           call henry_factor_so2(t_factor,       & // in
//                                 xk, xe, x2      ) // out
//           heso2(icol,kk)  = xk*(1._r8 + xe/xph(icol,kk)*(1._r8 + x2/xph(icol,kk)))

//           //-----------------------------------------------------------------
//           //        ... o3
//           //-----------------------------------------------------------------
//           call henry_factor_o3(t_factor,        & // in
//                                 xk              ) // out
//           heo3(icol,kk) = xk

//           //-----------------------------------------------
//           //       ... Partioning
//           //-----------------------------------------------
//           tz = tfld(icol,kk)

//           //------------------------------------------------------------------------
//           //        ... h2o2
//           //------------------------------------------------------------------------
//           px = heh2o2(icol,kk) * Ra * tz * xlwc
//           h2o2g =  xh2o2(icol,kk)/(1._r8+ px)

//           //------------------------------------------------------------------------
//           //         ... so2
//           //------------------------------------------------------------------------
//           px = heso2(icol,kk) * Ra * tz * xlwc
//           so2g =  xso2(icol,kk)/(1._r8+ px)

//           //------------------------------------------------------------------------
//           //         ... o3
//           //------------------------------------------------------------------------
//           px = heo3(icol,kk) * Ra * tz * xlwc
//           o3g =  xo3(icol,kk)/(1._r8+ px)

//           //-----------------------------------------------
//           //       ... Aqueous phase reaction rates
//           //           SO2 + H2O2 -> SO4
//           //           SO2 + O3   -> SO4
//           //-----------------------------------------------

//           //------------------------------------------------------------------------
//           //       ... S(IV) (HSO3) + H2O2
//           //------------------------------------------------------------------------
//           rah2o2 = 8.e4_r8 * exp( -3650._r8*t_factor )  &
//                / (.1_r8 + xph(icol,kk))

//           //------------------------------------------------------------------------
//           //        ... S(IV)+ O3
//           //------------------------------------------------------------------------
//           rao3   = 4.39e11_r8 * exp(-4131._r8/tz)  &
//                + 2.56e3_r8  * exp(-996._r8 /tz) /xph(icol,kk)

//           //-----------------------------------------------------------------
//           //       ... Prediction after aqueous phase
//           //       so4
//           //       When Cloud is present
//           //
//           //       S(IV) + H2O2 = S(VI)
//           //       S(IV) + O3   = S(VI)
//           //
//           //       reference:
//           //           (1) Seinfeld
//           //           (2) Benkovitz
//           //-----------------------------------------------------------------

//           //............................
//           //       S(IV) + H2O2 = S(VI)
//           //............................

//           if (xlwc >= small_value_lwc) then    //// WHEN CLOUD IS PRESENTED

//              call calc_sox_aqueous( modal_aerosols,     & // in
//                 rah2o2, h2o2g, so2g,   o3g,   rao3,     & // in
//                 patm, dtime, t_factor, xlwc,  const0,   & // in
//                 xhnm(icol,kk), heo3(icol,kk), heso2(icol,kk),      & // in
//                 xso2(icol,kk), xso4(icol,kk),           & // inout
//                 xso4_init(icol,kk), xh2o2(icol,kk),     & // inout
//                 xdelso4hp(icol,kk)                      ) // out

//           endif //// WHEN CLOUD IS PRESENTED

//        enddo col_loop1
//     enddo ver_loop1

//     call sox_cldaero_update( &
//          ncol, lchnk, loffset, dtime, mbar, pdel, press, tfld, & // in
//          cldnum, cldfrc, cfact, cldconc%xlwc, & // in
//          xdelso4hp, xh2so4, xso4, xso4_init,  & // in
//          qcw, qin  ) // inout

//     // diagnose variable
//     xphlwc(:,:) = 0._r8
//     do kk = 1, pver
//        do icol = 1, ncol
//           if (cldfrc(icol,kk)>=small_value_cf .and. lwc(icol,kk)>=small_value_lwc) then
//              xphlwc(icol,kk) = -1._r8*log10(xph(icol,kk)) * lwc(icol,kk)
//           endif
//        enddo
//     enddo
//     call outfld( 'XPH_LWC', xphlwc(:ncol,:), ncol , lchnk )

//     call sox_cldaero_destroy_obj(cldconc)

//   end subroutine setsox
#if 0

//===========================================================================
  subroutine calc_ph_values(                    &
                temperature, patm, xlwc, t_factor,   & // in
                xso2, xso4, xhnm,  so4_fact,    & // in
                Ra,   xkw,  const0,             & // in
                converged, xph                  ) // out
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
    implicit none

    real(r8),  intent(in) :: temperature        // temperature [K]
    real(r8),  intent(in) :: patm               // pressure [atm]
    real(r8),  intent(in) :: t_factor           // working variable to convert to 25 degC (1/T - 1/[298K])
    real(r8),  intent(in) :: xso2               // SO2 [mol/mol]
    real(r8),  intent(in) :: xso4               // SO4 [mol/mol]
    real(r8),  intent(in) :: xhnm               // [#/cm3]
    real(r8),  intent(in) :: xlwc               // in-cloud LWC [kg/L]
    real(r8),  intent(in) :: so4_fact           // factor for SO4
    real(r8),  intent(in) :: Ra                 // constant parameter
    real(r8),  intent(in) :: xkw                // constant parameter
    real(r8),  intent(in) :: const0             // constant parameter

    logical,  intent(out) :: converged          // if the method converge
    real(r8), intent(out) :: xph                // H+ ions concentration [mol/L]


    // local variables
    integer   :: iter  // iteration number
    real(r8)  :: yph_lo, yph_hi, yph    // pH values, lower and upper bounds
    real(r8)  :: ynetpos_lo, ynetpos_hi // lower and upper bounds of ynetpos
    real(r8)  :: xk, xe, x2     // output parameters in Henry's law
    real(r8)  :: fact1_so2, fact2_so2, fact3_so2, fact4_so2  // SO2 factors
    real(r8)  :: Eh2o, Eco2, Eso4 // effects of species [1/cm3]
    real(r8)  :: ynetpos        // net positive ions

    integer,  parameter :: itermax = 20  // maximum number of iterations
    real(r8), parameter :: co2g = 330.e-6_r8    //330 ppm = 330.e-6 atm
#include "../yaml/mo_setsox/f90_yaml/calc_ph_values_beg_yml.f90"


    //----------------------------------------
    // effect of chemical species
    //----------------------------------------

    // -------------- hno3 -------------------
    // FORTRAN refactoring: not incorporated in MAM4

    // -------------- nh3 -------------------
    // FORTRAN refactoring: not incorporated in MAM4

    // -------------- so2 -------------------
    // previous code
    //    heso2(i,k)  = xk*(1._r8 + xe/xph(i,k)*(1._r8 + x2/xph(i,k)))
    //    px = heso2(i,k) * Ra * tz * xl
    //    so2g =  xso2(i,k)/(1._r8+ px)
    //    Eso2 = xk*xe*so2g *patm
    // equivalent new code
    //    heso2 = xk + xk*xe/hplus * xk*xe*x2/hplus**2
    //    so2g = xso2/(1 + px)
    //         = xso2/(1 + heso2*ra*tz*xl)
    //         = xso2/(1 + xk*ra*tz*xl*(1 + (xe/hplus)*(1 + x2/hplus))
    //    eso2 = so2g*xk*xe*patm
    //          = xk*xe*patm*xso2/(1 + xk*ra*tz*xl*(1 + (xe/hplus)*(1 + x2/hplus))
    //          = ( fact1_so2    )/(1 + fact2_so2 *(1 + (fact3_so2/hplus)*(1 + fact4_so2/hplus)
    //    [hso3-] + 2*[so3--] = (eso2/hplus)*(1 + 2*x2/hplus)
    call henry_factor_so2(t_factor,     & // in
                        xk, xe, x2      ) // out
    fact1_so2 = xk*xe*patm*xso2
    fact2_so2 = xk*Ra*temperature*xlwc
    fact3_so2 = xe
    fact4_so2 = x2

    // -------------- h2o effects -------------------
    Eh2o = xkw

    // -------------- co2 effects -------------------
    call henry_factor_co2(t_factor,     & // in
                        xk, xe          ) // out
    Eco2 = xk*xe*co2g  *patm

    // -------------- so4 effects -------------------
    Eso4 = xso4*xhnm   &         // /cm3(a)
               *const0/xlwc

    //-----------------------------------------------------------------
    // now use bisection method to solve electro-neutrality equation
    // to calculate PH value and H+ concentration
    //
    // during the iteration loop,
    //    yph_lo = lower ph value that brackets the root (i.e., correct ph)
    //    yph_hi = upper ph value that brackets the root (i.e., correct ph)
    //    yph    = current ph value
    //    yposnet_lo and yposnet_hi = net positive ions for
    //       yph_lo and yph_hi
    //-----------------------------------------------------------------

    converged = .false.
    // ---------  1st iteration: set lower bound ph value ----------
    yph_lo = 2.0_r8
    yph_hi = yph_lo
    yph = yph_lo
    call calc_ynetpos (          yph,                   & // in
         fact1_so2,   fact2_so2, fact3_so2, fact4_so2,  & // in
         Eco2,  Eh2o, Eso4,      so4_fact,              & // in
         xph,   ynetpos                                 ) // out
    if (ynetpos <= 0.0_r8) then
    // the lower and upper bound ph values (2.0 and 7.0) do not bracket
    //    the correct ph, so use the lower bound
          converged = .true.
          return
    endif
    ynetpos_lo = ynetpos

    // ---------  2nd iteration: set upper bound ph value ----------
    yph_hi = 7.0_r8
    yph = yph_hi
    call calc_ynetpos (          yph,                   & // in
         fact1_so2,   fact2_so2, fact3_so2, fact4_so2,  & // in
         Eco2,  Eh2o, Eso4,      so4_fact,              & // in
         xph,   ynetpos                                 ) // out
    if (ynetpos >= 0.0_r8) then
    // the lower and upper bound ph values (2.0 and 7.0) do not bracket
    //    the correct ph, so use the lower bound
          converged = .true.
          return
    endif
    ynetpos_hi = ynetpos

    // --------- 3rd iteration and more ------------
    do iter = 3, itermax
        yph = 0.5_r8*(yph_lo + yph_hi)
        call calc_ynetpos (          yph,                   & // in
             fact1_so2,   fact2_so2, fact3_so2, fact4_so2,  & // in
             Eco2,  Eh2o, Eso4,      so4_fact,              & // in
             xph,   ynetpos                                 ) // out
        if (ynetpos >= 0.0_r8) then
           // net positive ions are >= 0 for both yph and yph_lo
           //    so replace yph_lo with yph
           yph_lo = yph
           ynetpos_lo = ynetpos
        else
           // net positive ions are <= 0 for both yph and yph_hi
           //    so replace yph_hi with yph
           yph_hi = yph
           ynetpos_hi = ynetpos
        endif

        if (abs(yph_hi - yph_lo) .le. 0.005_r8) then
           // |yph_hi - yph_lo| <= convergence criterion, so set
           //    final ph to their midpoint and exit
           // (.005 absolute error in pH gives .01 relative error in H+)
           yph = 0.5_r8*(yph_hi + yph_lo)
           xph = 10.0_r8**(-yph)
           converged = .true.
#include "../yaml/mo_setsox/f90_yaml/calc_ph_values_end_yml.f90"
           return
        endif
    enddo

  end subroutine calc_ph_values


//===========================================================================
  subroutine calc_ynetpos(   yph,                               & // in
                fact1_so2, fact2_so2, fact3_so2, fact4_so2,     & // in
                Eco2,      Eh2o,      Eso4,      so4_fact,      & // in
                xph,       ynetpos                              ) // out
    //-----------------------------------------------------------------
    // calculate net positive ions (ynetpos) for iterations in calc_ph_values
    // also calculate H+ concentration (xph) from ph value
    //-----------------------------------------------------------------
    implicit none

    real(r8), intent(in) :: yph         // pH value
    real(r8), intent(in) :: fact1_so2, fact2_so2, fact3_so2, fact4_so2 // factors for SO2
    real(r8), intent(in) :: Eco2, Eh2o, Eso4, so4_fact          // effects from species [1/cm3]

    real(r8), intent(out) :: xph        // H+ concentration from pH value [#/cm3]
    real(r8), intent(out) :: ynetpos    // net positive ions

    // local variables
    real(r8) :: Eso2    // effect of so2, which is related to pH value
    real(r8) :: tmp_hso3, tmp_so3, tmp_hco3, tmp_oh, tmp_so4  // temporary variables
    real(r8) :: tmp_pos, tmp_neg        // positive and negative values to calculate ynetpos
#include "../yaml/mo_setsox/f90_yaml/calc_ynetpos_beg_yml.f90"

    // calc current [H+] from ph
    xph = 10.0_r8**(-yph)

    //-----------------------------------------------------------------
    //          ... so2
    //-----------------------------------------------------------------
    Eso2 = fact1_so2/(1.0_r8 + fact2_so2*(1.0_r8 +(fact3_so2/xph) &
                    *(1.0_r8 + fact4_so2/xph)))

    tmp_hso3 = Eso2 / xph
    tmp_so3  = tmp_hso3 * 2.0_r8*fact4_so2/xph
    tmp_hco3 = Eco2 / xph
    tmp_oh   = Eh2o / xph
    tmp_so4 = so4_fact*Eso4

    // positive ions are H+ only
    tmp_pos = xph
    // all negative ions
    tmp_neg = tmp_oh + tmp_hco3 + tmp_hso3 + tmp_so3 + tmp_so4

    ynetpos = tmp_pos - tmp_neg

#include "../yaml/mo_setsox/f90_yaml/calc_ynetpos_end_yml.f90"
  end subroutine calc_ynetpos

//===========================================================================
  subroutine calc_sox_aqueous( modal_aerosols,         & // in
                rah2o2, h2o2g, so2g, o3g,      rao3,   & // in
                patm, dtime, t_factor, xlwc, const0,   & // in
                xhnm, heo3,  heso2,                    & // in
                xso2, xso4,  xso4_init, xh2o2,         & // inout
                xdelso4hp_ik                           ) // out
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
    implicit none

    logical,  intent(in) :: modal_aerosols      // if using MAM
    real(r8), intent(in) :: rah2o2      // reaction rate with h2o2
    real(r8), intent(in) :: rao3        // reaction rate with o3
    real(r8), intent(in) :: h2o2g, so2g, o3g
    real(r8), intent(in) :: patm        // pressure [atm]
    real(r8), intent(in) :: dtime       // time step [s]
    real(r8), intent(in) :: t_factor    // working variables to convert temperature
    real(r8), intent(in) :: xlwc        // in-cloud LWC [kg/L]
    real(r8), intent(in) :: const0
    real(r8), intent(in) :: xhnm
    real(r8), intent(in) :: heo3, heso2 // henry law constant
    real(r8), intent(inout) :: xso2, xso4, xso4_init, xh2o2 // mixing ratios
    real(r8), intent(out) :: xdelso4hp_ik // change of so4 in (i,k)

    // local variables
    real(r8) :: pso4    // production rate of so4
    real(r8) :: delta_s // so4 production in the time step
    real(r8) :: xk_so2, xk_h2o2 // for the use of Henry Law subroutines
    real(r8) :: xe, x2  // output of henry law subroutines but not used
    real(r8), parameter :: small_value_20 = 1.e-20_r8 // small value
    real(r8), parameter :: small_value_30 = 1.e-30_r8 // small value
#include "../yaml/mo_setsox/f90_yaml/calc_sox_aqueous_beg_yml.f90"

          //............................
          //       S(IV) + H2O2 = S(VI)
          //............................

// FORTRAN refactor: using henry law subroutines here break the BFB test because
// it changes the order of calculation. The original code is used now but new
// code is kept but commented out for future implementation.
//    call henry_factor_so2(t_factor, xk_so2, xe, x2)
//    call henry_factor_h2o2(t_factor, xk_h2o2, xe)
//    pso4 = rah2o2 * xk_h2o2* h2o2g * patm * xk_so2 * so2g * patm
    pso4 = rah2o2 * 7.4e4_r8*exp(6621._r8*t_factor) * h2o2g * patm &
                  * 1.23_r8 *exp(3120._r8*t_factor) * so2g * patm

    pso4 = pso4   & // [M/s] = [mole/L(w)/s]
         * xlwc   & // [mole/L(a)/s]
         / const0 & // [/L(a)/s]
         / xhnm


    delta_s = max(pso4*dtime, small_value_30)

    xso4_init=xso4

    if (delta_s<=xso2 .and. delta_s<=xh2o2) then
        xso4  = xso4  + delta_s
        xh2o2 = xh2o2 - delta_s
        xso2  = xso2  - delta_s
    elseif (xh2o2 > xso2) then
        xso4=xso4+xso2
        xh2o2=xh2o2-xso2
        xso2=small_value_20
    else
        xso4=xso4+xh2o2
        xso2=xso2-xh2o2
        xh2o2=small_value_20
    endif

    if (modal_aerosols) then
       xdelso4hp_ik  =  xso4 - xso4_init
    endif
             //...........................
             //       S(IV) + O3 = S(VI)
             //...........................

    pso4 = rao3 * heo3*o3g*patm * heso2*so2g*patm  // [M/s]

    pso4 = pso4        &                                // [M/s] =[mole/L(w)/s]
         * xlwc        &                                // [mole/L(a)/s]
         / const0      &                                // [/L(a)/s]
         / xhnm                                    // [mixing ratio/s]

    delta_s = max(pso4*dtime, small_value_30)

    xso4_init=xso4

    if (delta_s > xso2) then
       xso4 = xso4 + xso2
       xso2 = small_value_20
    else
       xso4 = xso4 + delta_s
       xso2 = xso2 - delta_s
    endif

#include "../yaml/mo_setsox/f90_yaml/calc_sox_aqueous_end_yml.f90"

  end subroutine calc_sox_aqueous

//===========================================================================
  subroutine henry_factor_so2(t_factor, xk, xe, x2)
    //-----------------------------------------------------------------
    // get Henry Law parameters xk, xe and x2 for SO2
    //-----------------------------------------------------------------

    implicit none

    real(r8), intent(in) :: t_factor    // temperature conversion factor
                                        // t_factor = (1/T - 1/298K)
    real(r8), intent(out) :: xk, xe, x2 // output variables

    // for so2
    xk = 1.23_r8  *exp( 3120._r8*t_factor )
    xe = 1.7e-2_r8*exp( 2090._r8*t_factor )
    x2 = 6.0e-8_r8*exp( 1120._r8*t_factor )

  end subroutine henry_factor_so2

//===========================================================================
  subroutine henry_factor_co2(t_factor, xk, xe)
    //-----------------------------------------------------------------
    // get Henry Law parameters xk and xe for CO2
    //-----------------------------------------------------------------

    implicit none

    real(r8), intent(in) :: t_factor    // temperature conversion factor
                                        // t_factor = (1/T - 1/298K)
    real(r8), intent(out) :: xk, xe     // output variables

    // for co2
    xk = 3.1e-2_r8*exp( 2423._r8*t_factor )
    xe = 4.3e-7_r8*exp(-913._r8 *t_factor )

  end subroutine henry_factor_co2

//===========================================================================
  subroutine henry_factor_h2o2(t_factor, xk, xe)
    //-----------------------------------------------------------------
    // get Henry Law parameters xk and xe for H2O2
    //-----------------------------------------------------------------

    implicit none

    real(r8), intent(in) :: t_factor    // temperature conversion factor
                                        // t_factor = (1/T - 1/298K)
    real(r8), intent(out) :: xk, xe     // output variables

    // for h2o2
    xk = 7.4e4_r8   *exp( 6621._r8*t_factor )
    xe = 2.2e-12_r8 *exp(-3730._r8*t_factor )

  end subroutine henry_factor_h2o2

//===========================================================================
  subroutine henry_factor_o3(t_factor, xk)
    //-----------------------------------------------------------------
    // get Henry Law parameters xk and xe for O3
    //-----------------------------------------------------------------

    implicit none

    real(r8), intent(in) :: t_factor    // temperature conversion factor
                                        // t_factor = (1/T - 1/298K)
    real(r8), intent(out) :: xk         // output variables

    // for o3
    xk = 1.15e-2_r8   *exp( 2560._r8*t_factor )

  end subroutine henry_factor_o3

//===========================================================================


#endif
} // namespace mo_setsox
} // namespace mam4
#endif
