#ifndef MAM4XX_MO_SETHET_HPP
#define MAM4XX_MO_SETHET_HPP

#include <haero/math.hpp>
#include <mam4xx/gas_chem.hpp>
#include <mam4xx/mam4_types.hpp>
#include <mam4xx/utils.hpp>

namespace mam4 {

namespace mo_sethet {

const int ktop = ConvProc::Config::ktop; //int ktop = 47; // BAD_CONSTANT: only true for nlev == 72
constexpr Real avo = haero::Constants::avogadro;
const Real pi = haero::Constants::pi;
constexpr Real rgrav =
    mo_chm_diags::rgrav; // reciprocal of acceleration of gravity ~ m/s^2
constexpr const int gas_pcnst = gas_chemistry::gas_pcnst;
const Real boltz_cgs = haero::Constants::boltzmann * 1.e7; // erg/K
// number of vertical levels
constexpr const int pver = mam4::nlev;

//FIXME: BAD CONSTANT
constexpr const Real mass_h2o = 18.0;           // mass of water vapor [amu]
constexpr const Real cm3_2_m3 = 1.0e-6;         // convert cm^3 to m^3
constexpr const Real liter_per_gram = 1.0e-3;
constexpr const Real avo2  = avo * liter_per_gram * cm3_2_m3; // [liter/gm/mol*(m/cm)^3]

using Real = haero::Real;
using View1D = DeviceType::view_1d<Real>;

//=================================================================================
KOKKOS_INLINE_FUNCTION 
void calc_het_rates(Real satf, // saturation fraction in cloud //in
                    Real rain, // rain rate [molecules/cm^3/s] //in
                    Real xhen,  // henry's law constant // in
                    Real tmp_hetrates, Real work1, Real work2,    // in
                    Real het_rates) // rainout loss rates [1/s]// out
{
  //-----------------------------------------------------------------
  // calculate het_rates
  // input arguments are different for different species
  //-----------------------------------------------------------------
    Real work3;
    Real h2o_mol  = 1.0e3 / mass_h2o; // [gm/mol water]
    // FIXME: BAD CONSTANT should haero::Constants::molec_weight_h2o be used somewhere here instead?

    work3 = satf *  max( rain / (h2o_mol*(work1 + 1.0 / (xhen * work2))), 0.0 );
    het_rates =  work3 + tmp_hetrates;

} //end calc_het_rates
//=================================================================================

//=================================================================================
KOKKOS_INLINE_FUNCTION 
void calc_precip_rescale(ColumnView cmfdqr,  // dq/dt for convection [kg/kg/s] //in
                         ColumnView nrain,   // stratoform precip [kg/kg/s] //in
                         ColumnView nevapr,  // evaporation [kg/kg/s] // in
                         ColumnView precip) {   // precipitation [kg/kg/s] // out
  // -----------------------------------------------------------------------
  // calculate precipitation rate at each grid
  // this is added to rescale the variable precip (which can only be positive)
  // to the actual vertical integral of positive and negative values. 
  // This removes point storms
  // -----------------------------------------------------------------------

    Real total_rain;      // total rain rate (both pos and neg) in the column
    Real total_pos;       // total positive rain rate in the column

   total_rain = 0.0;
   total_pos  = 0.0;
   for(int kk = 0; kk < pver; kk++) {
      precip(kk) = cmfdqr(kk) + nrain(kk) - nevapr(kk);
      total_rain = total_rain + precip(kk);
      if ( precip(kk) < 0.0 ) 
            precip(kk) = 0.0;
      total_pos  = total_pos  + precip(kk);
   }

   if ( total_rain <= 0.0 ) {
      for(int kk = 0; kk < pver; kk++) {
         precip(kk) = 0.0;       // set all levels to zero
      }
   } else {
      for(int kk = 0; kk < pver; kk++) {
         precip(kk) = precip(kk) * total_rain / total_pos;
      }   
   }

} //end subroutine calc_precip_rescale

//=================================================================================
KOKKOS_INLINE_FUNCTION
void gas_washout (int plev,         // calculate from this level below //in
                  Real xkgm,        // mass flux on rain drop //in
                  Real xliq_ik,     // liquid rain water content [gm/m^3] // in
                  ColumnView xhen_i, // henry's law constant 
                  ColumnView tfld_i, // temperature [K]
                  ColumnView delz_i, // layer depth about interfaces [cm]  // in
                  ColumnView xgas) {    // gas concentration // inout
   //------------------------------------------------------------------------
   // calculate gas washout by cloud if not saturated
   //------------------------------------------------------------------------
    //FIXME: BAD CONSTANTS
    Real allca = 0.0;   // total of ca between level plev and kk [#/cm3]
    Real xca, xeqca;
    Real const0 = boltz_cgs * 1.0e-6; // [atmospheres/deg k/cm^3]
    Real geo_fac  = 6.0;            // geometry factor (surf area/volume = geo_fac/diameter)
    Real xrm = .189;             // mean diameter of rain drop [cm]
    Real xum = 748.0;             // mean rain drop terminal velocity [cm/s]

     //-----------------------------------------------------------------
     //       ... calculate the saturation concentration eqca
     //-----------------------------------------------------------------
     for(int kk = 0; kk < pver; kk++) {      // cal washout below cloud
        xeqca =  xgas(kk) / (xliq_ik*avo2 + 1.0/(xhen_i(kk)*const0*tfld_i(kk))) * xliq_ik*avo2;

        //-----------------------------------------------------------------
        //       ... calculate ca; inside cloud concentration in  #/cm3(air)
        //-----------------------------------------------------------------
        xca = geo_fac*xkgm*xgas(kk)/(xrm*xum)*delz_i(kk) * xliq_ik * cm3_2_m3;

        //-----------------------------------------------------------------
        //       ... if is not saturated (take hno3 as an example)
        //               hno3(gas)_new = hno3(gas)_old - hno3(h2o)
        //           otherwise
        //               hno3(gas)_new = hno3(gas)_old
        //-----------------------------------------------------------------
        allca = allca + xca;
        if( allca < xeqca ) {
           xgas(kk) = max( xgas(kk) - xca, 0.0 );
        }
     }

} //end subroutine gas_washout

//=================================================================================
KOKKOS_INLINE_FUNCTION
void find_ktop(int ncol,  
               Real rlat,  // latitude in radians for columns
               ColumnView press, // pressure [Pa] // in
               int& ktop) { // index that only calculate het_rates above this level //out 
  //---------------------------------------------------------------------------
  // -------- find the top level that het_rates are set as 0 above it ---------
  //---------------------------------------------------------------------------

   Real p_limit;     // pressure limit [Pa]
   Real d2r = pi/180.0;   // degree to radian

   if ( haero::abs(rlat) > 60.0*d2r ) {
      p_limit = 300.0e2   // 300hPa for high latitudes
   } else {
      p_limit = 100.0e2   // 100hPa for low latitudes
   }

   for(int kk = pver-1; kk >= 0; kk--) {
      if( press(kk) < p_limit ) {
         ktop(icol) = kk
         return 
      }
   } // k_loop

} // end subroutine find_ktop

/*
KOKKOS_INLINE_FUNCTION
void sethet(const ColumnView het_rates[gas_pcnst], //[pver][gas_pcnst],  rainout loss rates [1/s] //out
            ColumnView press,           // pressure [pascals] //in
            ColumnView zmid,            // midpoint geopot [km]  //in
            Real phis,                  // surf geopotential //in
            ColumnView tfld,            // temperature [K]  //in
            ColumnView cmfdqr,          // dq/dt for convection [kg/kg/s] //in
            ColumnView nrain,           // stratoform precip [kg/kg/s] //in
            ColumnView nevapr,          // evaporation [kg/kg/s] //in
            Real delt,                  // time step [s] //in
            ColumnView xhnm,            // total atms density [cm^-3] //in
            ColumnView qin[gas_pcnst],  // xported species [vmr]  //in
            int lchnk                   // chunk index //in
            ) {

    //-----------------------------------------------------------------------      
    //       ... compute rainout loss rates (1/s)
    //-----------------------------------------------------------------------      

    use phys_grid,    only : get_rlat_all_p
    use cam_abortutils,   only : endrun

    //-----------------------------------------------------------------------      
    //       ... local variables       //FIXME: BAD CONSTANT
    //-----------------------------------------------------------------------       
    Real xrm   = .189;             // mean diameter of rain drop [cm]
    Real xum   = 748.0;             // mean rain drop terminal velocity [cm/s]
    Real xvv   = 6.18e-2;          // kinetic viscosity [cm^2/s]
    Real xdg   = .112;             // mass transport coefficient [cm/s]
    Real t0    = 298.0;             // reference temperature [K]
    Real xph0  = 1.0e-5;            // cloud [h+]
    Real satf_hno3  = .016;        // saturation factor for hno3 in clouds 
    Real satf_h2o2  = .016;        // saturation factor for h2o2 in clouds 
    Real satf_so2   = .016;        // saturation factor for so2 in clouds 
    Real const0   = boltz_cgs * 1.0e-6; // [atmospheres/deg k/cm^3]
    Real hno3_diss = 15.4;         // hno3 dissociation constant
    Real mass_air = 29.0;           // mass of background atmosphere [amu]
    Real km2cm    = 1.0e5;          // convert km to cm
    Real m2km     = 1.0e-3;         // convert m to km
    Real m3_2_cm3 = 1.0e6;          // convert m^3 to cm^3
    Real MISSING = -999999.0;
    Real large_value_lifetime = 1.0e29;  // a large lifetime value if no washout

    character(len=3) :: hetratestrg
    int ktop;     // tropopause level, 100mb for lat < 60 and 300mb for lat > 60
    int ktop_all;
    Real xkgm;           // mass flux on rain drop
    Real stay;          // fraction of layer traversed by falling drop in timestep delt
    Real xdtm;           // the traveling time in each dz [s]
    Real xxx2, xxx3;     // working variables for h2o2 (2) and so2 (3)
    Real yso2, yh2o2;    // washout lifetime [s]     
    Real rlat;    // latitude in radians for columns
    Real work1, work2;   // working variables
    Real t_factor;      // temperature factor to calculate henry's law parameters
    Real xk0;           // working variable
    Real zsurf;         // surface height [km]
    Real so2_diss;     // so2 dissociation constant
    ColumnView xgas2, xgas3  // gas phase species for h2o2 (2) and so2 (3) [molecules/cm^3]
    ColumnView delz;               // layer depth about interfaces [cm]
    ColumnView xh2o2;              // h2o2 concentration [molecules/cm^3]
    ColumnView xso2;               // so2 concentration [molecules/cm^3]
    ColumnView xliq;               // liquid rain water content in a grid cell [gm/m^3]
    ColumnView rain;               // precipitation (rain) rate [molecules/cm^3/s]
    ColumnView precip;             // precipitation rate [kg/kg/s]
    ColumnView xhen_h2o2, xhen_hno3, xhen_so2;   // henry law constants
    real(r8), dimension(ncol,pver,8) :: tmp_hetrates
#include "../yaml/mo_sethet/f90_yaml/sethet_beg_yml.f90"


    //-----------------------------------------------------------------
    //        note: the press array is in pascals and must be
    //              mutiplied by 10 to yield dynes/cm**2.
    //-----------------------------------------------------------------
    //       ... set wet deposition for
    //           1. h2o2         2. hno3
    //           3. ch2o         4. ch3ooh
    //           5. pooh         6. ch3coooh
    //           7. ho2no2       8. onit
    //           9. mvk         10. macr
    //          11. c2h5ooh     12. c3h7ooh
    //          13. rooh        14. ch3cocho
    //          15. pb          16. macrooh
    //          17. xooh        18. onitr
    //          19. isopooh     20. ch3oh
    //          21. c2h5oh      22. glyald
    //          23. hyac        24. hydrald
    //          25. ch3cho      26. isopno3
    //-----------------------------------------------------------------
    // FORTRAN refactor note: current MAM4 only have three species in default:
    // 'H2O2','H2SO4','SO2'.  Options for other species are then removed
    //-----------------------------------------------------------------


    het_rates(:,:,:) = 0.

    if ( .not. do_wetdep) return

    call get_rlat_all_p(lchnk, ncol, rlat)

    do mm = 1,gas_wetdep_cnt
       mm2 = wetdep_map(mm)
       if ( mm2>0 ) then
          het_rates(:,:,mm2) = MISSING
       endif
    enddo

    //-----------------------------------------------------------------
    //	... the 2 and .6 multipliers are from a formula by frossling (1938)
    //-----------------------------------------------------------------
    xkgm = xdg/xrm * 2. + xdg/xrm * .6 * sqrt( xrm*xum/xvv ) * (xvv/xdg)**(1./3.) 

    //-----------------------------------------------------------------
    //	... Find the level index that only calculate het_rates below
    //-----------------------------------------------------------------
    call find_ktop( ncol,  rlat,  press,  & // in
                    ktop                  ) // out
    ktop_all = minval( ktop(:) )

    // this is added to rescale the variable precip (which can only be positive)
    // to the actual vertical integral of positive and negative values.  This
    // removes point storms
    call calc_precip_rescale( ncol, cmfdqr, nrain, nevapr,  & // in
                              precip                        ) // out

    do kk = 1,pver
       rain(:ncol,kk)   = mass_air*precip(:ncol,kk)*xhnm(:ncol,kk) / mass_h2o
       xliq(:ncol,kk)   = precip(:ncol,kk) * delt * xhnm(:ncol,kk) / avo*mass_air * m3_2_cm3
       xh2o2(:ncol,kk)  = qin(:ncol,kk,spc_h2o2_ndx) * xhnm(:ncol,kk)
       xso2(:ncol,kk)  = qin(:ncol,kk,spc_so2_ndx) * xhnm(:ncol,kk)
    enddo

    zsurf(:ncol) = m2km * phis(:ncol) * rgrav
    do kk = ktop_all,pver-1
       delz(:ncol,kk) = abs( (zmid(:ncol,kk) - zmid(:ncol,kk+1))*km2cm ) 
    enddo
    delz(:ncol,pver) = abs( (zmid(:ncol,pver) - zsurf(:ncol) )*km2cm ) 

    //-----------------------------------------------------------------
    //       ... part 0b,  for temperature dependent of henrys
    //                     xxhe1 = henry con for hno3
    //                     xxhe2 = henry con for h2o2
    //lwh 10/00 -- take henry''s law constants from brasseur et al. [1999],
    //             appendix j. for hno3, also consider dissociation to
    //             get effective henry''s law constant; equilibrium
    //             constant for dissociation from brasseur et al. [1999],
    //             appendix k. assume ph=5 (set as xph0 above).
    //             heff = h*k/[h+] for hno3 (complete dissociation)
    //             heff = h for h2o2 (no dissociation)
    //             heff = h * (1 + k/[h+]) (in general)
    //-----------------------------------------------------------------
    do kk = ktop_all,pver
       //-----------------------------------------------------------------
       // 	... effective henry''s law constants:
       //	hno3, h2o2  (brasseur et al., 1999)
       //-----------------------------------------------------------------
       // temperature factor
       t_factor(:ncol) = (t0 - tfld(:ncol,kk))/(t0*tfld(:ncol,kk))
       xhen_h2o2(:,kk)     = 7.45e4 * exp( 6620. * t_factor(:) )
       // HNO3, for calculation of H2SO4 het rate use
       xk0(:)             = 2.1e5 *exp( 8700.*t_factor(:) )
       xhen_hno3(:,kk)     = xk0(:) * ( 1. + hno3_diss / xph0 )
       // SO2
       xk0(:)             = 1.23 * exp( 3120. * t_factor(:) )
       so2_diss(:)        = 1.23e-2 * exp( 1960. * t_factor(:) )
       xhen_so2(:,kk)   = xk0(:) * ( 1. + so2_diss(:) / xph0 )

       // initiate temporary array
       tmp_hetrates(:,kk,:) = 0.
    enddo

    //-----------------------------------------------------------------
    //       ... part 1, solve for high henry constant ( hno3, h2o2)
    //-----------------------------------------------------------------
    col_loop :  do icol = 1,ncol
       xgas2(:) = xh2o2(icol,:)                     // different levels wash 
       xgas3(:) = xso2 (icol,:)
       level_loop1  : do kk = ktop(icol),pver
          stay = 1.
          if( rain(icol,kk) /= 0. ) then            // finding rain cloud           
             stay = ((zmid(icol,kk) - zsurf(icol))*km2cm)/(xum*delt)
             stay = min( stay,1. )
             // calculate gas washout by cloud
             call gas_washout( kk,  xkgm,   xliq(icol,kk),       & // in
                  xhen_h2o2(icol,:), tfld(icol,:), delz(icol,:), & // in
                  xgas2                                          ) // inout
             call gas_washout( kk,  xkgm,   xliq(icol,kk),       & // in
                  xhen_so2(icol,:), tfld(icol,:), delz(icol,:),  & // in
                  xgas3                                          ) // inout
          endif
          //-----------------------------------------------------------------
          //       ... calculate the lifetime of washout (second)
          //             after all layers washout 
          //             the concentration of hno3 is reduced 
          //             then the lifetime xtt is calculated by
          //
          //                  xtt = (xhno3(ini) - xgas1(new))/(dt*xhno3(ini))
          //                  where dt = passing time (s) in vertical
          //                             path below the cloud
          //                        dt = dz(cm)/um(cm/s)
          //-----------------------------------------------------------------
          xdtm = delz(icol,kk) / xum                     // the traveling time in each dz

          xxx2 = (xh2o2(icol,kk) - xgas2(kk))
          if( xxx2 /= 0. ) then                       // if no washout lifetime = 1.e29
             yh2o2  = xh2o2(icol,kk)/xxx2 * xdtm     
          else
             yh2o2  = large_value_lifetime
          endif
          tmp_hetrates(icol,kk,2) = max( 1. / yh2o2,0. ) * stay

          xxx3 = (xso2( icol,kk) - xgas3(kk))
          if( xxx3 /= 0. ) then                       // if no washout lifetime = 1.e29
             yso2  = xso2( icol,kk)/xxx3 * xdtm     
          else
             yso2  = large_value_lifetime
          endif
          tmp_hetrates(icol,kk,3) = max( 1. / yso2, 0. ) * stay

       enddo level_loop1
    enddo col_loop

    //-----------------------------------------------------------------
    //       ... part 2, in-cloud solve for low henry constant
    //                   hno3 and h2o2 have both in and under cloud
    //-----------------------------------------------------------------
    level_loop2 : do kk = ktop_all,pver
       Column_loop2 : do icol=1,ncol
          if ( rain(icol,kk) <= 0. ) then
             het_rates(icol,kk,:) =  0. 
             cycle
          endif

          work1 = avo2 * xliq(icol,kk)
          work2 = const0 * tfld(icol,kk)

          if( h2o2_ndx > 0 ) then
             call calc_het_rates(satf_h2o2, rain(icol,kk), xhen_h2o2(icol,kk),& // in
                        tmp_hetrates(icol,kk,2), work1, work2,& // in
                        het_rates(icol,kk,h2o2_ndx)) // out
          endif

          if ( prog_modal_aero .and. so2_ndx>0 .and. h2o2_ndx>0 ) then
             het_rates(icol,kk,so2_ndx) = het_rates(icol,kk,h2o2_ndx)
          elseif( so2_ndx > 0 ) then
             call calc_het_rates(satf_so2, rain(icol,kk), xhen_so2(icol,kk),  & // in
                        tmp_hetrates(icol,kk,3), work1, work2,& // in
                        het_rates(icol,kk,so2_ndx)) // out
          endif

          if( h2so4_ndx > 0 ) then
             call calc_het_rates(satf_hno3, rain(icol,kk), xhen_hno3(icol,kk), & // in
                        tmp_hetrates(icol,kk,1), work1, work2, & // in
                        het_rates(icol,kk,h2so4_ndx)) // out
          endif

       enddo Column_loop2
    enddo level_loop2

    //-----------------------------------------------------------------
    //	... Set rates above tropopause = 0.
    //-----------------------------------------------------------------
    do mm = 1,gas_wetdep_cnt
       mm2 = wetdep_map(mm)
       do icol = 1,ncol
          do kk = 1,ktop(icol)
             het_rates(icol,kk,mm2) = 0.
          enddo
       enddo
       if ( any( het_rates(:ncol,:,mm2) == MISSING) ) then
          write(hetratestrg,'(I3)') mm2
          call endrun('sethet: het_rates (wet dep) not set for het reaction number : '//hetratestrg)
       endif
    enddo
#include "../yaml/mo_sethet/f90_yaml/sethet_end_yml.f90"
} // sethet
*/

} // namespace mo_sethet
} // namespace mam4
#endif