#ifndef MAM4XX_MO_SETHET_HPP
#define MAM4XX_MO_SETHET_HPP

#include <haero/math.hpp>
#include <mam4xx/gas_chem.hpp>
#include <mam4xx/mam4_types.hpp>
#include <mam4xx/mo_chm_diags.hpp>
#include <mam4xx/utils.hpp>

namespace mam4 {

namespace mo_sethet {

constexpr Real avo = haero::Constants::avogadro;
const Real pi = haero::Constants::pi;
constexpr Real rga = 1.0 / haero::Constants::gravity;
constexpr int gas_pcnst = gas_chemistry::gas_pcnst;
const Real boltz_cgs = haero::Constants::boltzmann * 1.e7; // erg/K
// number of vertical levels
constexpr int pver = mam4::nlev;

// FIXME: BAD CONSTANT
// mass of water vapor [amu] //convert to g/mol from kg/mol
constexpr Real mass_h2o = haero::Constants::molec_weight_h2o * 1000;
constexpr Real cm3_2_m3 = 1.0e-6; // convert cm^3 to m^3
constexpr Real liter_per_gram = 1.0e-3;
constexpr Real avo2 =
    avo * liter_per_gram * cm3_2_m3; // [liter/gm/mol*(m/cm)^3]

using Real = haero::Real;
using View1D = DeviceType::view_1d<Real>;

//=================================================================================
KOKKOS_INLINE_FUNCTION
void calc_het_rates(const Real satf, // saturation fraction in cloud //in
                    const Real rain, // rain rate [molecules/cm^3/s] //in
                    const Real xhen, // henry's law constant // in
                    const Real tmp_hetrates, const Real work1,
                    const Real work2, // in
                    Real &het_rates)  // rainout loss rates [1/s]// out
{
  //-----------------------------------------------------------------
  // calculate het_rates
  // input arguments are different for different species
  //-----------------------------------------------------------------
  Real work3;
  Real h2o_mol = 1.0e3 / mass_h2o; // [gm/mol water]

  work3 =
      satf * haero::max(rain / (h2o_mol * (work1 + 1.0 / (xhen * work2))), 0.0);
  het_rates = work3 + tmp_hetrates;

} // end calc_het_rates
//=================================================================================

//=================================================================================
KOKKOS_INLINE_FUNCTION
void calc_precip_rescale(
    const ColumnView &cmfdqr,   // dq/dt for convection [kg/kg/s] //in
    const ColumnView &nrain,    // stratoform precip [kg/kg/s] //in
    const ColumnView &nevapr,   // evaporation [kg/kg/s] // in
    const ColumnView &precip) { // precipitation [kg/kg/s] // out
  // -----------------------------------------------------------------------
  // calculate precipitation rate at each grid
  // this is added to rescale the variable precip (which can only be positive)
  // to the actual vertical integral of positive and negative values.
  // This removes point storms
  // -----------------------------------------------------------------------

  Real total_rain; // total rain rate (both pos and neg) in the column
  Real total_pos;  // total positive rain rate in the column

  total_rain = 0.0;
  total_pos = 0.0;
  for (int kk = 0; kk < pver; kk++) {
    precip(kk) = cmfdqr(kk) + nrain(kk) - nevapr(kk);
    total_rain = total_rain + precip(kk);
    if (precip(kk) < 0.0)
      precip(kk) = 0.0;
    total_pos = total_pos + precip(kk);
  }

  if (total_rain <= 0.0) {
    for (int kk = 0; kk < pver; kk++) {
      precip(kk) = 0.0; // set all levels to zero
    }
  } else {
    for (int kk = 0; kk < pver; kk++) {
      precip(kk) = precip(kk) * total_rain / total_pos;
    }
  }

} // end subroutine calc_precip_rescale

//=================================================================================
KOKKOS_INLINE_FUNCTION
void gas_washout(
    const ThreadTeam &team,
    const int plev,           // calculate from this level below //in
    const Real xkgm,          // mass flux on rain drop //in
    const Real xliq_ik,       // liquid rain water content [gm/m^3] // in
    const ColumnView &xhen_i, // henry's law constant
    const ColumnView &tfld_i, // temperature [K]
    const ColumnView &delz_i, // layer depth about interfaces [cm]  // in
    const ColumnView &xgas) { // gas concentration // inout
  //------------------------------------------------------------------------
  // calculate gas washout by cloud if not saturated
  //------------------------------------------------------------------------
  // FIXME: BAD CONSTANTS
  Real allca = 0.0; // total of ca between level plev and kk [#/cm3]
  Real const0 = boltz_cgs * 1.0e-6; // [atmospheres/deg k/cm^3]
  Real geo_fac = 6.0; // geometry factor (surf area/volume = geo_fac/diameter)
  Real xrm = .189;    // mean diameter of rain drop [cm]
  Real xum = 748.0;   // mean rain drop terminal velocity [cm/s]
  Real xeqca = 0.0;
  Real xca = 0.0;

  // -----------------------------------------------------------------
  //       ... calculate the saturation concentration eqca
  // -----------------------------------------------------------------
  for (int k = plev; k < pver; k++) {
    xeqca = xgas(k) /
            (xliq_ik * avo2 + 1.0 / (xhen_i(k) * const0 * tfld_i(k))) *
            xliq_ik * avo2;
    //-----------------------------------------------------------------
    //       ... calculate ca; inside cloud concentration in  #/cm3(air)
    //-----------------------------------------------------------------
    xca =
        geo_fac * xkgm * xgas(k) / (xrm * xum) * delz_i(k) * xliq_ik * cm3_2_m3;

    // -----------------------------------------------------------------
    //       ... if is not saturated (take hno3 as an example)
    //               hno3(gas)_new = hno3(gas)_old - hno3(h2o)
    //           otherwise
    //               hno3(gas)_new = hno3(gas)_old
    // -----------------------------------------------------------------
    allca += xca;
    if (allca < xeqca) {
      xgas(k) = haero::max(xgas(k) - xca, 0.0);
    }
  }

} // end subroutine gas_washout

//=================================================================================
KOKKOS_INLINE_FUNCTION
void find_ktop(
    Real rlat,               // latitude in radians for columns
    const ColumnView &press, // pressure [Pa] // in
    int &ktop) { // index that only calculate het_rates above this level //out
  //---------------------------------------------------------------------------
  // -------- find the top level that het_rates are set as 0 above it ---------
  //---------------------------------------------------------------------------

  Real p_limit = 0;      // pressure limit [Pa]
  Real d2r = pi / 180.0; // degree to radian

  if (haero::abs(rlat) > 60.0 * d2r) {
    p_limit = 300.0e2; // 300hPa for high latitudes
  } else {
    p_limit = 100.0e2; // 100hPa for low latitudes
  }

  for (int kk = pver - 1; kk >= 0; kk--) {
    if (press(kk) < p_limit) {
      ktop = kk;
      return;
    }
  } // k_loop

} // end subroutine find_ktop

KOKKOS_INLINE_FUNCTION
void sethet(
    const ThreadTeam &team,
    const ColumnView
        het_rates[gas_pcnst], //[pver][gas_pcnst], rainout rates [1/s] //out
    const Real rlat,          // latitude in radians for columns
    const ColumnView &press,  // pressure [pascals] //in
    const ColumnView &zmid,   // midpoint geopot [km]  //in
    const Real phis,          // surf geopotential //in
    const ColumnView &tfld,   // temperature [K]  //in
    const ColumnView &cmfdqr, // dq/dt for convection [kg/kg/s] //in
    const ColumnView &nrain,  // stratoform precip [kg/kg/s] //in
    const ColumnView &nevapr, // evaporation [kg/kg/s] //in
    const Real delt,          // time step [s] //in
    const ColumnView &xhnm,   // total atms density [cm^-3] //in
    const ColumnView qin[gas_pcnst], // xported species [vmr]  //in
    // working variables
    const ColumnView
        &xgas2, // gas phase species for h2o2 (2) and so2 (3) [molecules/cm^3]
    const ColumnView
        &xgas3, // gas phase species for h2o2 (2) and so2 (3) [molecules/cm^3]
    const ColumnView &delz,  // layer depth about interfaces [cm]
    const ColumnView &xh2o2, // h2o2 concentration [molecules/cm^3]
    const ColumnView &xso2,  // so2 concentration [molecules/cm^3]
    const ColumnView &xliq, // liquid rain water content in a grid cell [gm/m^3]
    const ColumnView &rain, // precipitation (rain) rate [molecules/cm^3/s]
    const ColumnView &precip,    // precipitation rate [kg/kg/s]
    const ColumnView &xhen_h2o2, // henry law constants
    const ColumnView &xhen_hno3, // henry law constants
    const ColumnView &xhen_so2,  // henry law constants
    const ColumnView tmp_hetrates[gas_pcnst], const int spc_h2o2_ndx,
    const int spc_so2_ndx, const int h2o2_ndx, const int so2_ndx,
    const int h2so4_ndx, const int gas_wetdep_cnt, const int wetdep_map[3]) {

  //-----------------------------------------------------------------------
  //       ... compute rainout loss rates (1/s)
  //-----------------------------------------------------------------------

  //-----------------------------------------------------------------------
  //       ... local variables       //FIXME: BAD CONSTANT
  //-----------------------------------------------------------------------
  Real xrm = .189;                  // mean diameter of rain drop [cm]
  Real xum = 748.0;                 // mean rain drop terminal velocity [cm/s]
  Real xvv = 6.18e-2;               // kinetic viscosity [cm^2/s]
  Real xdg = .112;                  // mass transport coefficient [cm/s]
  Real t0 = 298.0;                  // reference temperature [K]
  Real xph0 = 1.0e-5;               // cloud [h+]
  Real satf_hno3 = .016;            // saturation factor for hno3 in clouds
  Real satf_h2o2 = .016;            // saturation factor for h2o2 in clouds
  Real satf_so2 = .016;             // saturation factor for so2 in clouds
  Real const0 = boltz_cgs * 1.0e-6; // [atmospheres/deg k/cm^3]
  Real hno3_diss = 15.4;            // hno3 dissociation constant
  Real mass_air = 29.0;             // mass of background atmosphere [amu]
  Real km2cm = 1.0e5;               // convert km to cm
  Real m2km = 1.0e-3;               // convert m to km
  Real m3_2_cm3 = 1.0e6;            // convert m^3 to cm^3
  Real MISSING = -999999.0;
  Real large_value_lifetime = 1.0e29; // a large lifetime value if no washout

  int ktop;  // tropopause level, 100mb for lat < 60 and 300mb for lat > 60
  Real xkgm; // mass flux on rain drop
  Real stay; // fraction of layer traversed by falling drop in timestep delt
  Real xdtm; // the traveling time in each dz [s]
  Real xxx2, xxx3;   // working variables for h2o2 (2) and so2 (3)
  Real yso2, yh2o2;  // washout lifetime [s]
  Real work1, work2; // working variables
  Real t_factor;     // temperature factor to calculate henry's law parameters
  Real xk0;          // working variable
  Real zsurf;        // surface height [km]
  Real so2_diss;     // so2 dissociation constant

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

  for (int mm = 0; mm < gas_pcnst; mm++) {
    for (int kk = 0; kk < pver; kk++) {
      het_rates[mm](kk) = 0.0;
    }
  }

  // if ( .not. do_wetdep) return

  for (int mm = 0; mm < gas_wetdep_cnt; mm++) {
    int mm2 = wetdep_map[mm];
    if (mm2 > 0) {
      for (int kk = 0; kk < pver; kk++) {
        het_rates[mm2](kk) = MISSING;
      }
    }
  }

  //-----------------------------------------------------------------
  //	... the 2 and .6 multipliers are from a formula by frossling (1938)
  //-----------------------------------------------------------------
  xkgm = xdg / xrm * 2.0 + xdg / xrm * .6 * haero::sqrt(xrm * xum / xvv) *
                               haero::pow((xvv / xdg), (1.0 / 3.0));

  //-----------------------------------------------------------------
  //	... Find the level index that only calculate het_rates below
  //-----------------------------------------------------------------
  find_ktop(rlat, press, ktop); // populate ktop

  // this is added to rescale the variable precip (which can only be positive)
  // to the actual vertical integral of positive and negative values.  This
  // removes point storms
  calc_precip_rescale(cmfdqr, nrain, nevapr, precip); // populate precip

  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team, pver), KOKKOS_LAMBDA(int kk) {
        rain(kk) = mass_air * precip(kk) * xhnm(kk) / mass_h2o;
        xliq(kk) = precip(kk) * delt * xhnm(kk) / avo * mass_air * m3_2_cm3;
        xh2o2(kk) = qin[spc_h2o2_ndx](kk) * xhnm(kk);
        xso2(kk) = qin[spc_so2_ndx](kk) * xhnm(kk);
      });

  zsurf = m2km * phis * rga;

  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team, ktop, pver - 1), KOKKOS_LAMBDA(int kk) {
        delz(kk) = haero::abs((zmid(kk) - zmid(kk + 1)) * km2cm);
      });
  delz(pver - 1) = haero::abs((zmid(pver - 1) - zsurf) * km2cm);

  //-----------------------------------------------------------------
  //       ... part 0b,  for temperature dependent of henrys
  //                     xxhe1 = henry con for hno3
  //                     xxhe2 = henry con for h2o2
  // lwh 10/00 -- take henry''s law constants from brasseur et al. [1999],
  //             appendix j. for hno3, also consider dissociation to
  //             get effective henry''s law constant; equilibrium
  //             constant for dissociation from brasseur et al. [1999],
  //             appendix k. assume ph=5 (set as xph0 above).
  //             heff = h*k/[h+] for hno3 (complete dissociation)
  //             heff = h for h2o2 (no dissociation)
  //             heff = h * (1 + k/[h+]) (in general)
  //-----------------------------------------------------------------
  for (int kk = ktop; kk < pver; kk++) {
    //-----------------------------------------------------------------
    // 	... effective henry''s law constants:
    //	hno3, h2o2  (brasseur et al., 1999)
    //-----------------------------------------------------------------
    // temperature factor
    t_factor = (t0 - tfld(kk)) / (t0 * tfld(kk));
    xhen_h2o2(kk) = 7.45e4 * haero::exp(6620.0 * t_factor);
    // HNO3, for calculation of H2SO4 het rate use
    xk0 = 2.1e5 * haero::exp(8700.0 * t_factor);
    xhen_hno3(kk) = xk0 * (1.0 + hno3_diss / xph0);
    // SO2
    xk0 = 1.23 * haero::exp(3120.0 * t_factor);
    so2_diss = 1.23e-2 * haero::exp(1960.0 * t_factor);
    xhen_so2(kk) = xk0 * (1.0 + so2_diss / xph0);

    // initiate temporary array
    for (int mm = 0; mm < gas_pcnst; mm++) {
      tmp_hetrates[mm](kk) = 0.0;
    }
  }

  //-----------------------------------------------------------------
  //       ... part 1, solve for high henry constant ( hno3, h2o2)
  //-----------------------------------------------------------------
  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team, pver), KOKKOS_LAMBDA(int kk) {
        xgas2(kk) = xh2o2(kk); // different levels wash
        xgas3(kk) = xso2(kk);
      });

  for (int kk = ktop; kk < pver; kk++) {
    stay = 1.0;
    if (rain(kk) != 0.0) { // finding rain cloud
      stay = ((zmid(kk) - zsurf) * km2cm) / (xum * delt);
      stay = haero::min(stay, 1.0);
      // calculate gas washout by cloud
      gas_washout(team, kk, xkgm, xliq(kk), // in
                  xhen_h2o2, tfld, delz,    // in
                  xgas2);                   // inout
      gas_washout(team, kk, xkgm, xliq(kk), // in
                  xhen_so2, tfld, delz,     // in
                  xgas3);                   // inout
    }
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
    xdtm = delz(kk) / xum; // the traveling time in each dz

    xxx2 = (xh2o2(kk) - xgas2(kk));
    if (xxx2 != 0.0) { // if no washout lifetime = 1.e29
      yh2o2 = xh2o2(kk) / xxx2 * xdtm;
    } else {
      yh2o2 = large_value_lifetime;
    }
    tmp_hetrates[1](kk) =
        haero::max(1.0 / yh2o2, 0.0) * stay; // FIXME: bad constant index

    xxx3 = (xso2(kk) - xgas3(kk));
    if (xxx3 != 0.0) { // if no washout lifetime = 1.e29
      yso2 = xso2(kk) / xxx3 * xdtm;
    } else {
      yso2 = large_value_lifetime;
    }
    tmp_hetrates[2](kk) =
        haero::max(1.0 / yso2, 0.0) * stay; // FIXME: bad constant index
  }

  //-----------------------------------------------------------------
  //       ... part 2, in-cloud solve for low henry constant
  //                   hno3 and h2o2 have both in and under cloud
  //-----------------------------------------------------------------
  for (int kk = ktop; kk < pver; kk++) {
    bool skip = false;
    for (int mm = 0; mm < gas_pcnst; mm++) {
      if (rain(kk) <= 0.0) {
        het_rates[mm](kk) = 0.0;
        skip = true;
      }
    }
    if (skip)
      continue;

    work1 = avo2 * xliq(kk);
    work2 = const0 * tfld(kk);

    if (h2o2_ndx >= 0) {
      calc_het_rates(satf_h2o2, rain(kk), xhen_h2o2(kk), // in
                     tmp_hetrates[1](kk), work1, work2,  // in
                     het_rates[h2o2_ndx](kk));           // out
    }

    // if ( prog_modal_aero .and.
    if (so2_ndx >= 0 && h2o2_ndx >= 0) {
      het_rates[so2_ndx](kk) = het_rates[h2o2_ndx](kk);
    } else if (so2_ndx >= 0) {
      calc_het_rates(satf_so2, rain(kk), xhen_so2(kk),  // in
                     tmp_hetrates[2](kk), work1, work2, // in
                     het_rates[so2_ndx](kk));           // out
    }

    if (h2so4_ndx >= 0) {
      calc_het_rates(satf_hno3, rain(kk), xhen_hno3(kk), // in
                     tmp_hetrates[0](kk), work1, work2,  // in
                     het_rates[h2so4_ndx](kk));          // out
    }
  }

  //-----------------------------------------------------------------
  //	... Set rates above tropopause = 0.
  //-----------------------------------------------------------------

  for (int mm = 0; mm < gas_wetdep_cnt; mm++) {
    int mm2 = wetdep_map[mm];
    for (int kk = 0; kk < ktop; kk++) {
      het_rates[mm2](kk) = 0.0;
    }
    for (int kk = 0; kk < pver; kk++) {
      if (het_rates[mm2](kk) == MISSING) {
        return; // maybe?
        Kokkos::abort(
            "sethet: het_rates (wet dep) not set for het reaction number");
      }
    }
    // Didn't port
    //  if ( any( het_rates(:ncol,:,mm2) == MISSING) ) then
    //     write(hetratestrg,'(I3)') mm2
    //     call endrun('sethet: het_rates (wet dep) not set for het reaction
    //     number : '//hetratestrg)
    //  endif
  }
} // end subroutine sethet

} // namespace mo_sethet
} // namespace mam4
#endif