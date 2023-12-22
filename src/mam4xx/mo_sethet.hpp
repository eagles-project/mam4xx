#ifndef MAM4XX_MO_SETHET_HPP
#define MAM4XX_MO_SETHET_HPP

#include <haero/math.hpp>
#include <mam4xx/gas_chem.hpp>
#include <mam4xx/mam4_types.hpp>
#include <mam4xx/mo_chm_diags.hpp>
#include <mam4xx/utils.hpp>

namespace mam4 {

namespace mo_sethet {

// const int ktop = ConvProc::Config::ktop;
// int ktop = 47; // BAD_CONSTANT: only true for nlev == 72
constexpr Real avo = haero::Constants::avogadro;
const Real pi = haero::Constants::pi;
constexpr Real rgrav =
    mo_chm_diags::rgrav; // reciprocal of acceleration of gravity ~ m/s^2
constexpr const int gas_pcnst = gas_chemistry::gas_pcnst;
const Real boltz_cgs = haero::Constants::boltzmann * 1.e7; // erg/K
// number of vertical levels
constexpr const int pver = mam4::nlev;

// FIXME: BAD CONSTANT
constexpr const Real mass_h2o = 18.0;   // mass of water vapor [amu]
constexpr const Real cm3_2_m3 = 1.0e-6; // convert cm^3 to m^3
constexpr const Real liter_per_gram = 1.0e-3;
constexpr const Real avo2 =
    avo * liter_per_gram * cm3_2_m3; // [liter/gm/mol*(m/cm)^3]

using Real = haero::Real;
using View1D = DeviceType::view_1d<Real>;

//=================================================================================
KOKKOS_INLINE_FUNCTION
void calc_het_rates(Real satf, // saturation fraction in cloud //in
                    Real rain, // rain rate [molecules/cm^3/s] //in
                    Real xhen, // henry's law constant // in
                    Real tmp_hetrates, Real work1, Real work2, // in
                    Real &het_rates) // rainout loss rates [1/s]// out
{
  //-----------------------------------------------------------------
  // calculate het_rates
  // input arguments are different for different species
  //-----------------------------------------------------------------
  Real work3;
  Real h2o_mol = 1.0e3 / mass_h2o; // [gm/mol water]
  // FIXME: BAD CONSTANT should haero::Constants::molec_weight_h2o be used
  // somewhere here instead?

  work3 =
      satf * haero::max(rain / (h2o_mol * (work1 + 1.0 / (xhen * work2))), 0.0);
  het_rates = work3 + tmp_hetrates;

} // end calc_het_rates
//=================================================================================

//=================================================================================
KOKKOS_INLINE_FUNCTION
void calc_precip_rescale(
    ColumnView cmfdqr,   // dq/dt for convection [kg/kg/s] //in
    ColumnView nrain,    // stratoform precip [kg/kg/s] //in
    ColumnView nevapr,   // evaporation [kg/kg/s] // in
    ColumnView precip) { // precipitation [kg/kg/s] // out
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
void gas_washout(      // const ThreadTeam &team,
    int plev,          // calculate from this level below //in
    Real xkgm,         // mass flux on rain drop //in
    Real xliq_ik,      // liquid rain water content [gm/m^3] // in
    ColumnView xhen_i, // henry's law constant
    ColumnView tfld_i, // temperature [K]
    ColumnView delz_i, // layer depth about interfaces [cm]  // in
    ColumnView xgas) { // gas concentration // inout
  //------------------------------------------------------------------------
  // calculate gas washout by cloud if not saturated
  //------------------------------------------------------------------------
  // FIXME: BAD CONSTANTS
  Real allca = 0.0; // total of ca between level plev and kk [#/cm3]
  Real xca, xeqca;
  Real const0 = boltz_cgs * 1.0e-6; // [atmospheres/deg k/cm^3]
  Real geo_fac = 6.0; // geometry factor (surf area/volume = geo_fac/diameter)
  Real xrm = .189;    // mean diameter of rain drop [cm]
  Real xum = 748.0;   // mean rain drop terminal velocity [cm/s]

  //-----------------------------------------------------------------
  //       ... calculate the saturation concentration eqca
  //-----------------------------------------------------------------
  for (int kk = 0; kk < plev;
       kk++) { // FIXME: not sure if this should be a Kokkos for or not...
    // cal washout below cloud
    xeqca = xgas(kk) /
            (xliq_ik * avo2 + 1.0 / (xhen_i(kk) * const0 * tfld_i(kk))) *
            xliq_ik * avo2;

    //-----------------------------------------------------------------
    //       ... calculate ca; inside cloud concentration in  #/cm3(air)
    //-----------------------------------------------------------------
    xca = geo_fac * xkgm * xgas(kk) / (xrm * xum) * delz_i(kk) * xliq_ik *
          cm3_2_m3;

    //-----------------------------------------------------------------
    //       ... if is not saturated (take hno3 as an example)
    //               hno3(gas)_new = hno3(gas)_old - hno3(h2o)
    //           otherwise
    //               hno3(gas)_new = hno3(gas)_old
    //-----------------------------------------------------------------
    allca = allca + xca;
    if (allca < xeqca) {
      xgas(kk) = haero::max(xgas(kk) - xca, 0.0);
    }
  }
} // end subroutine gas_washout

//=================================================================================
KOKKOS_INLINE_FUNCTION
void find_ktop(
    Real rlat,        // latitude in radians for columns
    ColumnView press, // pressure [Pa] // in
    int &ktop) { // index that only calculate het_rates above this level //out
  //---------------------------------------------------------------------------
  // -------- find the top level that het_rates are set as 0 above it ---------
  //---------------------------------------------------------------------------

  Real p_limit;          // pressure limit [Pa]
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
} // namespace mo_sethet
} // namespace mam4
#endif