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
constexpr Real avo = haero::Constants::avogadro;
const Real pi = haero::Constants::pi;
constexpr Real rgrav =
    mo_chm_diags::rgrav; // reciprocal of acceleration of gravity ~ m/s^2
constexpr const int gas_pcnst = gas_chemistry::gas_pcnst;
const Real boltz_cgs = haero::Constants::boltzmann * 1.e7; // erg/K
// number of vertical levels
constexpr const int pver = mam4::nlev;

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
    const ColumnView cmfdqr,   // dq/dt for convection [kg/kg/s] //in
    const ColumnView nrain,    // stratoform precip [kg/kg/s] //in
    const ColumnView nevapr,   // evaporation [kg/kg/s] // in
    const ColumnView precip) { // precipitation [kg/kg/s] // out
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
    const int plev,          // calculate from this level below //in
    const Real xkgm,         // mass flux on rain drop //in
    const Real xliq_ik,      // liquid rain water content [gm/m^3] // in
    const ColumnView xhen_i, // henry's law constant
    const ColumnView tfld_i, // temperature [K]
    const ColumnView delz_i, // layer depth about interfaces [cm]  // in
    const ColumnView xeqca,  // internal variable
    const ColumnView xca,    // internal variable
    const ColumnView xgas) { // gas concentration // inout
  //------------------------------------------------------------------------
  // calculate gas washout by cloud if not saturated
  //------------------------------------------------------------------------
  // FIXME: BAD CONSTANTS
  Real allca = 0.0; // total of ca between level plev and kk [#/cm3]
  Real const0 = boltz_cgs * 1.0e-6; // [atmospheres/deg k/cm^3]
  Real geo_fac = 6.0; // geometry factor (surf area/volume = geo_fac/diameter)
  Real xrm = .189;    // mean diameter of rain drop [cm]
  Real xum = 748.0;   // mean rain drop terminal velocity [cm/s]

  //-----------------------------------------------------------------
  //       ... calculate the saturation concentration eqca
  //-----------------------------------------------------------------
  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team, plev, pver), KOKKOS_LAMBDA(int k) {
        // cal washout below cloud
        xeqca(k) = xgas(k) /
                   (xliq_ik * avo2 + 1.0 / (xhen_i(k) * const0 * tfld_i(k))) *
                   xliq_ik * avo2;
        //-----------------------------------------------------------------
        //       ... calculate ca; inside cloud concentration in  #/cm3(air)
        //-----------------------------------------------------------------
        xca(k) = geo_fac * xkgm * xgas(k) / (xrm * xum) * delz_i(k) * xliq_ik *
                 cm3_2_m3;
      });

  //-----------------------------------------------------------------
  //       ... if is not saturated (take hno3 as an example)
  //               hno3(gas)_new = hno3(gas)_old - hno3(h2o)
  //           otherwise
  //               hno3(gas)_new = hno3(gas)_old
  //-----------------------------------------------------------------
  for (int kk = 0; kk < plev; kk++) {
    allca += xca(kk);
    if (allca < xeqca(kk)) {
      xgas(kk) = haero::max(xgas(kk) - xca(kk), 0.0);
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
} // namespace mo_sethet
} // namespace mam4
#endif