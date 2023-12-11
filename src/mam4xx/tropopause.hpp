#ifndef MAM4XX_TROPOPAUSE_HPP
#define MAM4XX_TROPOPAUSE_HPP

#include <haero/math.hpp>
#include <mam4xx/aero_config.hpp>

#include <fstream>
#include <iostream>

namespace mam4 {

namespace tropopause {

using ConstColumnView = haero::ConstColumnView;
// From radconstants //FIXME: BAD CONSTANT
constexpr int nswbands = 14;
constexpr int nlwbands = 16;
using View2D = DeviceType::view_2d<Real>;
constexpr Real km_inv_to_m_inv = 0.001; // 1/km to 1/m
constexpr int pver = mam4::nlev;

constexpr Real shr_const_rgas =
    haero::Constants::r_gas * 1e3; // Universal gas constant ~ J/K/kmole
constexpr Real shr_const_mwdair = haero::Constants::molec_weight_dry_air *
                                  1e3; // molecular weight dry air ~ kg/kmole
constexpr Real shr_const_cpdair =
    haero::Constants::cp_dry_air; // specific heat of dry air [J/K/kmole]

constexpr Real cnst_kap =
    (shr_const_rgas / shr_const_mwdair) / shr_const_cpdair; //  ! R/Cp

constexpr Real cnst_faktor =
    -haero::Constants::gravity /
    haero::Constants::r_gas_dry_air; // acceleration of gravity ~ m/s^2/Dry air
                                     // gas constant     ~ J/K/kg
constexpr Real cnst_ka1 = cnst_kap - 1.0;

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
  Real pmk0 = 0;
  Real dtdz0 = 0;
  Real dtdz = 0; // temperature lapse rate vs. height [K/m]
  Real tm = 0;   // mean temperature [K]
  Real pmk = 0;
  Real pm = 0;
  Real pmk2 = 0;
  Real pm2 = 0;
  Real dtdz2 = 0;
  Real tm2 = 0;
  Real aquer = 0;
  Real asum = 0;
  Real p2km = 0;

  bool can_finish = false;

  trp = -99.0; // negative means not valid

  // initialize start level
  pmk = half * (haero::pow(pmid1d(pver - 2), cnst_kap) +
                haero::pow(pmid1d(pver - 1), cnst_kap));
  pm = haero::pow(pmk, (one / cnst_kap));

  get_dtdz(pm, pmk, pmid1d(pver - 2), pmid1d(pver - 1), temp1d(pver - 2),
           temp1d(pver - 1), dtdz, tm);

  for (int kk = pver - 2; kk >= 1; --kk) { // main_loop
    pmk0 = pmk;
    dtdz0 = dtdz;
    pmk = half * (haero::pow(pmid1d(kk - 1), cnst_kap) +
                  haero::pow(pmid1d(kk), cnst_kap));
    pm = haero::pow(pmk, (one / cnst_kap));

    get_dtdz(pm, pmk, pmid1d(kk - 1), pmid1d(kk), temp1d(kk - 1), temp1d(kk),
             dtdz, tm);

    // dt/dz valid?
    if (dtdz <= gam) {
      // go to next iteration
      // dt/dz < -2 K/km
      continue;
    } // dtdz<=gam
    if (pm > plimu) {
      // go to next iteration
      // pm too high
      continue;
    } // pm>plimu

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
      continue;
    } // ptph<pliml
    if (ptph > plimu) {
      // cycle main_loop
      continue;
    } // ptph>plimu

    //  2nd test: dtdz above 2 km must not exceed gam
    p2km = ptph + deltaz * (pm / tm) * cnst_faktor; //     p at ptph + 2km
    asum = zero;    //                                dtdz above
    int icount = 0; //   number of levels above

    // test until apm < p2km
    for (int jj = kk; jj >= 1; --jj) { // in_loop
      pmk2 = half * (haero::pow(pmid1d(jj - 1), cnst_kap) +
                     haero::pow(pmid1d(jj), cnst_kap)); // ! p mean ^kappa
      pm2 = haero::pow(pmk2,
                       one / cnst_kap); //                           ! p mean
      if (pm2 > ptph) {
        // cycle in_loop  -   doesn't happen
        continue;
      } // pm2>ptph
      if (pm2 < p2km) {
        // exit in_loop   -  ptropo is valid
        can_finish = true;
        break;
      } // pm2<p2km

      dtdz2 = zero;
      tm2 = zero;
      get_dtdz(pm2, pmk2, pmid1d(jj - 1), pmid1d(jj), temp1d(jj - 1),
               temp1d(jj), dtdz2, tm2);
      asum += dtdz2;
      icount++;
      aquer = asum / Real(icount); //               ! dt/dz mean

      if (aquer <= gam) { // ! discard ptropo ?
        // cycle main_loop      ! dt/dz above < gam
        jj = 1;
        break;
      }
    } // jj test next level (inner loop)
    if (can_finish) {
      trp = ptph;
      break;
    }
  } // kk (main loop)
} // twmo

} // namespace tropopause
} // end namespace mam4

#endif