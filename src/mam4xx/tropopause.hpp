#ifndef MAM4XX_TROPOPAUSE_HPP
#define MAM4XX_TROPOPAUSE_HPP

#include <haero/math.hpp>
#include <mam4xx/aero_config.hpp>

#include <mam4xx/modal_aer_opt.hpp>

namespace mam4 {

namespace tropopause {

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

} // namespace tropopause
} // end namespace mam4

#endif