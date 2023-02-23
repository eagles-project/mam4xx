// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MAM4XX_NUCLEATE_ICE_HPP
#define MAM4XX_NUCLEATE_ICE_HPP

#include <ekat/util/ekat_math_utils.hpp>

#include <haero/atmosphere.hpp>
#include <haero/math.hpp>

#include <mam4xx/aero_config.hpp>
#include <mam4xx/conversions.hpp>
#include <mam4xx/mam4_types.hpp>
#include <mam4xx/utils.hpp>

namespace mam4 {

namespace nucleate_ice {

/*-------------------------------------------------------------------------------
! Purpose:
!  A parameterization of ice nucleation.
!
!  *** This module is intended to be a "portable" code layer.  Ideally it should
!  *** not contain any use association of modules that belong to the model
framework.
!
!
! Method:
!  The current method is based on Liu & Penner (2005) & Liu et al. (2007)
!  It related the ice nucleation with the aerosol number, temperature and the
!  updraft velocity. It includes homogeneous freezing of sulfate & immersion
!  freezing on mineral dust (soot disabled) in cirrus clouds, and
!  Meyers et al. (1992) deposition nucleation in mixed-phase clouds
!
!  The effect of preexisting ice crystals on ice nucleation in cirrus clouds is
included, !  and also consider the sub-grid variability of temperature in cirrus
clouds, !  following X. Shi et al. ACP (2014).
!
!  Ice nucleation in mixed-phase clouds now uses classical nucleation theory
(CNT), !  follows Y. Wang et al. ACP (2014), Hoose et al. (2010).
!
! Authors:
!  Xiaohong Liu, 01/2005, modifications by A. Gettelman 2009-2010
!  Xiangjun Shi & Xiaohong Liu, 01/2014.
!
!  With help from C. C. Chen and B. Eaton (2014)
!-------------------------------------------------------------------------------*/

// FIXME
KOKKOS_INLINE_FUNCTION
Real svp_water(const Real Temperature) {
  Real es = 0;
  return es;
}

// FIXME

KOKKOS_INLINE_FUNCTION
Real svp_ice(const Real Temperature) {
  Real es = 0;
  return es;
}

KOKKOS_INLINE_FUNCTION
void calculate_regm_nucleati(const Real w_vlc, const Real Na, Real &regm) {
  /*-------------------------------------------------------------------------------
  ! Calculate temperature regime for ice nucleation based on
  ! Eq. 4.5 in Liu & Penner (2005), Meteorol. Z.
  !-------------------------------------------------------------------------------*/

  // w_vlc            ! vertical velocity [m/s]
  // Na               ! aerosol number concentration [#/cm^3]
  // regm             ! threshold temperature [C]

  const Real lnNa = haero::log(Na);
  // BAD CONSTANT
  const Real A_coef = -Real(1.4938) * lnNa + Real(12.884);
  const Real B_coef = -Real(10.41) * lnNa - Real(67.69);

  regm = A_coef * log(w_vlc) + B_coef;
} // end calculate_regm_nucleati

KOKKOS_INLINE_FUNCTION
void calculate_RHw_hf(const Real Temperature, const Real lnw, Real &RHw) {
  /*-------------------------------------------------------------------------------
  ! Calculate threshold relative humidity with respective to water (RHw) based
  on ! Eq. 3.1 in Liu & Penner (2005), Meteorol. Z.
  !-------------------------------------------------------------------------------*/

  // Temperature     ! temperature [C]
  // lnw             ! ln of vertical velocity
  // RHw             ! relative humidity threshold

  const Real A_coef = Real(6.0e-4) * lnw + Real(6.6e-3);
  const Real B_coef = Real(6.0e-2) * lnw + Real(1.052);
  const Real C_coef = Real(1.68) * lnw + Real(129.35);

  RHw = (A_coef * Temperature * Temperature + B_coef * Temperature + C_coef) *
        Real(0.01);
} // end calculate_RHw_hf

KOKKOS_INLINE_FUNCTION
void calculate_Ni_hf(const Real A1, const Real B1, const Real C1, const Real A2,
                     const Real B2, const Real C2, const Real Temperature,
                     const Real lnw, const Real Na, Real &Ni)

{
  /*-------------------------------------------------------------------------------
  ! Calculate number of ice crystals (Ni) based on
  ! Eq. 3.3 in Liu & Penner (2005), Meteorol. Z.
  !-------------------------------------------------------------------------------*/

  // A1, B1, C1     ! Coefficients
  // A2, B2, C2     ! Coefficients
  // Temperature    ! temperature [C]
  // lnw            ! ln of vertical velocity
  // Na             ! aerosol number concentrations [#/cm^3]
  // Ni             ! ice number concentrations [#/cm^3]

  const Real k1 = haero::exp(A2 + B2 * Temperature + C2 * lnw);
  const Real k2 = A1 + B1 * Temperature + C1 * lnw;

  Ni = haero::min(k1 * haero::pow(Na, k2), Na);
} // end calculate_Ni_hf

KOKKOS_INLINE_FUNCTION
void hf(const Real Temperature, const Real w_vlc, const Real RH, const Real Na,
        const Real subgrid, Real &Ni) {

  /*-------------------------------------------------------------------------------
  ! Calculate number of ice crystals by homogeneous freezing (Ni) based on
  ! Liu & Penner (2005), Meteorol. Z.
  !-------------------------------------------------------------------------------*/

  // Temperature     ! temperature [C]
  // w_vlc           ! vertical velocity [m/s]
  // RH              ! unitless relative humidity
  // Na              ! aerosol number concentrations [#/cm^3]
  // Ni              ! ice number concentrations [#/cm^3]

  /*---------------------------------------------------------------------
  ! parameters
  !---------------------------------------------------------------------*/

  const Real A1_fast = Real(0.0231);
  const Real A21_fast = -Real(1.6387); //(T>-64 deg)
  const Real A22_fast = -Real(6.045);  //(T<=-64 deg)
  const Real B1_fast = -Real(0.008);
  const Real B21_fast = -Real(0.042); //(T>-64 deg)
  const Real B22_fast = -Real(0.112); //(T<=-64 deg)
  const Real C1_fast = Real(0.0739);
  const Real C2_fast = Real(1.2372);

  const Real A1_slow = -Real(0.3949);
  const Real A2_slow = Real(1.282);
  const Real B1_slow = -Real(0.0156);
  const Real B2_slow = Real(0.0111);
  const Real B3_slow = Real(0.0217);
  const Real C1_slow = Real(0.120);
  const Real C2_slow = Real(2.312);

  /*---------------------------------------------------------------------
  ! local variables
  !---------------------------------------------------------------------*/
  const Real zero = 0;
  Real A2_fast, B2_fast, B4_slow = zero;
  Real lnw, RHw = zero;

  lnw = haero::log(w_vlc);

  Ni = zero;

  calculate_RHw_hf(Temperature, lnw, RHw);

  if ((Temperature <= -Real(37.0)) && (RH * subgrid >= RHw)) {
    // FIXME: This parameter is not used
    const Real regm = Real(6.07) * lnw - Real(55.0);

    if (Temperature >= regm) {
      // fast-growth regime
      if (Temperature > -Real(64.0)) //
      {
        A2_fast = A21_fast;
        B2_fast = B21_fast;
      } else {
        A2_fast = A22_fast;
        B2_fast = B22_fast;
      } // end Temperature

      calculate_Ni_hf(A1_fast, B1_fast, C1_fast, A2_fast, B2_fast, C2_fast,
                      Temperature, lnw, Na, Ni);

    } else {
        //  slow-growth regime

        B4_slow = B2_slow + B3_slow * lnw;

        calculate_Ni_hf(A1_slow, B1_slow, C1_slow, A2_slow, B4_slow, C2_slow,
                        Temperature, lnw, Na, Ni);

    } // end Temperature >= regm
   } // end Temperature <= -Real(37.0)

} // end hf


KOKKOS_INLINE_FUNCTION
void hetero(const Real Temperature, const Real w_vlc, const Real Ns, Real &Nis,
            Real &Nid) {

  /*-------------------------------------------------------------------------------
  ! Calculate number of ice crystals by heterogenous freezing (Nis) based on
  ! Eq. 4.7 in Liu & Penner (2005), Meteorol. Z.
  !-----------------------------------------------------------------------------*/

  // Temperature     ! temperature [C]
  // w_vlc           ! vertical velocity [m/s]
  // Ns              ! aerosol concentrations [#/cm^3]
  // Nis             ! ice number concentrations [#/cm^3]
  // Nid             ! ice number concentrations [#/cm^3]

  /*---------------------------------------------------------------------
  ! parameters
  !---------------------------------------------------------------------*/

  const Real A11 = Real(0.0263);
  const Real A12 = -Real(0.0185);
  const Real A21 = Real(2.758);
  const Real A22 = Real(1.3221);
  const Real B11 = -Real(0.008);
  const Real B12 = -Real(0.0468);
  const Real B21 = -Real(0.2667);
  const Real B22 = -Real(1.4588);

  const Real lnNs = haero::log(Ns);
  const Real lnw = haero::log(w_vlc);

  // ice from immersion nucleation (cm^-3)

  const Real B_coef = (A11 + B11 * lnNs) * lnw + (A12 + B12 * lnNs);
  const Real C_coef = A21 + B21 * lnNs;

  Nis = haero::exp(A22) * haero::pow(Ns, B22) *
        haero::exp(B_coef * Temperature) * haero::pow(w_vlc, C_coef);
  Nis = haero::min(Nis, Ns);
  // FIXME : Mention that this variables is set to zero in PR
  Nid = Real(
      0.0); // don't include deposition nucleation for cirrus clouds when T<-37C

} // hetero

KOKKOS_INLINE_FUNCTION
void nucleati( // inputs
    const Real wbar, const Real tair, const Real pmid, const Real relhum,
    const Real cldn, const Real rhoair, const Real so4_num, const Real dst3_num,
    // inputs
    const Real
        subgrid, // Subgrid scale factor on relative humidity (dimensionless)
    // outputs
    Real &nuci, Real &onihf, Real &oniimm, Real &onidep, Real &onimey) 
{
  /*---------------------------------------------------------------
  ! Purpose:
  !  The parameterization of ice nucleation.
  !
  ! Method: The current method is based on Liu & Penner (2005)
  !  It related the ice nucleation with the aerosol number, temperature and the
  !  updraft velocity. It includes homogeneous freezing of sulfate, immersion
  !  freezing of soot, and Meyers et al. (1992) deposition nucleation
  !
  ! Authors: Xiaohong Liu, 01/2005, modifications by A. Gettelman 2009-2010
  !---------------------------------------------------------------- */

  // Input Arguments
  // wbar        ! grid cell mean vertical velocity [m/s]
  // tair        ! temperature [K]
  // pmid        ! pressure at layer midpoints [pa]
  // relhum      ! relative humidity with respective to liquid [unitless]
  // cldn        ! new value of cloud fraction    [fraction]
  // rhoair      ! air density [kg/m3]
  // so4_num     ! so4 aerosol number [#/cm^3]
  // dst3_num     ! dust aerosol number [#/cm^3]

  // Output Arguments
  // nuci       ! ice number nucleated [#/kg]
  // onihf      ! nucleated number from homogeneous freezing of so4 [#/kg]
  // oniimm     ! nucleated number from immersion freezing [#/kg]
  // onidep     ! nucleated number from deposition nucleation [#/kg]
  // onimey     ! nucleated number from deposition nucleation  (meyers: mixed
  // phase) [#/kg]

  // Local workspace
  Real zero = 0;
  Real nihf = zero;  //                     ! nucleated number from homogeneous
                     //                     freezing of so4 [#/cm^3]
  Real niimm = zero; //                     ! nucleated number from immersion
                     //                     freezing [#/cm^3]
  Real nidep = zero; //                     ! nucleated number from deposition
                     //                     nucleation [#/cm^3]
  Real nimey = zero; //                    ! nucleated number from
  // deposition nucleation (meyers) [#/cm^3]
  Real n1 = zero;
  Real ni = zero; //                  ! nucleated number [#/cm^3]
  const Real tc =
      tair - Real(273.15); //                      ! air temperature [C]
  Real regm = zero;        //                    ! air temperature [C]

  // BAD CONSTANT
  const Real num_threshold = 1.0e-10;

  if (so4_num >= num_threshold && dst3_num >= num_threshold && cldn > zero) {
    if ((tc <= Real(-35.0)) && (relhum * nucleate_ice::svp_water(tair) /
                                    nucleate_ice::svp_ice(tair) / subgrid >=
                                Real(1.2))) {
      //! use higher RHi threshold
      nucleate_ice::calculate_regm_nucleati(wbar, dst3_num, regm);
      if (tc > regm) {
        // heterogeneous nucleation only
        // BAD CONSTANT
        if (tc < -Real(40) && wbar > Real(1.)) {
          // !exclude T<-40 & W> 1m / s from hetero.nucleation

          nucleate_ice::hf(tc, wbar, relhum, so4_num, subgrid, nihf);
          niimm = zero;
          nidep = zero;
          n1 = nihf;

        } else {

          hetero(tc, wbar, dst3_num, niimm, nidep);
          nihf = zero;
          n1 = niimm + nidep;

        } // end tc<Real(-40) ...
      } else if (tc < regm - Real(5.)) {
        // homogeneous nucleation only
        nucleate_ice::hf(tc, wbar, relhum, so4_num, subgrid, nihf);
        niimm = zero;
        nidep = zero;
        n1 = nihf;
      } else {
        // transition between homogeneous and heterogeneous: interpolate
        // in-between

        // BAD CONSTANT
        if (tc < -Real(40.) && wbar > Real(1.)) {
          // exclude T<-40 & W>1m/s from hetero. nucleation

          nucleate_ice::hf(tc, wbar, relhum, so4_num, subgrid, nihf);
          niimm = zero;
          nidep = zero;
          n1 = nihf;

        } else {

          nucleate_ice::hf(regm - Real(5.), wbar, relhum, so4_num, subgrid,
                           nihf);
          hetero(regm, wbar, dst3_num, niimm, nidep);

          if (nihf <= (niimm + nidep)) {
            n1 = nihf;
          } else {
            n1 = (niimm + nidep) *
                 haero::pow((niimm + nidep) / nihf, (tc - regm) / Real(5.));

          } // end nihf <= (niimm + nidep)

        }   // end tc < -40._r8

      }     // end 	tc > regm
      
      ni = n1;


    } // end tc ...

  } // end so4_num ..

  /* deposition/condensation nucleation in mixed clouds (-37<T<0C) (Meyers,
 1992) ! this part is executed but is always replaced by 0, because CNT scheme
 takes over ! the calculation. use_hetfrz_classnuc is always true. */
  // FIXME OD: why adding zero to nuci? is something missing?
  nimey = zero;
  // BAD CONSTANT
  nuci = ni + nimey;
  if (nuci > Real(9999.) || nuci < zero) {
    nuci = zero;
  } // end

  const Real one_millon = 1.e+6;
  nuci = nuci * one_millon / rhoair; //  ! change unit from #/cm3 to #/kg
  onimey = nimey * one_millon / rhoair;
  onidep = nidep * one_millon / rhoair;
  oniimm = niimm * one_millon / rhoair;
  onihf = nihf * one_millon / rhoair;

} // end nucleati

  } // end namespace nucleate_ice

/// @class nucleate_ice
/// This class implements MAM4's nucleate_ice parameterization.
class NucleateIce {
public:
  // nucleate_ice-specific configuration
  struct Config {
    Config() {}
    Config(const Config &) = default;
    ~Config() = default;
    Config &operator=(const Config &) = default;
  };

private:
  Config config_;

public:
  // name--unique name of the process implemented by this class
  const char *name() const { return "MAM4 nucleate_ice"; }
}; // end class nucleate_ice

} // end namespace mam4

#endif
