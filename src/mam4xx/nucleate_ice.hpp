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

// FIXME from wv_sat_methods.F90
//  Do we need to create a new files for these functions


KOKKOS_INLINE_FUNCTION
Real GoffGratch_svp_water(const Real temperature) {
  // ! Goff & Gratch (1946)
  // temperature  ! Temperature in Kelvin
  // 	es             ! SVP in Pa
  // ! uncertain below -70 C
  // BAD CONSTANT
  // FIXME where should we add this constant
  // FROM wv_saturation.F90
  // Boiling point of water at 1 atm (K)
  // This value is slightly high, but it seems to be the value for the
  // steam point of water originally (and most frequently) used in the
  // Goff & Gratch scheme.
  // FIXME add this constant to haero::Constants
  const Real tboil = 373.16;

  const Real ten = 10;
  const Real one = 1;
  return haero::pow(ten,
                    -Real(7.90298) * (tboil / temperature - one) +
                        Real(5.02808) * haero::log10(tboil / temperature) -
                        Real(1.3816e-7) *
                            (haero::pow(ten, Real(11.344) *
                                                 (one - temperature / tboil)) -
                             one) +
                        Real(8.1328e-3) *
                            (haero::pow(ten, -Real(3.49149) *
                                                 (tboil / temperature - one)) -
                             one) +
                        haero::log10(Real(1013.246))) *
         ten * ten;

} // GoffGratch_svp_water

KOKKOS_INLINE_FUNCTION
Real GoffGratch_svp_ice(const Real temperature) {
  // temperature  ! Temperature in Kelvin
  //  es             ! SVP in Pa

  // ! good down to -100 C
  // FIXME
  // Look for a place to place this constant
  // h2otrip ! Triple point temperature of water (K)
  // SHR_CONST_TKTRIP  = 273.16_R8       ! triple point of fresh water        ~
  // K
  const Real h2otrip = haero::Constants::triple_pt_h2o;
  const Real ten = 10;
  const Real one = 1;

  return haero::pow(ten,
                    -Real(9.09718) * (h2otrip / temperature - one) -
                        Real(3.56654) * haero::log10(h2otrip / temperature) +
                        Real(0.876793) * (one - temperature / h2otrip) +
                        haero::log10(6.1071)) *
         ten * ten;

} // end GoffGratch_svp_ice

// FIXME
// Compute saturation vapor pressure over water
KOKKOS_INLINE_FUNCTION
Real svp_water(const Real Temperature) {
  // FIXME
  // ask if we need to implement the other methods to compute svp_water
  // initial_default_idx = GoffGratch_idx
  return GoffGratch_svp_water(Temperature);
}



/*---------------------------------------------------------------------
! UTILITIES
!---------------------------------------------------------------------*/

// Get saturation specific humidity given pressure and SVP.
// Specific humidity is limited to range 0-1.
KOKKOS_INLINE_FUNCTION
Real wv_sat_svp_to_qsat(const Real es, const Real p) {
  // es  ! SVP
  // p   ! Current pressure.
  // qs
  // If pressure is less than SVP, set qs to maximum of 1.
  // epsilo  ! Ice-water transition range
  // omeps   ! 1._r8 - epsilo
  // epsilo       = shr_const_mwwv/shr_const_mwdair   ! ratio of h2o to dry air molecular weights 
  // real(R8),parameter :: SHR_CONST_MWDAIR  = 28.966_R8       ! molecular weight dry air ~ kg/kmole
  // real(R8),parameter :: SHR_CONST_MWWV    = 18.016_R8       ! molecular weight water vapor
  // FIXME: move these constants to hearo::constants
  const Real SHR_CONST_MWWV =  18.016;
  const Real SHR_CONST_MWDAIR = 28.966;
  const Real epsilo= SHR_CONST_MWWV/SHR_CONST_MWDAIR;

  const Real zero=0;
  const Real one =1;
  const Real omeps = one - epsilo;
  Real qs = zero;
  if ((p - es) <= zero){
    qs = one;
  } else {
    qs = epsilo*es / (p - omeps*es);
  }
  return qs;

} // wv_sat_svp_to_qsat


KOKKOS_INLINE_FUNCTION
void wv_sat_qsat_water(const Real t, const Real  p, Real& es, Real&qs)

{
  /*------------------------------------------------------------------!
  ! Purpose:                                                         !
  !   Calculate SVP over water at a given temperature, and then      !
  !   calculate and return saturation specific humidity.             !
  !------------------------------------------------------------------*/
  // Inputs
  // t    ! Temperature
  // p    ! Pressure
  // Outputs
  // es  ! Saturation vapor pressure
  // qs  ! Saturation specific humidity

  es = svp_water(t);
  qs = wv_sat_svp_to_qsat(es, p);
  // Ensures returned es is consistent with limiters on qs.
  es = haero::min(es, p);

}//wv_sat_qsat_water


// FIXME
KOKKOS_INLINE_FUNCTION
Real svp_ice(const Real Temperature) { return GoffGratch_svp_ice(Temperature); }

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

  const Real A1_fast = 0.0231;
  const Real A21_fast = -1.6387; //(T>-64 deg)
  const Real A22_fast = -6.045;  //(T<=-64 deg)
  const Real B1_fast = -0.008;
  const Real B21_fast = -0.042; //(T>-64 deg)
  const Real B22_fast = -0.112; //(T<=-64 deg)
  const Real C1_fast = 0.0739;
  const Real C2_fast = 1.2372;

  const Real A1_slow = -0.3949;
  const Real A2_slow = 1.282;
  const Real B1_slow = -0.0156;
  const Real B2_slow = 0.0111;
  const Real B3_slow = 0.0217;
  const Real C1_slow = 0.120;
  const Real C2_slow = 2.312;

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
  }   // end Temperature <= -Real(37.0)

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

  const Real A11 = 0.0263;
  const Real A12 = -0.0185;
  const Real A21 = 2.758;
  const Real A22 = 1.3221;
  const Real B11 = -0.008;
  const Real B12 = -0.0468;
  const Real B21 = -0.2667;
  const Real B22 = -1.4588;

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

} // end namespace nucleate_ice

/// @class nucleate_ice
/// This class implements MAM4's nucleate_ice parameterization.
class NucleateIce {
public:
  // nucleate_ice-specific configuration
  struct Config {
    // In Fortran code _nucleate_ice_subgrid is read from a file.
    Real _nucleate_ice_subgrid;
    Config(const Real nucleate_ice_subgrid = 0.001)
        : _nucleate_ice_subgrid(nucleate_ice_subgrid) {}
    Config(const Config &) = default;
    ~Config() = default;
    Config &operator=(const Config &) = default;
  };

private:
  Config config_;
  Real _alnsg_amode_aitken, _num_m3_to_cm3, _so4_sz_thresh_icenuc, _mincld,
      _nucleate_ice_subgrid;

public:
  // name--unique name of the process implemented by this class
  const char *name() const { return "MAM4 nucleate_ice"; }

  // init -- initializes the implementation with MAM4's configuration and with
  // a process-specific configuration.
  void init(const AeroConfig &aero_config,
            const Config &nucleate_ice_config = Config()) {

    // alnsg_amode(modeptr_aitken)
    // alog( sigmag_amode(m) )
    _nucleate_ice_subgrid = nucleate_ice_config._nucleate_ice_subgrid;

    _num_m3_to_cm3 = 1.0e-6;
    // BAD CONSTANT
    // FIXME
    // std::numeric_limits<Real>::max()
    // this values is from a txt file
    _so4_sz_thresh_icenuc =
        1e-6; // huge(1.0_r8) !ice nucleation SO2 size threshold for aitken mode
    // minimum allowed cloud fraction
    // BAD CONSTANT
    _mincld = 0.0001;

    const int aitken_idx = int(ModeIndex::Aitken);
    _alnsg_amode_aitken = haero::log(modes(aitken_idx).mean_std_dev);

  } // end(init)

  KOKKOS_INLINE_FUNCTION
  void compute_tendencies(const AeroConfig &config, const ThreadTeam &team,
                          Real t, Real dt, const Atmosphere &atmosphere,
                          const Prognostics &prognostics,
                          const Diagnostics &diagnostics,
                          const Tendencies &tendencies) const {

    const int nk = atmosphere.num_levels();
    const Real tmelt_m_five = haero::Constants::freezing_pt_h2o - 5;
    const int coarse_idx = int(ModeIndex::Coarse);

    auto &coarse_dust = prognostics.q_aero_i[coarse_idx][int(AeroId::DST)];
    auto &coarse_nacl = prognostics.q_aero_i[coarse_idx][int(AeroId::NaCl)];
    auto &coarse_so4 = prognostics.q_aero_i[coarse_idx][int(AeroId::SO4)];

    auto &coarse_mom = prognostics.q_aero_i[coarse_idx][int(AeroId::MOM)];
    auto &coarse_bc = prognostics.q_aero_i[coarse_idx][int(AeroId::BC)];
    auto &coarse_pom = prognostics.q_aero_i[coarse_idx][int(AeroId::POM)];
    auto &coarse_soa = prognostics.q_aero_i[coarse_idx][int(AeroId::SOA)];

    auto &num_coarse = prognostics.n_mode_i[coarse_idx];
    auto &num_aitken = prognostics.n_mode_i[coarse_idx];

    // mode dry radius [m]
    // dgnum(icol,kk,mode_aitken_idx)
    const int aitken_idx = int(ModeIndex::Aitken);
    auto &dgnum_aitken = diagnostics.dry_geometric_mean_diameter_i[aitken_idx];
    // FIXME
    // wsubi(:,:)           ! updraft velocity for ice nucleation [m/s]
    auto &wsubi = atmosphere.updraft_vel_ice_nucleation;
    // could fraction [unitless]
    auto &ast = atmosphere.cloud_fraction;

    // FIXME
    const Real subgrid = _nucleate_ice_subgrid;

    // number of activated aerosol for ice nucleation [#/kg]
    // auto &naai = atmosphere.temperature;
    // output number conc of ice nuclei due to heterogeneous freezing [1/m3]
    auto &nihf = diagnostics.icenuc_num_hetfrz;
    // output number conc of ice nuclei due to immersion freezing (hetero nuc)
    // [1/m3]
    auto &niimm = diagnostics.icenuc_num_immfrz;
    // output number conc of ice nuclei due to deposition nucleation (hetero
    // nuc) [1/m3]
    auto &nidep = diagnostics.icenuc_num_depnuc;
    // !output number conc of ice nuclei due to meyers deposition [1/m3]
    auto &nimey = diagnostics.icenuc_num_meydep;
    
    // number of activated aerosol for ice nucleation (homogeneous freezing only) [#/kg]
    auto &naai_hom = diagnostics.num_act_aerosol_ice_nucle_hom;
    // number of activated aerosol for ice nucleation [#/kg]
    auto &naai     = diagnostics.num_act_aerosol_ice_nucle;

    const Real num_m3_to_cm3 = _num_m3_to_cm3;
    // FIXME
    // huge(1.0_r8) !ice nucleation SO2 size threshold for aitken mode
    const Real so4_sz_thresh_icenuc = _so4_sz_thresh_icenuc;

    const Real mincld = _mincld;
    const Real alnsg_amode_aitken = _alnsg_amode_aitken;

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, nk), KOKKOS_CLASS_LAMBDA(int kk) {
          const Real temp = atmosphere.temperature(kk);
          if (temp < tmelt_m_five) {

            const Real zero = 0;
            const Real half = 0.5;
            const Real two = 2;

            const Real pmid = atmosphere.pressure(kk);
            // CHECK units
            const Real air_density =
                conversions::density_of_ideal_gas(temp, pmid);

            // FIXME: cloud fraction [unitless]
            // could fraction of part of atm?

            // CHECK if this part of code is consistent with original code.
            // relative humidity [unitless]
            Real qv = atmosphere.vapor_mixing_ratio(kk);
            // very low temperature produces inf relhum
            Real es =zero;
            Real qs = zero; 

            nucleate_ice::wv_sat_qsat_water(temp, pmid, es, qs);
            const Real relhum =  qv/qs;

            // Real relhum =
            //     conversions::relative_humidity_from_vapor_mixing_ratio(qv, pmid,
            //                                                            temp);
            const Real icldm = haero::max(ast(kk), mincld);

            // compute aerosol number for so4, soot, and dust with units #/cm^3
            // remove soot number, because it is set to zero
            Real so4_num = zero;
            Real dst3_num = zero;

            /* For modal aerosols, assume for the upper troposphere:
            soot = accumulation mode
            sulfate = aiken mode
            dust = coarse mode
            since modal has internal mixtures. */
            Real dmc = coarse_dust(kk) * air_density;
            Real ssmc = coarse_nacl(kk) * air_density;
            Real so4mc = coarse_so4(kk) * air_density;

            Real mommc = coarse_mom(kk) * air_density;
            Real bcmc = coarse_bc(kk) * air_density;
            Real pommc = coarse_pom(kk) * air_density;
            Real soamc = coarse_soa(kk) * air_density;

            if (dmc > zero) {
              const Real wght =
                  dmc / (ssmc + dmc + so4mc + bcmc + pommc + soamc + mommc);
              dst3_num = wght * num_coarse(kk) * air_density * num_m3_to_cm3;
            } // end dmc

            if (dgnum_aitken(kk) > zero) {
              // only allow so4 with D>0.1 um in ice nucleation
              so4_num =
                  num_aitken(kk) * air_density * num_m3_to_cm3 *
                  (half -
                   half *
                       haero::erf(
                           haero::log(so4_sz_thresh_icenuc / dgnum_aitken(kk)) /
                           haero::pow(two, half * alnsg_amode_aitken)));
            } // end dgnum_aitken

            so4_num = haero::max(zero, so4_num);

            // Real naai = zero;

            nucleati(wsubi(kk), temp, pmid, relhum, icldm, air_density, so4_num,
                     dst3_num, subgrid,
                     // outputs
                     naai(kk), nihf(kk), niimm(kk), nidep(kk), nimey(kk));

            // QUESTION why nihf instead of naai
            naai_hom(kk) = nihf(kk);
            // is naai not saved?

            // output activated ice (convert from #/kg -> #/m3)
            // QUESTION: note that these variables are divided by rho in
            // nucleati
            nihf(kk) *= air_density;
            niimm(kk) *= air_density;
            nidep(kk) *= air_density;
            nimey(kk) *= air_density;

          } // end temp
        }); // kokkos::parfor(k)
  }

public:
  KOKKOS_INLINE_FUNCTION
  void nucleati( // inputs
      const Real wbar, const Real tair, const Real pmid, const Real relhum,
      const Real cldn, const Real rhoair, const Real so4_num,
      const Real dst3_num,
      // inputs
      const Real
          subgrid, // Subgrid scale factor on relative humidity (dimensionless)
      // outputs
      Real &nuci, Real &onihf, Real &oniimm, Real &onidep, Real &onimey) const {
    /*---------------------------------------------------------------
    ! Purpose:
    !  The parameterization of ice nucleation.
    !
    ! Method: The current method is based on Liu & Penner (2005)
    !  It related the ice nucleation with the aerosol number, temperature and
    the !  updraft velocity. It includes homogeneous freezing of sulfate,
    immersion !  freezing of soot, and Meyers et al. (1992) deposition
    nucleation
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
    Real nihf = zero; //                     ! nucleated number from homogeneous
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
                                      nucleate_ice::svp_ice(tair) * subgrid >=
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

            nucleate_ice::hetero(tc, wbar, dst3_num, niimm, nidep);
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
            nucleate_ice::hetero(regm, wbar, dst3_num, niimm, nidep);

            if (nihf <= (niimm + nidep)) {
              n1 = nihf;
            } else {
              n1 = (niimm + nidep) *
                   haero::pow((niimm + nidep) / nihf, (tc - regm) / Real(5.));

            } // end nihf <= (niimm + nidep)

          } // end tc < -40.

        } // end 	tc > regm

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

}; // end class nucleate_ice

} // end namespace mam4

#endif
