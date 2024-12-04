// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MAM4XX_WET_DEPOSITION_HPP
#define MAM4XX_WET_DEPOSITION_HPP

#include <ekat/kokkos/ekat_subview_utils.hpp>
#include <haero/atmosphere.hpp>
#include <haero/constants.hpp>
#include <haero/math.hpp>
#include <limits>
#include <mam4xx/aero_config.hpp>
#include <mam4xx/aero_model.hpp>
#include <mam4xx/modal_aer_opt.hpp>
#include <mam4xx/utils.hpp>

// Based on e3sm_mam4_refactor/components/eam/src/chemistry/aerosol/wetdep.F90
namespace mam4 {

namespace wetdep {

/**
 * @brief Calculate local precipitation generation rate (kg/m2/s)
 *        from source (condensation) and sink (evaporation) terms.
 *
 * @param[in] pdel Pressure difference across layers [Pa].
 * @param[in] source_term Precipitation source term rate (condensation)
 * [kg/kg/s].
 * @param[in] sink_term Precipitation sink term rate (evaporation) [kg/kg/s].
 * @param[in] atm Atmosphere object (used for number of levels).
 *
 * @param[out] lprec Local production rate of precipitation [kg/m2/s].
 *
 * @pre pdel, source_term, sink_term and lprec are all an array
 *      of size pver == atm.num_levels().
 *
 * @pre In F90, ncol == 1 as we only operate over one column at a time
 *      as outer loops will iterate over columns, so we drop ncol as input.
 * @pre In F90 code, pcols is the number of columns in the mesh.
 *      Since we are only operating over one column, pcols == 1.
 * @pre In F90 code, ncol == pcols. Since ncol == 1, pcols == 1.
 *
 * @pre atm is initialized correctly and has the correct number of levels.
 */

using ConstView1D = DeviceType::view_1d<const Real>;
using View1D = DeviceType::view_1d<Real>;
using Int1D = DeviceType::view_1d<int>;
using View2D = DeviceType::view_2d<Real>;
KOKKOS_INLINE_FUNCTION
Real local_precip_production(const Real pdel, const Real source_term,
                             const Real sink_term) {
  return (pdel / Constants::gravity) * (source_term - sink_term);
}

// Function to call to initialize the arrays passe to the
// aero_model_wetdep function. This data is constant so should
// be allocaed on host and copied to device.
inline void init_scavimptbl(
    Real scavimptblvol[aero_model::nimptblgrow_total][AeroConfig::num_modes()],
    Real scavimptblnum[aero_model::nimptblgrow_total]
                      [AeroConfig::num_modes()]) {
  const int num_modes = AeroConfig::num_modes();
  Real dgnum_amode[num_modes];
  Real sigmag_amode[num_modes];
  Real aerosol_dry_density[num_modes];
  for (int i = 0; i < num_modes; ++i) {
    dgnum_amode[i] = modes(i).nom_diameter;
    sigmag_amode[i] = modes(i).mean_std_dev;
  }
  // Note: Original code uses the following aerosol densities.
  // sulfate, sulfate, dust, p-organic
  aerosol_dry_density[0] = mam4::mam4_density_so4;
  aerosol_dry_density[1] = mam4::mam4_density_so4;
  aerosol_dry_density[2] = mam4::mam4_density_dst;
  aerosol_dry_density[3] = mam4::mam4_density_pom;
  aero_model::modal_aero_bcscavcoef_init(dgnum_amode, sigmag_amode,
                                         aerosol_dry_density, scavimptblnum,
                                         scavimptblvol);
}

// clang-format off
/**
 * @brief Calculate cloudy volume which is occupied by rain or cloud water as the
 *        max between the local cloud amount or the:
 *          sum above of (cloud * positive precip production)      sum total precip from above
 *            ---------------------------------------------    X    -------------------------
 *                   sum above of ( positive precip )             sum positive precip from above
 *
 * @param[in] cld Cloud faction [fraction, unitless].
 * @param[in] lprec Local production rate of precipitation [kg/m2/s].
 * @param[in] is_tot_cld When is_tot_cld is true, this function computes the
 *                       total cloud volume. Otherwise, it computes the cloud
 *                       volume of either the stratiform clouds or the
 *                       convective clouds (depending upon the provided cloud fraction).
 * @param[in] atm Atmosphere object (used for number of levels).
 *
 * @param[out] cldv Fraction occupied by rain or cloud water [fraction, unitless].
 *
 * @pre cld, lprec, cldv are all an array
 *      of size nlev == atm.num_levels().
 *
 * @pre In F90, ncol == 1 as we only operate over one column at a time
 *      as outer loops will iterate over columns, so we drop ncol as input.
 * @pre In F90 code, pcols is the number of columns in the mesh.
 *      Since we are only operating over one column, pcols == 1.
 * @pre In F90 code, ncol == pcols. Since ncol == 1, pcols == 1.
 *
 * @pre atm is initialized correctly and has the correct number of levels.
 *
 */
// clang-format on
template <typename FUNC>
KOKKOS_INLINE_FUNCTION void
calculate_cloudy_volume(const int nlev, const Real cld[/*nlev*/], FUNC lprec,
                        const bool is_tot_cld, Real cldv[/*nlev*/]) {
  // BAD CONSTANT
  const Real small_value_30 = 1.e-30;
  const Real small_value_36 = 1.e-36;
  Real sumppr = 0.0; // Precipitation rate [kg/m2/s]
  Real cldv1 = 0.0;  // Precip weighted cloud fraction from above [kg/m2/s]
  Real sumpppr = small_value_36; // Sum of positive precips from above

  for (int i = 0; i < nlev; i++) {
    const Real clouds = haero::min(1.0, cldv1 / sumpppr) * (sumppr / sumpppr);
    if (is_tot_cld) {
      cldv[i] = haero::max(clouds, cld[i]);
    } else {
      // For convective and stratiform precipitation volume at the top
      // interface of each layer. Neglect the current layer.
      cldv[i] = haero::max(clouds, 0.0);
    }
    // Local production rate of precip [kg/m2/s] if positive
    const Real prec = lprec(i);
    const Real lprecp = haero::max(prec, small_value_30);
    cldv1 += cld[i] * lprecp;
    sumppr += prec;
    sumpppr += lprecp;
  }
}

// ==============================================================================
KOKKOS_INLINE_FUNCTION
void update_scavenging(const int mam_prevap_resusp_optcc, const Real pdel_ik,
                       const Real omsm, const Real srcc, const Real srcs,
                       const Real srct, const Real fins, const Real finc,
                       const Real fracev_st, const Real fracev_cu,
                       const Real resusp_c, const Real resusp_s,
                       const Real precs_ik, const Real evaps_ik,
                       const Real cmfdqr_ik, const Real evapc_ik,
                       Real &scavt_ik, Real &iscavt_ik, Real &icscavt_ik,
                       Real &isscavt_ik, Real &bcscavt_ik, Real &bsscavt_ik,
                       Real &rcscavt_ik, Real &rsscavt_ik, Real &scavabs,
                       Real &scavabc, Real &precabc, Real &precabs) {
  // clang-format off
  // ------------------------------------------------------------------------------
  // update scavenging variables
  // *_ik are variables at the grid (icol, kk)
  // ------------------------------------------------------------------------------
  /*
  // input variables
  in :: mam_prevap_resusp_optcc       ! suspension options
  in :: pdel_ik       ! pressure thikness [Pa]
  in :: omsm          ! 1 - (a small number), to prevent roundoff errors below zero
  in :: srcc          ! tend for convective rain scavenging [kg/kg/s]
  in :: srcs          ! tend for stratiform rain scavenging [kg/kg/s]
  in :: srct          ! total scavenging tendency for conv+strat rain [kg/kg/s]
  in :: fins          ! fraction of rem. rate by strat rain [fraction]
  in :: finc          ! fraction of rem. rate by conv. rain [fraction]
  in :: fracev_st     ! fraction of stratiform precip from above that is evaporating [fraction]
  in :: fracev_cu     ! Fraction of convective precip from above that is evaporating [fraction]
  in :: resusp_c      ! aerosol mass re-suspension in a particular layer from convective rain [kg/m2/s]
  in :: resusp_s      ! aerosol mass re-suspension in a particular layer from stratiform rain [kg/m2/s]
  in :: precs_ik      ! rate of production of stratiform precip [kg/kg/s]
  in :: evaps_ik      ! rate of evaporation of precip [kg/kg/s]
  in :: cmfdqr_ik     ! rate of production of convective precip [kg/kg/s]
  in :: evapc_ik      ! Evaporation rate of convective precipitation [kg/kg/s]
  // output variables
  out :: scavt_ik    ! scavenging tend [kg/kg/s]
  out :: iscavt_ik   ! incloud scavenging tends [kg/kg/s]
  out :: icscavt_ik  ! incloud, convective [kg/kg/s]
  out :: isscavt_ik  ! incloud, stratiform [kg/kg/s]
  out :: bcscavt_ik  ! below cloud, convective [kg/kg/s]
  out :: bsscavt_ik  ! below cloud, stratiform [kg/kg/s]
  out :: rcscavt_ik  ! resuspension, convective [kg/kg/s]
  out :: rsscavt_ik  ! resuspension, stratiform [kg/kg/s]
  inout :: scavabs   ! stratiform scavenged tracer flux from above [kg/m2/s]
  inout :: scavabc   ! convective scavenged tracer flux from above [kg/m2/s]
  inout :: precabc   ! conv precip from above [kg/m2/s]
  inout :: precabs   ! strat precip from above [kg/m2/s]
  */
  // clang-format on
  const Real gravit = Constants::gravity;

  if (mam_prevap_resusp_optcc == 0)
    scavt_ik =
        -srct + (fracev_st * scavabs + fracev_cu * scavabc) * gravit / pdel_ik;
  else
    scavt_ik = -srct + (resusp_s + resusp_c) * gravit / pdel_ik;

  iscavt_ik = -(srcc * finc + srcs * fins) * omsm;
  icscavt_ik = -(srcc * finc) * omsm;
  isscavt_ik = -(srcs * fins) * omsm;

  if (mam_prevap_resusp_optcc == 0) {
    bcscavt_ik =
        -(srcc * (1 - finc)) * omsm + fracev_cu * scavabc * gravit / pdel_ik;
    bsscavt_ik =
        -(srcs * (1 - fins)) * omsm + fracev_st * scavabs * gravit / pdel_ik;
    rcscavt_ik = 0.0;
    rsscavt_ik = 0.0;
  } else {
    // here mam_prevap_resusp_optcc == 130, 210, 230
    bcscavt_ik = -(srcc * (1 - finc)) * omsm;
    rcscavt_ik = resusp_c * gravit / pdel_ik;
    bsscavt_ik = -(srcs * (1 - fins)) * omsm;
    rsscavt_ik = resusp_s * gravit / pdel_ik;
  }

  // now keep track of scavenged mass and precip
  if (mam_prevap_resusp_optcc == 0) {
    scavabs = scavabs * (1 - fracev_st) + srcs * pdel_ik / gravit;
    scavabc = scavabc * (1 - fracev_cu) + srcc * pdel_ik / gravit;
    precabs = precabs + (precs_ik - evaps_ik) * pdel_ik / gravit;
    precabc = precabc + (cmfdqr_ik - evapc_ik) * pdel_ik / gravit;
  }
}
// ==============================================================================
KOKKOS_INLINE_FUNCTION
Real flux_precnum_vs_flux_prec_mpln(const Real flux_prec, const int jstrcnv) {
  // clang-format off
  // --------------------------------------------------------------------------------
  //  flux_precnum_vs_flux_prec_mp = precipitation number flux at the cloud base [drops/m^2/s]
  //  Options of assuming log-normal or marshall-palmer raindrop size distribution
  // --------------------------------------------------------------------------------
  /*
  in :: flux_prec     ! [drops/m^2/s]
  in :: jstrcnv
  out :: flux_precnum_vs_flux_prec_mpln  ! [drops/m^2/s]
  */
  // clang-format on

  // BAD CONSTANT
  const Real small_value_36 = 1.e-36;

  // current only two options: 1 for marshall-palmer distribution, 2 for
  // log-normal distribution
  Real a0, a1;
  if (jstrcnv <= 1) {
    // marshall-palmer distribution
    a0 = 1.0885896550304022e+01;
    a1 = 4.3660645528167907e-01;
  } else {
    // log-normal distribution
    a0 = 9.9067806476181524e+00;
    a1 = 4.2690709912134056e-01;
  }
  Real y_var;
  if (flux_prec >= small_value_36) {
    const Real x_var = haero::log(flux_prec);
    y_var = haero::exp(a0 + a1 * x_var);
  } else {
    y_var = 0.0;
  }
  return y_var;
}

// ==============================================================================
KOKKOS_INLINE_FUNCTION
Real faer_resusp_vs_fprec_evap_mpln(const Real fprec_evap, const int jstrcnv) {
  // clang-format off
  //  --------------------------------------------------------------------------------
  //  corresponding fraction of precipitation-borne aerosol flux that is resuspended
  //  Options of assuming log-normal or marshall-palmer raindrop size distribution
  //  note that these fractions are relative to the cloud-base fluxes,
  //  and not to the layer immediately above fluxes
  //  --------------------------------------------------------------------------------
  /*
  in :: fprec_evap ! [fraction]
  in :: jstrcnv !  current only two options: 1 for marshall-palmer distribution, 2 for log-normal distribution
  out : faer_resusp_vs_fprec_evap_mpln ! [fraction]
  */
  // clang-format on

  // current only two options: 1 for marshall-palmer distribution, 2 for
  // log-normal distribution
  Real a01, a02, a03, a04, a05, a06, a07, a08, a09, x_lox_lin, y_lox_lin;
  if (jstrcnv <= 1) {
    // marshall-palmer distribution
    a01 = 8.6591133737322856e-02;
    a02 = -1.7389168499601941e+00;
    a03 = 2.7401882373663732e+01;
    a04 = -1.5861714653209464e+02;
    a05 = 5.1338179363011193e+02;
    a06 = -9.6835933124501412e+02;
    a07 = 1.0588489932213311e+03;
    a08 = -6.2184513459217271e+02;
    a09 = 1.5184126886039758e+02;
    x_lox_lin = 5.0000000000000003e-02;
    y_lox_lin = 2.5622471203221014e-03;
  } else {
    // log-normal distribution
    a01 = 6.1944215103685640e-02;
    a02 = -2.0095166685965378e+00;
    a03 = 2.3882460251821236e+01;
    a04 = -1.2695611774753374e+02;
    a05 = 4.0086943562320101e+02;
    a06 = -7.4954272875943707e+02;
    a07 = 8.1701055892023624e+02;
    a08 = -4.7941894659538502e+02;
    a09 = 1.1710291076059025e+02;
    x_lox_lin = 1.0000000000000001e-01;
    y_lox_lin = 6.2227889828044350e-04;
  }

  Real y_var;
  const Real x_var = utils::min_max_bound(0.0, 1.0, fprec_evap);
  if (x_var < x_lox_lin)
    y_var = y_lox_lin * (x_var / x_lox_lin);
  else
    y_var =
        x_var *
        (a01 +
         x_var *
             (a02 +
              x_var *
                  (a03 +
                   x_var *
                       (a04 +
                        x_var * (a05 +
                                 x_var * (a06 +
                                          x_var * (a07 +
                                                   x_var * (a08 +
                                                            x_var * a09))))))));

  return y_var;
}

//==============================================================================
KOKKOS_INLINE_FUNCTION
Real fprecn_resusp_vs_fprec_evap_mpln(const Real fprec_evap,
                                      const int jstrcnv) {
  // clang-format off
  // --------------------------------------------------------------------------------
  // Rain number evaporation fraction
  // Options of assuming log-normal or marshall-palmer raindrop size distribution
  // note that these fractions are relative to the cloud-base fluxes,
  // and not to the layer immediately above fluxes
  // --------------------------------------------------------------------------------
  /*
  in :: fprec_evap     ! [fraction]
  in :: jstrcnv  ! current only two options: 1 for marshall-palmer distribution, 2 for log-normal distribution
  out :: fprecn_resusp_vs_fprec_evap_mpln  ! [fraction]
  */
  // clang-format on

  // current only two options: 1 for marshall-palmer distribution, 2 for
  // log-normal distribution
  Real a01, a02, a03, a04, a05, a06, a07, a08, a09, x_lox_lin, y_lox_lin;
  if (jstrcnv <= 1) {
    // marshall-palmer distribution
    a01 = 4.5461070198414655e+00;
    a02 = -3.0381753620077529e+01;
    a03 = 1.7959619926085665e+02;
    a04 = -6.7152282193785618e+02;
    a05 = 1.5651931323557126e+03;
    a06 = -2.2743927701175126e+03;
    a07 = 2.0004645897056735e+03;
    a08 = -9.7351466279626209e+02;
    a09 = 2.0101198012962413e+02;
    x_lox_lin = 5.0000000000000003e-02;
    y_lox_lin = 1.7005858490684875e-01;
  } else {
    // log-normal distribution
    a01 = -5.2335291116884175e-02;
    a02 = 2.7203158069178226e+00;
    a03 = 9.4730878152409375e+00;
    a04 = -5.0573187592544798e+01;
    a05 = 9.4732631441282862e+01;
    a06 = -8.8265926556465814e+01;
    a07 = 3.5247835268269142e+01;
    a08 = 1.5404586576716444e+00;
    a09 = -3.8228795492549068e+00;
    x_lox_lin = 1.0000000000000001e-01;
    y_lox_lin = 2.7247994766566485e-02;
  }

  Real y_var;
  const Real x_var = utils::min_max_bound(0.0, 1.0, fprec_evap);
  if (x_var < x_lox_lin)
    y_var = y_lox_lin * (x_var / x_lox_lin);
  else
    y_var =
        x_var *
        (a01 +
         x_var *
             (a02 +
              x_var *
                  (a03 +
                   x_var *
                       (a04 +
                        x_var * (a05 +
                                 x_var * (a06 +
                                          x_var * (a07 +
                                                   x_var * (a08 +
                                                            x_var * a09))))))));

  return y_var;
}
// ==============================================================================
KOKKOS_INLINE_FUNCTION
void wetdep_prevap(const int is_st_cu, const int mam_prevap_resusp_optcc,
                   const Real pdel_ik, const Real pprdx, const Real srcx,
                   const Real arainx, const Real precabx_old,
                   const Real precabx_base_old, const Real scavabx_old,
                   const Real precnumx_base_old, Real &precabx_new,
                   Real &precabx_base_new, Real &scavabx_new,
                   Real &precnumx_base_new) {
  // clang-format off
  // ------------------------------------------------------------------------------
  // do precip production and scavenging
  // ------------------------------------------------------------------------------
  /*
  in :: is_st_cu      ! options for stratiform (1) or convective (2) clouds
                      ! raindrop size distribution is
                      ! different for different cloud:
                      ! 1: assume marshall-palmer distribution
                      ! 2: assume log-normal distribution
  in :: mam_prevap_resusp_optcc       ! suspension options
  in :: pdel_ik       ! pressure thikness at current column and level [Pa]
  in :: pprdx  ! precipitation generation rate [kg/kg/s]
  in :: srcx   ! scavenging tendency [kg/kg/s]
  in :: arainx ! precipitation and cloudy volume,at the top interface of current layer [fraction]
  in :: precabx_base_old ! input of precipitation at cloud base [kg/m2/s]
  in :: precabx_old ! input of precipitation above this layer [kg/m2/s]
  in :: scavabx_old ! input scavenged tracer flux from above [kg/m2/s]
  in :: precnumx_base_old ! input of rain number at cloud base [#/m2/s]
  out :: precabx_base_new ! output of precipitation at cloud base [kg/m2/s]
  out :: precabx_new ! output of precipitation above this layer [kg/m2/s]
  out :: scavabx_new ! output scavenged tracer flux from above [kg/m2/s]
  out :: precnumx_base_new ! output of rain number at cloud base [#/m2/s]
  */
  // clang-format on
  // BAD CONSTANT
  const Real small_value_30 = 1.e-30;
  const Real gravit = Constants::gravity;

  // initiate *_new in case they are not calculated
  // precabx_base_new and precabx_new are always calculated
  scavabx_new = scavabx_old;
  precnumx_base_new = precnumx_base_old;

  Real tmpa = haero::max(0.0, pprdx * pdel_ik / gravit);
  precabx_base_new = haero::max(0.0, precabx_base_old + tmpa);
  precabx_new = utils::min_max_bound(0.0, precabx_base_new, precabx_old + tmpa);

  if (mam_prevap_resusp_optcc <= 130) {
    // aerosol mass scavenging
    tmpa = haero::max(0.0, srcx * pdel_ik / gravit);
    scavabx_new = haero::max(0.0, scavabx_old + tmpa);
  } else {
    // raindrop number increase
    if (precabx_base_new < small_value_30) {
      precnumx_base_new = 0.0;
    } else if (precabx_base_new > precabx_base_old) {
      // note - calc rainshaft number flux from rainshaft water flux,
      // then multiply by rainshaft area to get grid-average number flux
      tmpa = arainx * flux_precnum_vs_flux_prec_mpln(precabx_base_new / arainx,
                                                     is_st_cu);
      precnumx_base_new = haero::max(0.0, tmpa);
    } else {
      precnumx_base_new = precnumx_base_old;
    }
  }
}
// ==============================================================================
// ==============================================================================
KOKKOS_INLINE_FUNCTION
void wetdep_resusp_nonlinear(
    const int is_st_cu, const int mam_prevap_resusp_optcc,
    const Real precabx_old, const Real precabx_base_old, const Real scavabx_old,
    const Real precnumx_base_old, const Real precabx_new, Real &scavabx_new,
    Real &resusp_x) {

  // clang-format off
  //  ------------------------------------------------------------------------------
  //  do nonlinear resuspension of aerosol mass or number
  //  ------------------------------------------------------------------------------
  /*
   in :: is_st_cu      ! options for stratiform (1) or convective (2) clouds
                       ! raindrop size distribution is
                       ! different for different cloud:
                       ! 1: assume marshall-palmer distribution
                       ! 2: assume log-normal distribution
   in :: mam_prevap_resusp_optcc       ! suspension options
   in :: precabx_base_old ! input of precipitation at cloud base [kg/m2/s]
   in :: precabx_old  ! input of precipitation above this layer [kg/m2/s]
   in :: scavabx_old  ! input scavenged tracer flux from above [kg/m2/s]
   in :: precnumx_base_old ! precipitation number at cloud base [#/m2/s]
   in :: precabx_new  ! output of precipitation above this layer [kg/m2/s]
   out :: scavabx_new ! output scavenged tracer flux from above [kg/m2/s]
   out :: resusp_x    ! aerosol mass re-suspension in a particular layer [kg/m2/s]
  */
  // clang-format on

  // BAD CONSTANT
  const Real small_value_30 = 1.e-30;

  // fraction of precabx and precabx_base
  const Real u_old =
      utils::min_max_bound(0.0, 1.0, precabx_old / precabx_base_old);
  // fraction after calling function *_resusp_vs_fprec_evap_mpln
  Real x_old, x_new;
  if (mam_prevap_resusp_optcc <= 130) {
    // non-linear resuspension of aerosol mass
    x_old = 1.0 - faer_resusp_vs_fprec_evap_mpln(1.0 - u_old, is_st_cu);
  } else {
    // non-linear resuspension of aerosol number based on raindrop number
    x_old = 1.0 - fprecn_resusp_vs_fprec_evap_mpln(1.0 - u_old, is_st_cu);
  }
  x_old = utils::min_max_bound(0.0, 1.0, x_old);

  Real x_ratio; // fraction of x_tmp/x_old
  if (x_old < small_value_30) {
    x_new = 0.0;
    x_ratio = 0.0;
  } else {
    // fraction of precabx and precabx_base
    Real u_new = utils::min_max_bound(0.0, 1.0, precabx_new / precabx_base_old);
    u_new = haero::min(u_new, u_old);
    if (mam_prevap_resusp_optcc <= 130) {
      // non-linear resuspension of aerosol mass
      x_new = 1.0 - faer_resusp_vs_fprec_evap_mpln(1.0 - u_new, is_st_cu);
    } else {
      // non-linear resuspension of aerosol number based on raindrop number
      x_new = 1.0 - fprecn_resusp_vs_fprec_evap_mpln(1.0 - u_new, is_st_cu);
    }
    x_new = utils::min_max_bound(0.0, 1.0, x_new);
    x_new = haero::min(x_new, x_old);
    x_ratio = utils::min_max_bound(0.0, 1.0, x_new / x_old);
  }

  // update aerosol resuspension
  if (mam_prevap_resusp_optcc <= 130) {
    // aerosol mass resuspension
    scavabx_new = haero::max(0.0, scavabx_old * x_ratio);
    resusp_x = haero::max(0.0, scavabx_old - scavabx_new);
  } else {
    // number resuspension
    scavabx_new = 0;
    resusp_x = haero::max(0.0, precnumx_base_old * (x_old - x_new));
  }
}
// ==============================================================================
// ==============================================================================
KOKKOS_INLINE_FUNCTION
void wetdep_resusp_noprecip(const int is_st_cu,
                            const int mam_prevap_resusp_optcc,
                            const Real precabx_old, const Real precabx_base_old,
                            const Real scavabx_old,
                            const Real precnumx_base_old, Real &precabx_new,
                            Real &precabx_base_new, Real &scavabx_new,
                            Real &resusp_x) {
  // clang-format off
  // ------------------------------------------------------------------------------
  // do complete resuspension when precipitation rate is zero
  // ------------------------------------------------------------------------------
  /*
  in :: is_st_cu      ! options for stratiform (1) or convective (2) clouds
                      ! raindrop size distribution is
                      ! different for different cloud:
                      ! 1: assume marshall-palmer distribution
                      ! 2: assume log-normal distribution
  in :: mam_prevap_resusp_optcc       ! suspension options
  in :: precabx_base_old ! input of precipitation at cloud base [kg/m2/s]
  in :: precabx_old ! input of precipitation above this layer [kg/m2/s]
  in :: scavabx_old ! input of scavenged tracer flux from above [kg/m2/s]
  in :: precnumx_base_old ! precipitation number at cloud base [#/m2/s]
  out :: precabx_base_new ! output of precipitation at cloud base [kg/m2/s]
  out :: precabx_new ! output of precipitation above this layer [kg/m2/s]
  inout :: scavabx_new ! output of scavenged tracer flux from above [kg/m2/s]
  out :: resusp_x    ! aerosol mass re-suspension in a particular layer [kg/m2/s]
  */
  // clang-format on

  // BAD CONSTANT
  const Real small_value_30 = 1.e-30;

  if (mam_prevap_resusp_optcc <= 130) {
    scavabx_new = 0.0;
    // linear resuspension based on scavenged aerosol mass or number
    resusp_x = scavabx_old;
  } else {
    if (precabx_base_old < small_value_30) {
      resusp_x = 0.0;
    } else {
      // non-linear resuspension of aerosol number based on raindrop number
      const Real u_old =
          utils::min_max_bound(0.0, 1.0, precabx_old / precabx_base_old);
      Real x_old =
          1.0 - fprecn_resusp_vs_fprec_evap_mpln(1.0 - u_old, is_st_cu);
      x_old = utils::min_max_bound(0.0, 1.0, x_old);
      const Real x_new = 0.0;
      resusp_x = haero::max(0.0, precnumx_base_old * (x_old - x_new));
    }
  }
  // setting both these precip rates to zero causes the resuspension
  // calculations to start fresh if there is any more precip production
  precabx_new = 0.0;
  precabx_base_new = 0.0;
}
// ==============================================================================
// ==============================================================================
KOKKOS_INLINE_FUNCTION
void wetdep_scavenging(const int is_st_cu, const bool is_strat_cloudborne,
                       const Real deltat, const Real fracp, const Real precabx,
                       const Real cldv_ik, const Real scavcoef_ik,
                       const Real sol_factb, const Real sol_facti,
                       const Real tracer_1, const Real tracer_2, Real &src,
                       Real &fin) {
  // clang-format off
  // ------------------------------------------------------------------------------
  // do scavenging for both convective and stratiform
  //
  // assuming liquid clouds (no ice)
  //
  // set odds proportional to fraction of the grid box that is swept by the
  // precipitation =precabc/rhoh20*(area of sphere projected on plane
  //                                /volume of sphere)*deltat
  // assume the radius of a raindrop is 1 e-3 m from Rogers and Yau,
  // unless the fraction of the area that is cloud is less than odds, in which
  // case use the cloud fraction (assumes precabs is in kg/m2/s)
  // is really: precabs*3/4/1000./1e-3*deltat
  // here I use .1 from Balkanski
  // ------------------------------------------------------------------------------
  /*
  in :: is_strat_cloudborne   ! if tracer is stratiform-cloudborne aerosol or not
  in :: is_st_cu ! options for stratiform (1) or convective (2) clouds

  in :: deltat       ! timestep [s]
  in :: fracp        ! fraction of cloud water converted to precip [fraction]
  in :: precabx      ! precip from above of the layer [kg/m2/s]
  in :: cldv_ik      ! precipitation area at the top interface [fraction]
  in :: scavcoef_ik  ! Dana and Hales coefficient [1/mm]
  in :: sol_factb    ! solubility factor (frac of aerosol scavenged below cloud) [fraction]
  in :: sol_facti    ! solubility factor (frac of aerosol scavenged in cloud) [fraction]
  in :: tracer_1     ! tracer input for calculate src1 [kg/kg]
  in :: tracer_2     ! tracer input for calculate src2 [kg/kg]
  out :: src         ! total scavenging (incloud + belowcloud) [kg/kg/s]
  out :: fin         ! fraction of incloud scavenging [fraction]

  */
  // clang-format on
  // BAD CONSTANT
  const Real small_value_36 = 1.e-36;
  const Real small_value_5 = 1.e-5; // for cloud fraction

  // calculate limitation of removal rate using Dana and Hales coefficient
  // odds  : limit on removal rate (proportional to prec) [fraction]
  Real odds =
      precabx / haero::max(cldv_ik, small_value_5) * scavcoef_ik * deltat;
  odds = utils::min_max_bound(0.0, 1.0, odds);

  Real src1; // incloud scavenging tendency [kg/kg/s]
  Real src2; // below-cloud scavenging tendency [kg/kg/s]
  if (is_strat_cloudborne) {
    if (is_st_cu == 2) {
      // convective cloud does not affect strat-cloudborne aerosol
      src1 = 0;
    } else {
      // strat in-cloud removal only affects strat-cloudborne aerosol
      // in-cloud scavenging:
      src1 = sol_facti * fracp * tracer_1 / deltat;
    }
    // no below-cloud scavenging for strat-cloudborne aerosol
    src2 = 0;
  } else {
    if (is_st_cu == 2) { // convective
      src1 = sol_facti * fracp * tracer_1 / deltat;
    } else { // stratiform
      // strat in-cloud removal only affects strat-cloudborne aerosol
      src1 = 0;
    }
    src2 = sol_factb * cldv_ik * odds * tracer_2 / deltat;
  }

  src = src1 + src2; // total stratiform or convective scavenging
  fin = src1 / (src + small_value_36); // fraction taken by incloud processes
}
// =============================================================================
// =============================================================================
KOKKOS_INLINE_FUNCTION
void compute_evap_frac(const int mam_prevap_resusp_optcc, const Real pdel_ik,
                       const Real evap_ik, const Real precabx, Real &fracevx) {
  // clang-format off
  //  ------------------------------------------------------------------------------
  //  calculate the fraction of strat precip from above
  //                  which evaporates within this layer
  //  ------------------------------------------------------------------------------
  /*
  in :: mam_prevap_resusp_optcc       ! suspension options
  in :: pdel_ik       ! pressure thikness at current column and level [Pa]
  in :: evap_ik       ! evaporation in this layer [kg/kg/s]
  in :: precabx       ! precipitation from above [kg/m2/s]
  out :: fracevx      ! fraction of evaporation [fraction]
  */
  // clang-format on
  // BAD CONSTANT
  const Real small_value_12 = 1.e-12;
  const Real gravit = Constants::gravity;
  if (mam_prevap_resusp_optcc == 0) {
    fracevx = 0.0;
  } else {
    fracevx = evap_ik * pdel_ik / gravit / haero::max(small_value_12, precabx);
    // trap to ensure reasonable ratio bounds
    fracevx = utils::min_max_bound(0., 1., fracevx);
  }
}
// =============================================================================
// =============================================================================
KOKKOS_INLINE_FUNCTION
Real rain_mix_ratio(const Real temperature, const Real pmid,
                    const Real sumppr) {
  // clang-format off
  // -----------------------------------------------------------------------
  //  Purpose:
  //  calculate rain mixing ratio from precipitation rate above.
  //
  //  extracted from clddiag subroutine
  //  for C++ portint, Shuaiqi Tang in 9/22/2022
  // -----------------------------------------------------------------------
  /*
  in :: temperature      ! temperature [K]
  in :: pmid   ! pressure at layer midpoints [Pa]
  in :: sumppr ! sum of precipitation rate above each layer [kg/m2/s]
  out :: rain ! mixing ratio of rain [kg/kg]
  */
  // clang-format on

  // BAD CONSTANT
  const Real small_value_14 = 1.e-14;
  const Real gravit = Constants::gravity;
  const Real rhoh2o = Constants::density_h2o;
  const Real tmelt = Constants::freezing_pt_h2o;
  const Real rair = Constants::r_gas_dry_air;

  // define the constant convfw. taken from cldwat.F90
  // reference: Tripoli and Cotton (1980)
  // falling velocity at air density = 1 kg/m3 [m/s * sqrt(rho)]
  const Real convfw = 1.94 * 2.13 * haero::sqrt(rhoh2o * gravit * 2.7e-4);

  Real rain = 0;
  if (temperature > tmelt) {
    // rho =air density [kg/m3]
    const Real rho = pmid / (rair * temperature);
    //  vfall = calculated raindrop falling velocity [m/s]
    const Real vfall = convfw / haero::sqrt(rho);
    rain = sumppr / (rho * vfall);
    if (rain < small_value_14)
      rain = 0;
  }
  return rain;
}

// ==============================================================================
// ==============================================================================
KOKKOS_INLINE_FUNCTION
void wetdep_resusp(const int is_st_cu, const int mam_prevap_resusp_optcc,
                   const Real pdel_ik, const Real evapx, const Real precabx_old,
                   const Real precabx_base_old, const Real scavabx_old,
                   const Real precnumx_base_old, Real &precabx_new,
                   Real &precabx_base_new, Real &scavabx_new,
                   Real &precnumx_base_new, Real &resusp_x) {
  // clang-format off
  // ------------------------------------------------------------------------------
  // do precip production, resuspension and scavenging
  // ------------------------------------------------------------------------------
  /*
  in :: is_st_cu ! options for stratiform (1) or convective (2) clouds
                      ! raindrop size distribution is
                      ! different for different cloud:
                      ! 1: assume marshall-palmer distribution
                      ! 2: assume log-normal distribution
  in :: mam_prevap_resusp_optcc       ! suspension options
  in :: pdel_ik       ! pressure thikness at current column and level [Pa]
  in :: evapx         ! evaporation at current layer [kg/kg/s]
  in :: precabx_base_old ! input of precipitation at cloud base [kg/m2/s]
  in :: precabx_old ! input of precipitation above this layer [kg/m2/s]
  in :: scavabx_old ! input of scavenged tracer flux from above [kg/m2/s]
  in :: precnumx_base_old ! input of precipitation number at cloud base [#/m2/s]
  out :: precabx_base_new ! output of precipitation at cloud base [kg/m2/s]
  out :: precabx_new ! output of precipitation above this layer [kg/m2/s]
  out :: scavabx_new ! output of scavenged tracer flux from above [kg/m2/s]
  out :: precnumx_base_new ! output of precipitation number at cloud base [#/m2/s]
  out :: resusp_x    ! aerosol mass re-suspension in a particular layer [kg/m2/s]
  */
  // clang-format on

  // BAD CONSTANT
  const Real small_value_30 = 1.e-30;

  const Real gravit = Constants::gravity;

  // initiate *_new in case they are not calculated
  scavabx_new = scavabx_old;
  precnumx_base_new = precnumx_base_old;
  precabx_base_new = precabx_base_old;

  const Real tmpa = haero::max(0.0, evapx * pdel_ik / gravit);
  precabx_new = utils::min_max_bound(0.0, precabx_base_new, precabx_old - tmpa);

  if (precabx_new < small_value_30) {
    // precip rate is essentially zero so do complete resuspension
    wetdep_resusp_noprecip(is_st_cu, mam_prevap_resusp_optcc, precabx_old,
                           precabx_base_old, scavabx_old, precnumx_base_old,
                           precabx_new, precabx_base_new, scavabx_new,
                           resusp_x);
  } else if (evapx <= 0.0) {
    // no evap so no resuspension
    if (mam_prevap_resusp_optcc <= 130) {
      scavabx_new = scavabx_old;
    }
    resusp_x = 0.0;
  } else {
    // regular non-linear resuspension
    wetdep_resusp_nonlinear(is_st_cu, mam_prevap_resusp_optcc, precabx_old,
                            precabx_base_old, scavabx_old, precnumx_base_old,
                            precabx_new, scavabx_new, resusp_x);
  }
}

// ==============================================================================
// ==============================================================================
KOKKOS_INLINE_FUNCTION
void wetdepa_v2(const Real deltat, const Real pdel, const Real cmfdqr,
                const Real evapc, const Real dlf, const Real conicw,
                const Real precs, const Real evaps, const Real cwat,
                const Real cldt, const Real cldc, const Real cldvcu,
                const Real cldvcu_lower_level, const Real cldvst,
                const Real cldvst_lower_level, const Real sol_factb,
                const Real sol_facti, const Real sol_factic,
                const int mam_prevap_resusp_optcc,
                const bool is_strat_cloudborne, const Real scavcoef,
                const Real f_act_conv, const Real tracer, const Real qqcw,
                Real &fracis, Real &scavt, Real &iscavt, Real &icscavt,
                Real &isscavt, Real &bcscavt, Real &bsscavt, Real &rcscavt,
                Real &rsscavt, Real &precabs, Real &precabc, Real &scavabs,
                Real &scavabc, Real &precabs_base, Real &precabc_base,
                Real &precnums_base, Real &precnumc_base) {
  // clang-format off
  // -----------------------------------------------------------------------
  //  Purpose:
  //  scavenging code for very soluble aerosols
  //
  //  Author: P. Rasch
  //  Modified by T. Bond 3/2003 to track different removals
  //  Sungsu Park. Mar.2010 : Impose consistencies with a few changes in physics.

  //  this section of code is for highly soluble aerosols,
  //  the assumption is that within the cloud that
  //  all the tracer is in the cloud water
  //
  //  for both convective and stratiform clouds,
  //  the fraction of cloud water converted to precip defines
  //  the amount of tracer which is pulled out.
  // -----------------------------------------------------------------------
  /*
  in ::
         deltat,   ! time step [s]
         pdel,     ! pressure thikness [Pa]
         cmfdqr,   ! rate of production of convective precip [kg/kg/s]
         evapc,    ! Evaporation rate of convective precipitation [kg/kg/s]
         dlf,      ! Detrainment of convective condensate [kg/kg/s]
         conicw,   ! convective cloud water [kg/kg]
         precs,    ! rate of production of stratiform precip [kg/kg/s]
         evaps,    ! rate of evaporation of precip [kg/kg/s]
         cwat,     ! cloud water amount [kg/kg]
         cldt,     ! total cloud fraction [fraction]
         cldc,     ! convective cloud fraction [fraction]
         cldvcu,   ! Convective precipitation area at the top interface of each layer [fraction]
         cldvcu_lower_level  Convective precipitation at the next lower level, (kk+1 relative to cldvcu[kk]
	                     or at cldvcu[nlev-1] if kk==nlev) area at the top interface of each layer [fraction]
         cldvst,   ! Stratiform precipitation area at the top interface of each layer [fraction]
         cldvst_lower_level, Stratiform precipitation at the next lower level, (kk+1 relative to cldvst[kk]
                             or at cldvst[nlev-1] if kk==nlev)area at the top interface of each layer [fraction]
         tracer    ! trace species [kg/kg]

  in :: mam_prevap_resusp_optcc ! suspension options.
     0 = no resuspension
     1 = linear resuspension of aerosol mass or number following original mam
         coding and history_aero_prevap_resusp = .false.
     2 = same as 1 but history_aero_prevap_resusp = .true.
     3 = same as 2 but with some added "xxx = max( 0, xxx)" lines
   130 = non-linear resuspension of aerosol mass   based on scavenged aerosol mass
   230 = non-linear resuspension of aerosol number based on raindrop number
     (1,2,3 are not used in the current code)


  in :: is_strat_cloudborne = true if tracer is stratiform-cloudborne aerosol; else false

  in :: scavcoef ! Dana and Hales coefficient [1/mm]
  in :: f_act_conv ! [fraction] ! f_act_conv = conv-cloud activation fraction when is_strat_cloudborne==.false.; else 0.0
  in :: qqcw  ! [kg/kg] ! qqcw = strat-cloudborne aerosol corresponding to tracer when is_strat_cloudborne==.false.; else 0.0
  in :: sol_factb   ! solubility factor (frac of aerosol scavenged below cloud) [fraction]
  in :: sol_facti   ! solubility factor (frac of aerosol scavenged in cloud) [fraction]
  in :: sol_factic  ! sol_facti for convective clouds [fraction]

  out :: fracis  ! fraction of species not scavenged [fraction]
  out :: scavt   ! scavenging tend [kg/kg/s]
  out :: iscavt  ! incloud scavenging tends [kg/kg/s]
  out :: icscavt  ! incloud, convective [kg/kg/s]
  out :: isscavt  ! incloud, stratiform [kg/kg/s]
  out :: bcscavt  ! below cloud, convective [kg/kg/s]
  out :: bsscavt  ! below cloud, stratiform [kg/kg/s]
  out :: rcscavt  ! resuspension, convective [kg/kg/s]
  out :: rsscavt  ! resuspension, stratiform [kg/kg/s]
  */
  // clang-format on
#if 0
      ! local variables
      integer  :: icol          ! column index
      integer  :: kk            ! z index

#endif
  // BAD CONSTANT
  const Real small_value_2 = 1.e-2;
  const Real small_value_12 = 1.e-12;
  const Real small_value_36 = 1.e-36;

  fracis = scavt = iscavt = icscavt = isscavt = bcscavt = bsscavt = rcscavt =
      rsscavt = 0;

  // omsm = (1 - small number) used to prevent roundoff errors below zero
  // in Fortran EPSILON(X) returns the smallest number E of the same kind
  // as X such that 1 + E > 1.
  // C++ Returns the machine epsilon, that is, the difference between 1.0
  // and the next value representable by the floating-point type T.
  const Real omsm = 1.0 - 2 * haero::epsilon();
  // precabs, precabc, scavabs, scavabc, precabs_base, precabc_base,
  // precnums_base, and precnumc_base are input/output in this routine.
  // precabs: strat precip from above [kg/m2/s]
  // precabc: conv precip from above [kg/m2/s]
  // scavabs: stratiform scavenged tracer flux from above [kg/m2/s]
  // scavabc: convective scavenged tracer flux from above [kg/m2/s]
  // precabs_base: strat precip at an effective cloud base for calculations in a
  // particular layer [kg/m2/s] precabc_base: conv precip at an effective cloud
  // base for calculations in a particular layer [kg/m2/s] precnums_base:
  // stratiform precip number flux at the bottom of a particular layer [#/m2/s]
  // precnumc_base: convective precip number flux at the bottom of a particular
  // layer [#/m2/s]
  // ****************** Evaporation **************************
  // fraction of stratiform precip from above that is evaporating [fraction]
  Real fracev_st;
  // Fraction of convective precip from above that is evaporating [fraction]
  Real fracev_cu;
  // stratiform
  compute_evap_frac(mam_prevap_resusp_optcc, pdel, evaps, precabs, fracev_st);
  // convective
  compute_evap_frac(mam_prevap_resusp_optcc, pdel, evapc, precabc, fracev_cu);

  // ****************** Scavenging **************************

  // temporary saved tracer value
  const Real clddiff = cldt - cldc;
  // temporarily calculation of tracer [kg/kg]
  const Real tracer_tmp = haero::min(
      qqcw, tracer * (clddiff / haero::max(small_value_2, (1. - clddiff))));
  // calculate in-cumulus and mean tracer values for wetdep_scavenging use
  // in-cumulus tracer concentration [kg/kg]
  const Real tracer_incu = f_act_conv * (tracer + tracer_tmp);
  // mean tracer concenration [kg/kg]
  Real tracer_mean =
      tracer * (1. - cldc * f_act_conv) - cldc * f_act_conv * tracer_tmp;
  tracer_mean = haero::max(0., tracer_mean);

  // now do the convective scavenging

  // fracp: fraction of convective cloud water converted to rain
  // Sungsu: Below new formula of 'fracp' is necessary since 'conicw'
  // is a LWC/IWC that has already precipitated out, that is, 'conicw' does
  // not contain precipitation at all !
  Real fracp =
      cmfdqr * deltat /
      haero::max(small_value_12, cldc * conicw + (cmfdqr + dlf) * deltat);
  fracp = utils::min_max_bound(0.0, 1.0, fracp) * cldc;

  Real srcc = 0; // tendency for convective rain scavenging [kg/kg/s]
  Real finc = 0; // fraction of rem. rate by conv. rain [fraction]
  // 2 is for convective:
  wetdep_scavenging(2, is_strat_cloudborne, deltat, fracp, precabc, cldvcu,
                    scavcoef, sol_factb, sol_factic, tracer_incu, tracer_mean,
                    srcc, finc);

  // now do the stratiform scavenging

  // fracp: fraction of convective cloud water converted to rain
  fracp = precs * deltat / haero::max(cwat + precs * deltat, small_value_12);
  fracp = utils::min_max_bound(0.0, 1.0, fracp);

  Real srcs; // tendency for stratiform rain scavenging [kg/kg/s]
  Real fins; // fraction of rem. rate by strat rain [fraction]
  // 1 for stratiform:
  wetdep_scavenging(1, is_strat_cloudborne, deltat, fracp, precabs, cldvst,
                    scavcoef, sol_factb, sol_facti, tracer, tracer_mean, srcs,
                    fins);

  // rat =  ratio of amount available to amount removed [fraction]
  // make sure we dont take out more than is there
  // ratio of amount available to amount removed
  const Real rat = tracer / haero::max(deltat * (srcc + srcs), small_value_36);
  if (rat < 1) {
    srcs = srcs * rat;
    srcc = srcc * rat;
  }
  // total scavenging tendency [kg/kg/s]
  const Real srct = (srcc + srcs) * omsm;

  // fraction that is not removed within the cloud
  // (assumed to be interstitial, and subject to convective transport)
  fracp = deltat * srct / haero::max(cldvst * tracer, small_value_36);
  fracis = 1. - utils::min_max_bound(0.0, 1.0, fracp);

  // ****************** Resuspension **************************

  Real resusp_c; // aerosol mass re-suspension in a particular layer from
                 // convective rain [kg/m2/s]
  Real resusp_s; // aerosol mass re-suspension in a particular layer from
                 // stratiform rain [kg/m2/s]
  // tend is all tracer removed by scavenging, plus all re-appearing from
  // evaporation above
  if (mam_prevap_resusp_optcc >= 100) {
    // for stratiform clouds
    // precipitation and cloudy volume,at the top interface of current layer
    // [fraction]
    Real arainx = haero::max(cldvst_lower_level, small_value_2); // non-zero
    Real precabx_tmp = 0;       // temporary store precabc or precabs [kg/m2/s]
    Real precabx_base_tmp = 0;  // temporarily store precab*_base [kg/m2/s]
    Real precnumx_base_tmp = 0; // temporarily store precnum*_base [#/m2/s]
    Real scavabx_tmp = 0;       // temporarily store scavab* [kg/m2/s]
    // step 1 - do evaporation and resuspension
    wetdep_resusp(1, mam_prevap_resusp_optcc, pdel, evaps, precabs,
                  precabs_base, scavabs, precnums_base, precabx_tmp,
                  precabx_base_tmp, scavabx_tmp, precnumx_base_tmp, resusp_s);
    // step 2 - do precip production and scavenging
    wetdep_prevap(1, mam_prevap_resusp_optcc, pdel, precs, srcs, arainx,
                  precabx_tmp, precabx_base_tmp, scavabx_tmp, precnumx_base_tmp,
                  precabs, precabs_base, scavabs, precnums_base);

    // for convective clouds
    arainx = haero::max(cldvcu_lower_level, small_value_2); // non-zero
    wetdep_resusp(2, mam_prevap_resusp_optcc, pdel, evapc, precabc,
                  precabc_base, scavabc, precnumc_base, precabx_tmp,
                  precabx_base_tmp, scavabx_tmp, precnumx_base_tmp, resusp_c);
    // step 2 - do precip production and scavenging
    wetdep_prevap(2, mam_prevap_resusp_optcc, pdel, cmfdqr, srcc, arainx,
                  precabx_tmp, precabx_base_tmp, scavabx_tmp, precnumx_base_tmp,
                  precabc, precabc_base, scavabc, precnumc_base);
  } else { // mam_prevap_resusp_optcc = 0, no resuspension
    resusp_c = fracev_cu * scavabc;
    resusp_s = fracev_st * scavabs;
  }

  // ****************** update scavengingfor output ***************
  Real scavt_ik = 0;  // scavenging tend at current  [kg/kg/s]
  Real iscavt_ik = 0; // incloud scavenging tends at current  [kg/kg/s]
  Real icscavt_ik =
      0; // incloud, convective scavenging tends at current  [kg/kg/s]
  Real isscavt_ik =
      0; // incloud, stratiform scavenging tends at current  [kg/kg/s]
  Real bcscavt_ik = 0; // below cloud, convective scavenging tends at current
                       // [kg/kg/s]
  Real bsscavt_ik = 0; // below cloud, stratiform scavenging tends at current
                       // [kg/kg/s]
  Real rcscavt_ik = 0; // resuspension, convective tends at current  [kg/kg/s]
  Real rsscavt_ik = 0; // resuspension, stratiform tends at current  [kg/kg/s]
  update_scavenging(mam_prevap_resusp_optcc, pdel, omsm, srcc, srcs, srct, fins,
                    finc, fracev_st, fracev_cu, resusp_c, resusp_s, precs,
                    evaps, cmfdqr, evapc, scavt_ik, iscavt_ik, icscavt_ik,
                    isscavt_ik, bcscavt_ik, bsscavt_ik, rcscavt_ik, rsscavt_ik,
                    scavabs, scavabc, precabc, precabs);

  scavt = scavt_ik;
  iscavt = iscavt_ik;
  icscavt = icscavt_ik;
  isscavt = isscavt_ik;
  bcscavt = bcscavt_ik;
  bsscavt = bsscavt_ik;
  rcscavt = rcscavt_ik;
  rsscavt = rsscavt_ik;
}
// ==============================================================================

/**
 * @brief Estimate the cloudy volume which is occupied by rain or cloud water as
 *        the max between the local cloud amount or the sum above of
 *       (cloud * positive precip production)   sum total precip from above
 *        ----------------------------------- X -------------------------
 *        sum above of ( positive precip )      sum positive precip from above
 *
 * @param[in] temperature Temperature [K].
 * @param[in] pmid Pressure at layer midpoints [Pa].
 * @param[in] pdel Pressure difference across layers [Pa].
 * @param[in] cmfdqr to convective rainout [kg/kg/s].
 * @param[in] evapc Evaporation rate of convective precipitation ( >= 0 )
 * [kg/kg/s].
 * @param[in] cldt Total cloud fraction [fraction, unitless].
 * @param[in] cldcu Cumulus cloud fraction [fraction, unitless].
 * @param[in] clst Stratus cloud fraction [fraction, unitless].
 * @param[in] evapr rate of evaporation of falling precipitation [kg/kg/s].
 * @param[in] prain rate of conversion of condensate to precipitation [kg/kg/s].
 * @param[in] atm Atmosphere object (used for number of levels).
 *
 * @param[out] cldv Fraction occupied by rain or cloud water [fraction,
 * unitless].
 * @param[out] cldvcu Convective precipitation volume [fraction, unitless].
 * @param[out] cldvst Stratiform precipitation volume [fraction, unitless].
 * @param[out] rain Rain mixing ratio [kg/kg].
 *
 * @pre In F90, ncol == 1 as we only operate over one column at a time
 *      as outer loops will iterate over columns, so we drop ncol as input.
 * @pre In F90 code, pcols is the number of columns in the mesh.
 *      Since we are only operating over one column, pcols == 1.
 * @pre In F90 code, ncol == pcols. Since ncol == 1, pcols == 1.
 *
 * @pre atm is initialized correctly and has the correct number of levels.
 */
KOKKOS_INLINE_FUNCTION
void clddiag(const int nlev, const Real *temperature, const Real *pmid,
             const Real *pdel, const Real *cmfdqr, const Real *evapc,
             const Real *cldt, const Real *cldcu, const Real *cldst,
             const Real *evapr, const Real *prain, Real *cldv, Real *cldvcu,
             Real *cldvst, Real *rain) {
  // Calculate local precipitation production rate
  // In src/chemistry/aerosol/wetdep.F90, (prain + cmfdqr) is used for
  // source_term

  // TODO - !FIXME: Possible bug: why there is no evapc in sumppr_all
  // calculation?
  // FIXME: Do we need a parallel_reduce ?
  Real sumppr_all = 0;
  for (int i = 0; i < nlev; i++) {
    const Real source_term = prain[i] + cmfdqr[i];
    sumppr_all += local_precip_production(pdel[i], source_term, evapr[i]);
    // Calculate rain mixing ratio
    rain[i] = rain_mix_ratio(temperature[i], pmid[i], sumppr_all);
  }

  // Calculate cloudy volume which is occupied by rain or cloud water
  // Total
  auto prec = [&](int i) -> Real {
    const Real source_term = prain[i] + cmfdqr[i];
    return local_precip_production(pdel[i], source_term, evapr[i]);
  };
  calculate_cloudy_volume(nlev, cldt, prec, true, cldv);

  // Convective
  auto prec_cu = [&](int i) -> Real {
    return local_precip_production(pdel[i], cmfdqr[i], evapc[i]);
  };
  calculate_cloudy_volume(nlev, cldcu, prec_cu, false, cldvcu);

  // Stratiform
  auto prec_st = [&](int i) -> Real {
    return local_precip_production(pdel[i], prain[i], evapr[i]);
  };
  calculate_cloudy_volume(nlev, cldst, prec_st, false, cldvst);
}

template <typename VIEWTYPE>
KOKKOS_INLINE_FUNCTION void sum_values(const ThreadTeam &team,
                                       const View1D &sum, VIEWTYPE x,
                                       VIEWTYPE y, const int nlev) {
  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, nlev),
                       [&](int k) { sum[k] = x[k] + y[k]; });
}
KOKKOS_INLINE_FUNCTION
void zero_values(const ThreadTeam &team, const View1D &vec, const int nlev) {
  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, nlev),
                       [&](int k) { vec[k] = 0; });
}

KOKKOS_INLINE_FUNCTION
void sum_deep_and_shallow(const ThreadTeam &team, const View1D &conicw,
                          const ConstView1D &icwmrdp,
                          const ConstView1D &dp_frac,
                          const ConstView1D &icwmrsh,
                          const ConstView1D &sh_frac, const int nlev) {
  // BAD CONSTANT
  const Real small_value_2 = 1.e-2;
  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, nlev), [&](int k) {
    // sum deep and shallow convection contributions
    conicw[k] = (icwmrdp[k] * dp_frac[k] + icwmrsh[k] * sh_frac[k]) /
                haero::max(small_value_2, sh_frac[k] + dp_frac[k]);
  });
}

KOKKOS_INLINE_FUNCTION
void cloud_diagnostics(const ThreadTeam &team,
                       haero::ConstColumnView temperature,
                       haero::ConstColumnView pmid, haero::ConstColumnView pdel,
                       const View1D &cmfdqr, const View1D &evapc,
                       const haero::ConstColumnView &cldt, const View1D &cldcu,
                       const View1D &cldst, const haero::ConstColumnView &evapr,
                       const haero::ConstColumnView &prain,
                       // outputs
                       const View1D &cldv, const View1D &cldvcu,
                       const View1D &cldvst, const View1D &rain,
                       const int nlev) {
  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, 1), [&](int k) {
    wetdep::clddiag(nlev, temperature.data(), pmid.data(), pdel.data(),
                    cmfdqr.data(), evapc.data(), cldt.data(), cldcu.data(),
                    cldst.data(), evapr.data(), prain.data(),
                    // outputs
                    cldv.data(), cldvcu.data(), cldvst.data(), rain.data());
  });
}

KOKKOS_INLINE_FUNCTION
void set_f_act(const ThreadTeam &team, const Kokkos::View<int *> &isprx,
               const View1D &f_act_conv_coarse,
               const View1D &f_act_conv_coarse_dust,
               const View1D &f_act_conv_coarse_nacl,
               haero::ConstColumnView pdel, haero::ConstColumnView prain,
               const View1D &cmfdqr, const ConstView1D &evapr,
               const View2D &state_q, const View2D &ptend_q, const Real dt,
               const int nlev) {

  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, nlev), [&](int k) {
    isprx[k] = aero_model::examine_prec_exist(k, pdel.data(), prain.data(),
                                              cmfdqr.data(), evapr.data());

    aero_model::set_f_act_coarse(k, state_q, ptend_q, dt, f_act_conv_coarse[k],
                                 f_act_conv_coarse_dust[k],
                                 f_act_conv_coarse_nacl[k]);
  });
}

// Computes lookup table for aerosol impaction/interception scavenging rates
KOKKOS_INLINE_FUNCTION
void modal_aero_bcscavcoef_get(
    const ThreadTeam &team, const Diagnostics &diags,
    const Kokkos::View<int *> &isprx,
    const Real scavimptblvol[aero_model::nimptblgrow_total]
                            [AeroConfig::num_modes()],
    const Real scavimptblnum[aero_model::nimptblgrow_total]
                            [AeroConfig::num_modes()],
    const View1D &scavcoefnum, const View1D &scavcoefvol, const int imode,
    const int nlev) {
  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, nlev), [&](int k) {
    scavcoefnum[k] = scavcoefvol[k] = 0;
    const bool let_it_rain = (isprx[k] == 1);
    if (let_it_rain) {
      const Real dgnum_amode_imode = modes(imode).nom_diameter;
      ColumnView dgn_awet_imode = diags.wet_geometric_mean_diameter_i[imode];
      const Real dgn_awet_imode_k = dgn_awet_imode[k];
      aero_model::modal_aero_bcscavcoef_get(
          imode, dgn_awet_imode_k, dgnum_amode_imode, scavimptblvol,
          scavimptblnum, scavcoefnum[k], scavcoefvol[k]);
    }
  });
}

// Computes lookup table for aerosol impaction/interception scavenging rates
KOKKOS_INLINE_FUNCTION
void modal_aero_bcscavcoef_get(
    const ThreadTeam &team, const View2D &wet_geometric_mean_diameter_i,
    const Kokkos::View<int *> &isprx,
    const Real scavimptblvol[aero_model::nimptblgrow_total]
                            [AeroConfig::num_modes()],
    const Real scavimptblnum[aero_model::nimptblgrow_total]
                            [AeroConfig::num_modes()],
    const View1D &scavcoefnum, const View1D &scavcoefvol, const int imode,
    const int nlev) {
  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, nlev), [&](int k) {
    scavcoefnum[k] = scavcoefvol[k] = 0;
    const bool let_it_rain = (isprx[k] == 1);
    if (let_it_rain) {
      const Real dgnum_amode_imode = modes(imode).nom_diameter;
      const Real dgn_awet_imode_k = wet_geometric_mean_diameter_i(imode, k);
      aero_model::modal_aero_bcscavcoef_get(
          imode, dgn_awet_imode_k, dgnum_amode_imode, scavimptblvol,
          scavimptblnum, scavcoefnum[k], scavcoefvol[k]);
    }
  });
}

// define sol_factb and sol_facti values, and f_act_conv
KOKKOS_INLINE_FUNCTION
void define_act_frac(const ThreadTeam &team, const View1D &sol_facti,
                     const View1D &sol_factic, const View1D &sol_factb,
                     const View1D &f_act_conv, const int lphase,
                     const int imode, const int nlev) {
  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, nlev), [&](int k) {
    aero_model::define_act_frac(lphase, imode, sol_facti[k], sol_factic[k],
                                sol_factb[k], f_act_conv[k]);
  });
}

KOKKOS_INLINE_FUNCTION
void compute_q_tendencies_phase_1(
    Real &scavt, Real &bcscavt, Real &rcscavt, Real rtscavt_sv[],
    const Real f_act_conv, const Real scavcoefnum, const Real scavcoefvol,
    const Real totcond, const Real cmfdqr, const Real conicw, const Real evapc,
    const Real evapr, const Real prain, const Real dlf, const Real cldt,
    const Real cldcu, const Real cldvst_k, const Real cldvst_k_p1,
    const Real cldvcu_k, const Real cldvcu_k_p1, const Real sol_facti,
    const Real sol_factic, const Real sol_factb, const Real state_q,
    const Real ptend_q, const Real qqcw_sav, const Real pdel, const Real dt,
    const int mam_prevap_resusp_optcc, const int jnv, const int mm,
    Real &precabs, Real &precabc, Real &scavabs, Real &scavabc,
    Real &precabs_base, Real &precabc_base, Real &precnums_base,
    Real &precnumc_base) {
  // traces reflects changes from modal_aero_calcsize and is the
  // "most current" q
  const Real tracer = state_q + ptend_q * dt;
  Real scavcoef = 0;
  if (jnv)
    scavcoef = (1 == jnv) ? scavcoefnum : scavcoefvol;

  Real fracis = 0;  // fraction of transported species that are insoluble
  Real iscavt = 0;  // incloud scavenging tends [kg/kg/s]
  Real icscavt = 0; // incloud, convective [kg/kg/s]
  Real isscavt = 0; // incloud, stratiform [kg/kg/s]
  Real bsscavt = 0; // below cloud, stratiform [kg/kg/s]
  Real rsscavt = 0; // resuspension, stratiform [kg/kg/s]
  // is_strat_cloudborne = true if tracer is
  // stratiform-cloudborne aerosol; else false
  const bool is_strat_cloudborne = false;
  wetdep::wetdepa_v2(
      dt, pdel, cmfdqr, evapc, dlf, conicw, prain, evapr, totcond, cldt, cldcu,
      cldvcu_k, cldvcu_k_p1, cldvst_k, cldvst_k_p1, sol_factb, sol_facti,
      sol_factic, mam_prevap_resusp_optcc, is_strat_cloudborne, scavcoef,
      f_act_conv, tracer, qqcw_sav, fracis, scavt, iscavt, icscavt, isscavt,
      bcscavt, bsscavt, rcscavt, rsscavt, precabs, precabc, scavabs, scavabc,
      precabs_base, precabc_base, precnums_base, precnumc_base);
  // resuspension goes to coarse mode
  const bool update_dqdt = true;
  aero_model::calc_resusp_to_coarse(mm, update_dqdt, rcscavt, rsscavt, scavt,
                                    rtscavt_sv);
}

KOKKOS_INLINE_FUNCTION
void compute_q_tendencies_phase_2(
    Real &scavt, Real &bcscavt, Real &rcscavt, Real rtscavt_sv[],
    const Real qqcw_tmp, const Real tracer,

    // const Prognostics &progs,
    const Real f_act_conv, const Real scavcoefnum, const Real scavcoefvol,
    const Real totcond, const Real cmfdqr, const Real conicw, const Real evapc,
    const Real evapr, const Real prain, const Real dlf, const Real cldt,
    const Real cldcu, const Real cldvst_k, const Real cldvst_k_p1,
    const Real cldvcu_k, const Real cldvcu_k_p1, const Real sol_facti,
    const Real sol_factic, const Real sol_factb, const Real pdel, const Real dt,
    const int mam_prevap_resusp_optcc, const int jnv, const int mm, const int k,
    Real &precabs, Real &precabc, Real &scavabs, Real &scavabc,
    Real &precabs_base, Real &precabc_base, Real &precnums_base,
    Real &precnumc_base) {

  // static constexpr int pcnst = aero_model::pcnst;
  // There is no cloud-borne aerosol water in the model, so this
  // code block should NEVER execute for lspec =
  // nspec_amode(m)+1 (i.e., jnummaswtr = 2). The code only
  // worked because the "do lspec" loop cycles when lspec =
  // nspec_amode(m)+1, but that does not make the code correct.
  // qqcw_sav = tracer;
  Real fracis = 0;  // fraction of species not scavenged [fraction]
  Real iscavt = 0;  // incloud scavenging tends [kg/kg/s]
  Real icscavt = 0; // incloud, convective [kg/kg/s]
  Real isscavt = 0; // incloud, stratiform [kg/kg/s]
  Real bsscavt = 0; // below cloud, stratiform [kg/kg/s]
  Real rsscavt = 0; // resuspension, stratiform [kg/kg/s]
  Real scavcoef = 0;
  if (jnv)
    scavcoef = (1 == jnv) ? scavcoefnum : scavcoefvol;
  const bool is_strat_cloudborne = true;
  wetdep::wetdepa_v2(
      dt, pdel, cmfdqr, evapc, dlf, conicw, prain, evapr, totcond, cldt, cldcu,
      cldvcu_k, cldvcu_k_p1, cldvst_k, cldvst_k_p1, sol_factb, sol_facti,
      sol_factic, mam_prevap_resusp_optcc, is_strat_cloudborne, scavcoef,
      f_act_conv, tracer, qqcw_tmp, fracis, scavt, iscavt, icscavt, isscavt,
      bcscavt, bsscavt, rcscavt, rsscavt, precabs, precabc, scavabs, scavabc,
      precabs_base, precabc_base, precnums_base, precnumc_base);

  // resuspension goes to coarse mode
  const bool update_dqdt = false;
  aero_model::calc_resusp_to_coarse(mm, update_dqdt, rcscavt, rsscavt, scavt,
                                    rtscavt_sv);

  // Setting ptend_q is the same as the Fortran version:
  // qqcw_all[mm] += scavt[k] * dt;
  // utils::inject_qqcw_to_prognostics(qqcw_all, progs, k);
}

KOKKOS_INLINE_FUNCTION
void compute_q_tendencies(
    const ThreadTeam &team,
    // const Prognostics &progs,
    const View1D &f_act_conv, const View1D &f_act_conv_coarse,
    const View1D &f_act_conv_coarse_dust, const View1D &f_act_conv_coarse_nacl,
    const View1D &scavcoefnum, const View1D &scavcoefvol, const View1D &totcond,
    const View1D &cmfdqr, const View1D &conicw, const View1D &evapc,
    const ConstView1D &evapr, const ConstView1D &prain, const ConstView1D &dlf,
    const ConstView1D &cldt, const View1D &cldcu, const View1D &cldst,
    const View1D &cldvst, const View1D &cldvcu, const View1D &sol_facti,
    const View1D &sol_factic, const View1D &sol_factb, const View1D &scavt,
    const View1D &bcscavt, const View1D &rcscavt, const View2D &rtscavt_sv,
    const View2D &state_q, const View2D &qqcw, const View2D &ptend_q,
    // Kokkos::View<Real * [aero_model::maxd_aspectype + 2][aero_model::pcnst]>
    //     qqcw_sav,
    haero::ConstColumnView pdel, const Real dt, const int jnummaswtr,
    const int jnv, const int mm, const int lphase, const int imode,
    const int lspec) {

  // clang-format off
  //   0 = no resuspension
  // 130 = non-linear resuspension of aerosol mass based on scavenged aerosol mass
  // 230 = non-linear resuspension of aerosol number based on raindrop number
  // the 130 thru 230 all use the new prevap_resusp code block in subr wetdepa_v2
  // clang-format on
  const int mam_prevap_resusp_no = 0;
  const int mam_prevap_resusp_mass = 130;
  const int mam_prevap_resusp_num = 230;
  const int jaeronumb = 0, jaeromass = 1;
  Real precabs = 0;
  Real precabc = 0;
  Real scavabs = 0;
  Real scavabc = 0;
  Real precabs_base = 0;
  Real precabc_base = 0;
  Real precnums_base = 0;
  Real precnumc_base = 0;
  // NOTE: The following k loop cannot be converted to parallel_for
  // because precabs requires values from the previous elevation (k-1).
  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, 1), [&](int idummy) {
    for (int k = 0; k < nlev; ++k) {
      const auto rtscavt_sv_k = ekat::subview(rtscavt_sv, k);
      // mam_prevap_resusp_optcc values control the prevap_resusp
      // calculations in wetdepa_v2:
      //     0 = no resuspension
      //   130 = non-linear resuspension of aerosol mass   based on
      //   scavenged aerosol mass 230 = non-linear resuspension of
      //   aerosol number based on raindrop number the 130 thru 230
      //   all use the new prevap_resusp code block in subr wetdepa_v2
      int mam_prevap_resusp_optcc = mam_prevap_resusp_no;
      const int modeptr_coarse = static_cast<int>(ModeIndex::Coarse);
      if (jnummaswtr == jaeromass) // dry mass
        mam_prevap_resusp_optcc = mam_prevap_resusp_mass;
      else if (jnummaswtr == jaeronumb && lphase == 1 &&
               imode == modeptr_coarse) // number
        mam_prevap_resusp_optcc = mam_prevap_resusp_num;

      // set f_act_conv for interstitial (lphase=1) coarse mode
      // species for the convective in-cloud, we conceptually treat
      // the coarse dust and seasalt as being externally mixed, and
      // apply f_act_conv = f_act_conv_coarse_dust/nacl to
      // dust/seasalt number and sulfate are conceptually partitioned
      // to the dust and seasalt on a mass basis, so the f_act_conv
      // for number and sulfate are mass-weighted averages of the
      // values used for dust/seasalt
      if (lphase == 1 && imode == modeptr_coarse) {
        f_act_conv[k] = f_act_conv_coarse[k];
        if (jnummaswtr == jaeromass) {
          if (aero_model::lmassptr_amode(lspec, imode) ==
              aero_model::lptr_dust_a_amode(imode))
            f_act_conv[k] = f_act_conv_coarse_dust[k];
          else if (aero_model::lmassptr_amode(lspec, imode) ==
                   aero_model::lptr_nacl_a_amode(imode))
            f_act_conv[k] = f_act_conv_coarse_nacl[k];
        }
      }
      const int k_p1 = static_cast<int>(haero::min(k + 1, nlev - 1));
      // OK, this is from the old mam4: Phase 2 is before Phase 1.
      // Note that the phase loops goes from 2 to 1 in reverse order
      // and the qqcw_sav is set first in phase 2 the used in phase 1.
      if (lphase == 1) {
        // traces reflects changes from modal_aero_calcsize and is the
        // "most current" q
        compute_q_tendencies_phase_1(
            // These are the output values
            scavt[k], bcscavt[k], rcscavt[k], rtscavt_sv_k.data(),
            // The rest of the values are input only.
            f_act_conv[k], scavcoefnum[k], scavcoefvol[k], totcond[k],
            cmfdqr[k], conicw[k], evapc[k], evapr[k], prain[k], dlf[k], cldt[k],
            cldcu[k], cldvst[k], cldvst[k_p1], cldvcu[k], cldvcu[k_p1],
            sol_facti[k], sol_factic[k], sol_factb[k], state_q(k, mm),
            ptend_q(k, mm), qqcw(k, mm), pdel[k], dt, mam_prevap_resusp_optcc,
            jnv, mm, precabs, precabc, scavabs, scavabc, precabs_base,
            precabc_base, precnums_base, precnumc_base);

      } else { // if (lphase == 2)
        // There is no cloud-borne aerosol water in the model, so this
        // code block should NEVER execute for lspec =
        // nspec_amode(m)+1 (i.e., jnummaswtr = 2). The code only
        // worked because the "do lspec" loop cycles when lspec =
        // nspec_amode(m)+1, but that does not make the code correct.
        // FIXME: Not sure if this is a bug or not as qqcw_tmp seem
        // different from the previous call and qqcw_tmp is always
        // zero. May need further check.  - Shuaiqi Tang in
        // refactoring for MAM4xx
        const Real qqcw_tmp = 0.0;
        compute_q_tendencies_phase_2(
            // These are the output values
            scavt[k], bcscavt[k], rcscavt[k], rtscavt_sv_k.data(), qqcw_tmp,
            qqcw(k, mm),
            // The rest of the values are input only.
            // progs,
            f_act_conv[k], scavcoefnum[k], scavcoefvol[k], totcond[k],
            cmfdqr[k], conicw[k], evapc[k], evapr[k], prain[k], dlf[k], cldt[k],
            cldcu[k], cldvst[k], cldvst[k_p1], cldvcu[k], cldvcu[k_p1],
            sol_facti[k], sol_factic[k], sol_factb[k], pdel[k], dt,
            mam_prevap_resusp_optcc, jnv, mm, k, precabs, precabc, scavabs,
            scavabc, precabs_base, precabc_base, precnums_base, precnumc_base);
      }
    }
  });
}

KOKKOS_INLINE_FUNCTION
void update_q_tendencies(const ThreadTeam &team, const View2D &ptend_q,
                         const View1D &scavt, const int mm, const int nlev) {
  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, nlev), [&](int k) {
    Kokkos::atomic_add(&ptend_q(k, mm), scavt[k]);
  });
}

// =============================================================================
KOKKOS_INLINE_FUNCTION
int get_aero_model_wetdep_work_len() {
  // wet_geometric_mean_diameter_i + state_q + qqcw
  int work_len =
      // mam4::nlev * AeroConfig::num_modes() * mam4::nlev + //
      2 * mam4::nlev * pcnst + // state_q + qqcw
      25 * mam4::nlev +        // cldcu, cldt, evapc, cmfdqr,
                        // prain, totcond, conicw, isprx, f_act_conv_coarse,
      // f_act_conv_coarse_dust, f_act_conv_coarse_nacl
      // rain, ptend_q, cldv, cldvcu, cldvst, scavcoefnum, scavcoefvol
      // sol_facti, sol_factic, sol_factb, f_act_conv, scavt, rcscavt, bcscavt
      3 * pcnst +             //  qsrflx_mzaer2cnvpr, rtscavt_sv
      2 * mam4::nlev * pcnst; // ptend_q, rtscavt_sv
                              // dry_geometric_mean_diameter_i, qaerwat, wetdens
  return work_len;
}
// =============================================================================

KOKKOS_INLINE_FUNCTION
void aero_model_wetdep(
    const ThreadTeam &team, const Atmosphere &atm, Prognostics &progs,
    Tendencies &tends, const Real dt,
    // inputs
    const haero::ConstColumnView &cldt, const haero::ConstColumnView &rprdsh,
    const haero::ConstColumnView &rprddp, const haero::ConstColumnView &evapcdp,
    const haero::ConstColumnView &evapcsh,
    const haero::ConstColumnView &dp_frac,
    const haero::ConstColumnView &sh_frac,
    const haero::ConstColumnView &icwmrdp,
    const haero::ConstColumnView &icwmrsh, const haero::ConstColumnView &evapr,
    const haero::ConstColumnView &dlf, const haero::ConstColumnView &prain,
    // const Int1D &isprx,
    const Real scavimptblnum[aero_model::nimptblgrow_total]
                            [AeroConfig::num_modes()],
    const Real scavimptblvol[aero_model::nimptblgrow_total]
                            [AeroConfig::num_modes()],
    // in/out calcsize and water_uptake
    const View2D &wet_geometric_mean_diameter_i,
    const View2D &dry_geometric_mean_diameter_i, const View2D &qaerwat,
    const View2D &wetdens,
    // output
    const View1D &aerdepwetis, const View1D &aerdepwetcw, const View1D &work) {
  // cldn layer cloud fraction [fraction]; CLD

  // FIXME: do we need to set the variables inside of set_srf_wetdep ?
  // aerdepwetis aerosol_wet_deposition_interstitial;
  // aerdepwetcw aerosol_wet_deposition_cloud_water;

  // evapr evaporation_of_falling_precipitation;
  // shallow+deep convective detrainment [kg/kg/s]
  // dlf = diags.total_convective_detrainment;

  constexpr int ntot_amode = AeroConfig::num_modes();
  constexpr int nlev = mam4::nlev;
  constexpr int maxd_aspectype = mam4::aero_model::maxd_aspectype;
  constexpr int zero = 0.0;

  auto work_ptr = (Real *)work.data();
  // FIXME: is an input/output wet_geometric_mean_diameter_i ?
  // DGNUMWET
  // FIXME: is an input/output dry_geometric_mean_diameter_i ?
  // ColumnView dry_geometric_mean_diameter_i[ntot_amode];
  // // DGNUM
  // for (int m = 0; m < ntot_amode; ++m) {
  //   dry_geometric_mean_diameter_i[m] = ColumnView(work_ptr, mam4::nlev);
  //   work_ptr += mam4::nlev;
  // }

  // FIXME: is an input/output qaerwat ?
  // aerosol water [kg/kg]
  // qaerwat_idx    = pbuf_get_index('QAERWAT')
  // ColumnView qaerwat[ntot_amode];
  // for (int m = 0; m < ntot_amode; ++m) {
  //   qaerwat[m] = ColumnView(work_ptr, mam4::nlev);
  //   work_ptr += mam4::nlev;
  // }
  // // FIXME: is an input/output wetdens ?
  // // wet aerosol density [kg/m3]
  // // WETDENS_AP
  // ColumnView wetdens[ntot_amode];
  // for (int m = 0; m < ntot_amode; ++m) {
  //   wetdens[m] = ColumnView(work_ptr, mam4::nlev);
  //   work_ptr += mam4::nlev;
  // }

  View2D state_q(work_ptr, mam4::nlev, pcnst);
  work_ptr += mam4::nlev * pcnst;

  View2D qqcw(work_ptr, mam4::nlev, pcnst);
  work_ptr += mam4::nlev * pcnst;

  View1D cldcu(work_ptr, mam4::nlev);
  work_ptr += mam4::nlev;

  View1D cldst(work_ptr, mam4::nlev);
  work_ptr += mam4::nlev;

  View1D evapc(work_ptr, mam4::nlev);
  work_ptr += mam4::nlev;

  View1D cmfdqr(work_ptr, mam4::nlev);
  work_ptr += mam4::nlev;

  View1D totcond(work_ptr, mam4::nlev);
  work_ptr += mam4::nlev;

  View1D conicw(work_ptr, mam4::nlev);
  work_ptr += mam4::nlev;

  // inputs
  View1D f_act_conv_coarse(work_ptr, mam4::nlev);
  work_ptr += mam4::nlev;

  View1D f_act_conv_coarse_dust(work_ptr, mam4::nlev);
  work_ptr += mam4::nlev;

  View1D f_act_conv_coarse_nacl(work_ptr, mam4::nlev);
  work_ptr += mam4::nlev;

  View1D rain(work_ptr, mam4::nlev);
  work_ptr += mam4::nlev;

  // FIXME: I need to get this variables from calcsize
  // need to connect ptend_q to tends
  View2D ptend_q(work_ptr, mam4::nlev, pcnst);
  work_ptr += mam4::nlev * pcnst;

  // CHECK; is work array ?
  View1D cldv(work_ptr, mam4::nlev);
  work_ptr += mam4::nlev;

  // CHECK; is work array ?
  View1D cldvcu(work_ptr, mam4::nlev);
  work_ptr += mam4::nlev;

  // CHECK; is work array ?
  View1D cldvst(work_ptr, mam4::nlev);
  work_ptr += mam4::nlev;

  View1D scavcoefnum(work_ptr, mam4::nlev);
  work_ptr += mam4::nlev;

  View1D scavcoefvol(work_ptr, mam4::nlev);
  work_ptr += mam4::nlev;

  View1D sol_facti(work_ptr, mam4::nlev);
  work_ptr += mam4::nlev;

  View1D sol_factic(work_ptr, mam4::nlev);
  work_ptr += mam4::nlev;

  View1D sol_factb(work_ptr, mam4::nlev);
  work_ptr += mam4::nlev;

  View1D f_act_conv(work_ptr, mam4::nlev);
  work_ptr += mam4::nlev;

  View1D scavt(work_ptr, mam4::nlev);
  work_ptr += mam4::nlev;

  View1D rcscavt(work_ptr, mam4::nlev);
  work_ptr += mam4::nlev;

  View2D rtscavt_sv(work_ptr, mam4::nlev, pcnst);
  work_ptr += pcnst * mam4::nlev;
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, mam4::nlev), [&](int i) {
    Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, pcnst),
                         [&](int j) { rtscavt_sv(i, j) = zero; });
  });

  View1D bcscavt(work_ptr, mam4::nlev);
  work_ptr += mam4::nlev;

  wetdep::zero_values(team, aerdepwetis, pcnst);
  wetdep::zero_values(team, aerdepwetcw, pcnst);

  View2D qsrflx_mzaer2cnvpr(work_ptr, aero_model::pcnst, 2);
  work_ptr += aero_model::pcnst * 2;

  /// error check
  const int workspace_used(work_ptr - work.data()),
      workspace_extent(work.extent(0));
  if (workspace_used > workspace_extent) {
    Kokkos::abort("Error aero_model_wetdep : workspace used is larger than it "
                  "is provided\n");
  }

  // inputs:
  // cldn; can we get it from atm?
  // cldt // layer cloud fraction [fraction] from pbuf_get_field
  // FIXME:
  constexpr int nwetdep = 1; // number of elements in wetdep_list

  // inputs
  // Compute variables needed for convproc unified convective transport
  // rprdsh // pbuf_get_field rain production, shallow convection [kg/kg/s]
  // rprddp // pbuf_get_field rain production, deep convection [kg/kg/s]
  // evapcdp // pbuf_get_field Evaporation rate of shallow convective
  // precipitation >=0. [kg/kg/s] evapcsh // pbuf_get_field Evaporation rate of
  // deep    convective precipitation >=0. [kg/kg/s]

  // icwmrdp in cloud water mixing ratio, deep convection [kg/kg]
  // icwmrsh in cloud water mixing ratio, shallow convection [kg/kg]

  // inputs
  // dp_frac Deep convective cloud fraction [fraction]
  // sh_frac Shallow convective cloud fraction [fraction]
  // dp_ccf
  // sh_ccf

  // ouputs:
  // rprdshsum
  // rprddpsum
  // evapcdpsum
  // evapcshsum

  //
  haero::ConstColumnView temperature = atm.temperature;
  haero::ConstColumnView pmid = atm.pressure;
  haero::ConstColumnView pdel = atm.hydrostatic_dp; // layer thickness (Pa)
  haero::ConstColumnView q_liq = atm.liquid_mixing_ratio;
  haero::ConstColumnView q_ice = atm.ice_mixing_ratio;

  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, nlev), [&](int kk) {
    // copy data from prog to stateq
    const auto state_q_kk = ekat::subview(state_q, kk);
    const auto qqcw_kk = ekat::subview(qqcw, kk);
    const auto ptend_q_kk = ekat::subview(ptend_q, kk);
    utils::extract_stateq_from_prognostics(progs, atm, state_q_kk.data(), kk);
    utils::extract_qqcw_from_prognostics(progs, qqcw_kk.data(), kk);
    utils::extract_ptend_from_tendencies(tends, ptend_q_kk.data(), kk);
  });
  team.team_barrier();

  //
  // Do calculations of mode radius and water uptake if:
  // 1) modal aerosols are affecting the climate, or
  // 2) prognostic modal aerosols are enabled
  // If not using prognostic aerosol call the diagnostic version

  // Calculate aerosol size distribution parameters
  // for prognostic modal aerosols the transfer of mass between aitken and
  // accumulation modes is done in conjunction with the dry radius calculation
  // compute calcsize and

  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, 0, nlev), [&](int kk) {
    const auto state_q_kk = ekat::subview(state_q, kk);
    const auto qqcw_kk = ekat::subview(qqcw, kk);
    const auto ptend_q_kk = ekat::subview(ptend_q, kk);
    Real dgnumwet_m_kk[ntot_amode] = {};
    // wetdens and qaerwat are input/ouput to water_uptake
    Real qaerwat_m_kk[ntot_amode] = {};
    Real wetdens_kk[ntot_amode] = {};
    // dgncur_a is aerosol particle diameter and is an input to
    // calcsize. But calcsize reset its value.
    Real dgnumdry_m_kk[ntot_amode] = {};
    for (int imode = 0; imode < ntot_amode; imode++) {
      dgnumwet_m_kk[imode] = wet_geometric_mean_diameter_i(imode, kk);
      dgnumdry_m_kk[imode] = dry_geometric_mean_diameter_i(imode, kk);
      qaerwat_m_kk[imode] = qaerwat(imode, kk);
      wetdens_kk[imode] = wetdens(imode, kk);
    }

    {

      const bool do_adjust = true;
      const bool do_aitacc_transfer = true;
      const bool update_mmr = true;

      int numptr_amode[ntot_amode];
      int mam_idx[ntot_amode][ndrop::nspec_max];
      int mam_cnst_idx[ntot_amode][ndrop::nspec_max];
      int nspec_amode[ntot_amode];
      int lspectype_amode[maxd_aspectype][ntot_amode];
      int lmassptr_amode[maxd_aspectype][ntot_amode];
      Real specdens_amode[maxd_aspectype];
      Real spechygro[maxd_aspectype];

      // Get MAM4 parameters (all arguments in the following call are output
      // variables)
      ndrop::get_e3sm_parameters(
          nspec_amode, lspectype_amode, lmassptr_amode, // output
          numptr_amode, specdens_amode, spechygro,      // output
          mam_idx, mam_cnst_idx);                       // output

      Real inv_density[ntot_amode][AeroConfig::num_aerosol_ids()] = {};
      Real num2vol_ratio_min[ntot_amode] = {};
      Real num2vol_ratio_max[ntot_amode] = {};
      Real num2vol_ratio_max_nmodes[ntot_amode] = {};
      Real num2vol_ratio_min_nmodes[ntot_amode] = {};
      Real num2vol_ratio_nom_nmodes[ntot_amode] = {};
      Real dgnmin_nmodes[ntot_amode] = {};
      Real dgnmax_nmodes[ntot_amode] = {};
      Real dgnnom_nmodes[ntot_amode] = {};
      //
      Real mean_std_dev_nmodes[ntot_amode] = {};

      // outputs
      bool noxf_acc2ait[AeroConfig::num_aerosol_ids()] = {};
      int n_common_species_ait_accum = {};
      int ait_spec_in_acc[AeroConfig::num_aerosol_ids()] = {};
      int acc_spec_in_ait[AeroConfig::num_aerosol_ids()] = {};

      // Initialize all contant variables needed for calcsize
      //(all arguments in the following call are output variables)
      modal_aero_calcsize::init_calcsize(
          inv_density, num2vol_ratio_min, num2vol_ratio_max,
          num2vol_ratio_max_nmodes, num2vol_ratio_min_nmodes,
          num2vol_ratio_nom_nmodes, dgnmin_nmodes, dgnmax_nmodes, dgnnom_nmodes,
          mean_std_dev_nmodes, noxf_acc2ait, n_common_species_ait_accum,
          ait_spec_in_acc, acc_spec_in_ait);

      Real dgncur_c_kk[ntot_amode] = {};
      Real dqqcwdt_kk[pcnst] = {};
      //  Calculate aerosol size distribution parameters and aerosol water
      //  uptake for prognostic aerosols
      modal_aero_calcsize::modal_aero_calcsize_sub(
          // Inputs
          state_q_kk.data(), qqcw_kk.data(), dt, do_adjust, do_aitacc_transfer,
          update_mmr, lmassptr_amode, numptr_amode, inv_density,
          num2vol_ratio_min, num2vol_ratio_max, num2vol_ratio_max_nmodes,
          num2vol_ratio_min_nmodes, num2vol_ratio_nom_nmodes, dgnmin_nmodes,
          dgnmax_nmodes, dgnnom_nmodes, mean_std_dev_nmodes, noxf_acc2ait,
          n_common_species_ait_accum, ait_spec_in_acc, acc_spec_in_ait,
          // Outputs
          dgnumdry_m_kk, dgncur_c_kk, ptend_q_kk.data(), dqqcwdt_kk);
      // NOTE: dgnumdry_m_kk is interstitial dry diameter size and
      // dgncur_c_kk is cloud borne dry diameter size

      // update could aerosol.
      if (update_mmr) {
        // Note: it only needs to update aerosol variables.
        for (int i = utils::aero_start_ind(); i < pcnst; ++i) {
          qqcw(kk, i) = haero::max(zero, qqcw(kk, i) + dqqcwdt_kk[i] * dt);
        }
      } // end update could aerosols.

      mam4::water_uptake::modal_aero_water_uptake_dr(
          // inputs
          nspec_amode, specdens_amode, spechygro, lspectype_amode,
          state_q_kk.data(), temperature(kk), pmid(kk), cldt(kk), dgnumdry_m_kk,
          // outputs
          dgnumwet_m_kk, qaerwat_m_kk, wetdens_kk);
    }

    // team.team_barrier();

    // save diameters, we use them in wet_dep.
    for (int imode = 0; imode < ntot_amode; imode++) {
      wet_geometric_mean_diameter_i(imode, kk) = dgnumwet_m_kk[imode];
      dry_geometric_mean_diameter_i(imode, kk) = dgnumdry_m_kk[imode];
      qaerwat(imode, kk) = qaerwat_m_kk[imode];
      wetdens(imode, kk) = wetdens_kk[imode];
    }
  }); // klev parallel_for loop

  team.team_barrier();

  // skip wet deposition if nwetdep is non-positive
  if (nwetdep < 1)
    return;
  {

    // // change mode order as mmode_loop_aa loops in a different order
    const int mode_order_change[4] = {0, 1, 3, 2};

    const int jaerowater = 2;

    // cumulus cloud fraction =  dp_frac + sh_frac
    wetdep::sum_values(team,                    // input
                       cldcu,                   // output
                       dp_frac, sh_frac, nlev); // inputs

    // total cloud fraction [fraction] = dp_ccf + sh_ccf
    // Stratiform cloud fraction cldst  = cldt - cldcu  Stratiform cloud
    // fraction
    team.team_barrier();
    Kokkos::parallel_for(Kokkos::TeamVectorRange(team, nlev),
                         [&](int k) { cldst[k] = cldt[k] - cldcu[k]; });

    // FIXME: where does eq come from?
    // FIXME: in fortran code cldt is equal to cln
    // wetdep::sum_values(team, cldt, dp_ccf, sh_ccf, nlev);
    // evaporation from convection (deep + shallow)
    wetdep::sum_values(team,                    // input
                       evapc,                   // output
                       evapcsh, evapcdp, nlev); // inputs

    // dq/dt due to convective cmfdqr =  rprddp + rprdsh
    wetdep::sum_values(team,                  // input
                       cmfdqr,                // output
                       rprddp, rprdsh, nlev); // inputs
    // rate of conversion of condensate to precipitation [kg/kg/s].
    // wetdep::sum_values(team, prain, icwmrdp, icwmrsh, nlev);
    // wetdep::sum_values(team, prain, icwmrdp, icwmrsh, nlev);
    // total condensate (ice+liq) [kg/kg] = q_liq + q_ice
    wetdep::sum_values(team,                // input
                       totcond,             // output
                       q_liq, q_ice, nlev); // input
    // sum deep and shallow convection contributions
    wetdep::sum_deep_and_shallow(team,                               // input
                                 conicw,                             // output
                                 icwmrdp, dp_frac, icwmrsh, sh_frac, // inputs
                                 nlev);                              // inputs

    // Estimate the cloudy volume which is occupied by rain or cloud water
    wetdep::cloud_diagnostics(team, temperature, pmid, pdel, cmfdqr, evapc,
                              cldt, cldcu, cldst, evapr, prain,
                              // outputs
                              cldv, cldvcu, cldvst, rain,
                              // inputs
                              nlev);

    team.team_barrier(); // for cldcu

    Int1D isprx;
    // calculate the mass-weighted sol_factic for coarse mode species
    // set the mass-weighted sol_factic for coarse mode species.
    wetdep::set_f_act(
        // input
        team,
        // outputs
        isprx, f_act_conv_coarse, f_act_conv_coarse_dust,
        f_act_conv_coarse_nacl,
        // inputs
        pdel, prain, cmfdqr, evapr, state_q, ptend_q, dt, nlev);
    team.team_barrier();

    // main loop over aerosol modes
    for (int mtmp = 0; mtmp < AeroConfig::num_modes(); ++mtmp) {
      // for mam4, do accum, aitken, pcarbon, then coarse
      // so change the order of 2 and 3 here
      // for mam4:
      // do   accum = 0,
      // then aitken = 1,
      // then pcarbon - 3,
      // then coarse = 2
      const int imode = mode_order_change[mtmp];

      // loop over interstitial (1) and cloud-borne (2) forms
      // BSINGH (09/12/2014):Do cloudborne first for unified convection
      // scheme so that the resuspension of cloudborne can be saved then
      // applied to interstitial (RCE)

      // do cloudborne (2) first then interstitial (1)
      for (int lphase = 2; 1 <= lphase; --lphase) {
        if (lphase == 1) { // interstial aerosol
          // Computes lookup table for aerosol impaction/interception scavenging
          // rates
          wetdep::modal_aero_bcscavcoef_get(
              // inputs
              team, wet_geometric_mean_diameter_i, isprx, scavimptblvol,
              scavimptblnum,
              // outputs
              scavcoefnum, scavcoefvol,
              // inputs
              imode, nlev);
        }
        // define sol_factb and sol_facti values, and f_act_conv
        wetdep::define_act_frac(
            // input
            team,
            // outputs
            sol_facti, sol_factic, sol_factb, f_act_conv,
            // inputs
            lphase, imode, nlev);
        team.team_barrier();

        // REASTER 08/12/2015 - changed ordering (mass then number) for
        // prevap resuspend to coarse loop over number + chem constituents +
        // water index for aerosol number / chem-mass / water-mass

        for (int lspec = 0; lspec < num_species_mode(imode) + 2; ++lspec) {
          int mm, jnv, jnummaswtr;
          aero_model::index_ordering(lspec, imode, lphase, mm, jnv, jnummaswtr);
          // bypass wet aerosols
          if (0 <= mm && jnummaswtr != jaerowater) {
            // The following call mimics wetdepa_v2 subroutine call in
            // aero_model.F90
            wetdep::compute_q_tendencies( // tendencies are in scavt
                team, f_act_conv, f_act_conv_coarse, f_act_conv_coarse_dust,
                f_act_conv_coarse_nacl, scavcoefnum, scavcoefvol, totcond,
                cmfdqr, conicw, evapc, evapr, prain, dlf, cldt, cldcu, cldst,
                cldvst, cldvcu, sol_facti, sol_factic, sol_factb,
                // outputs
                scavt, bcscavt, rcscavt, rtscavt_sv,
                // inputs
                state_q, qqcw,
                // outputs
                ptend_q,
                // inputs
                pdel, dt, jnummaswtr, jnv, mm, lphase, imode, lspec);
            team.team_barrier();

            // Note: update tendencies only in lphase == 1
            if (lphase == 1) {
              // Update ptend_q from the tendency, scavt
              wetdep::update_q_tendencies(team,             // input
                                          ptend_q,          // output
                                          scavt, mm, nlev); // inputs
            }
            if (lphase == 1) {
              aerdepwetis[mm] = aero_model::calc_sfc_flux(team,        // input
                                                          scavt,       // output
                                                          pdel, nlev); // inputs
            } else // if (lphase == 2)
            {
              aerdepwetcw[mm] = aero_model::calc_sfc_flux(team,        // input
                                                          scavt,       // output
                                                          pdel, nlev); // inputs
              team.team_barrier();
              Kokkos::parallel_for(
                  Kokkos::TeamVectorRange(team, nlev),
                  [&](int kk) { qqcw(kk, mm) += scavt(kk) * dt; });
            }
#if 0
            // Note: Commenting it out because it produces unused variable warnings.
            Real rprdshsum = aero_model::calc_sfc_flux(team, rprdsh, pdel, nlev);
            Real rprddpsum = aero_model::calc_sfc_flux(team, rprddp, pdel, nlev);
            Real evapcdpsum = aero_model::calc_sfc_flux(team, evapcdp, pdel, nlev);
            Real evapcshsum = aero_model::calc_sfc_flux(team, evapcsh, pdel, nlev);

            // NOTE. Adding this team_barrier fixed one race condition.
            team.team_barrier();
            const Real sflxbc =
                aero_model::calc_sfc_flux(team, bcscavt, pdel, nlev);
            const Real sflxec =
                aero_model::calc_sfc_flux(team, rcscavt, pdel, nlev);

            // apportion convective surface fluxes to deep and shallow
            // conv this could be done more accurately in subr wetdepa
            // since deep and shallow rarely occur simultaneously, and
            // these fields are just diagnostics, this approximate method
            // is adequate only do this for interstitial aerosol, because
            // conv clouds to not affect the stratiform-cloudborne
            // aerosol.
            // NOTE. Adding this team_barrier fixed one race condition.
            team.team_barrier();

            // FIXME: The following code is causing race condition errors in the
            // computer-sanitizer.
            //  I commented it out because we do not need it in the emaxx-mam4xx
            //  interface.
            {
              Real sflxbcdp, sflxecdp;
              aero_model::apportion_sfc_flux_deep(rprddpsum, rprdshsum,
                                                evapcdpsum, evapcshsum, sflxbc,
                                                sflxec, sflxbcdp, sflxecdp);

              // when ma_convproc_intr is used, convective in-cloud wet
              // removal is done there the convective (total and deep)
              // precip-evap-resuspension includes in- and below-cloud
              // contributions, so pass the below-cloud contribution to
              // ma_convproc_intr
              //
              // NOTE: ma_convproc_intr no longer uses these
              qsrflx_mzaer2cnvpr(mm, 0) = sflxec;
              qsrflx_mzaer2cnvpr(mm, 1) = sflxecdp;
            }
#endif
          }
        }
      }
    }
  }
  // make sure that ptend is updated in tendencies
  team.team_barrier();
  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, 0, nlev), [&](int kk) {
    const auto ptend_q_kk = ekat::subview(ptend_q, kk);
    const auto state_q_kk = ekat::subview(state_q, kk);
    const auto qqcw_kk = ekat::subview(qqcw, kk);
    utils::inject_qqcw_to_prognostics(qqcw_kk.data(), progs, kk);
    utils::inject_stateq_to_prognostics(state_q_kk.data(), progs, kk);
    utils::inject_ptend_to_tendencies(ptend_q_kk.data(), tends, kk);
  });
  team.team_barrier();

} // aero_model_wetdep

} // namespace wetdep

/// @class WedDeposition
/// Wet Deposition process for MAM4 aerosol model.
class WetDeposition {
public:
  struct Config {
    Config(){};
    Config(const Config &) = default;
    ~Config() = default;
    Config &operator=(const Config &) = default;
    int nlev = mam4::nlev;

    // mam_prevap_resusp_optcc values control the prevap_resusp calculations in
    // wetdepa_v2:
    //     0 = no resuspension
    //   130 = non-linear resuspension of aerosol mass   based on scavenged
    //   aerosol mass 230 = non-linear resuspension of aerosol number based on
    //   raindrop number the 130 thru 230 all use the new prevap_resusp code
    //   block in subr wetdepa_v2
    int mam_prevap_resusp_optcc = 0;
  };

  const char *name() const { return "MAM4 Wet Deposition"; }

  inline void init(const AeroConfig &aero_config,
                   const Config &wed_dep_config = Config());

  // compute_tendencies -- computes tendencies and updates diagnostics
  // NOTE: that both diags and tends are const below--this means their views
  // NOTE: are fixed, but the data in those views is allowed to vary.
  KOKKOS_INLINE_FUNCTION
  void compute_tendencies(const AeroConfig &config, const ThreadTeam &team,
                          Real t, Real dt, const Atmosphere &atm,
                          const Surface &sfc, const Prognostics &progs,
                          const Diagnostics &diags,
                          const Tendencies &tends) const;

  Kokkos::View<Real *[2]> qsrflx_mzaer2cnvpr;

private:
  Config config_;

  Kokkos::View<int *> isprx;

  Kokkos::View<Real *> cldv;
  Kokkos::View<Real *> cldvcu;
  Kokkos::View<Real *> cldvst;
  Kokkos::View<Real *> rain;
  Kokkos::View<Real *> cldcu;
  Kokkos::View<Real *> cldt;
  Kokkos::View<Real *> evapc;
  Kokkos::View<Real *> cmfdqr;
  Kokkos::View<Real *> prain;
  Kokkos::View<Real *> conicw;
  Kokkos::View<Real *> totcond;

  Kokkos::View<Real *> f_act_conv_coarse;
  Kokkos::View<Real *> f_act_conv_coarse_dust;
  Kokkos::View<Real *> f_act_conv_coarse_nacl;

  Kokkos::View<Real *> scavcoefnum;
  Kokkos::View<Real *> scavcoefvol;

  Kokkos::View<Real *> sol_facti;
  Kokkos::View<Real *> sol_factic;
  Kokkos::View<Real *> sol_factb;
  Kokkos::View<Real *> f_act_conv;

  Kokkos::View<Real *> scavt;   // scavenging tend [kg/kg/s]
  Kokkos::View<Real *> bcscavt; // below cloud, convective [kg/kg/s]
  Kokkos::View<Real *> rcscavt; // resuspension, convective [kg/kg/s]

  Kokkos::View<Real *> rtscavt_sv;

  Kokkos::View<Real * [aero_model::maxd_aspectype + 2][aero_model::pcnst]>
      qqcw_sav;

  Real scavimptblnum[aero_model::nimptblgrow_total][AeroConfig::num_modes()];
  Real scavimptblvol[aero_model::nimptblgrow_total][AeroConfig::num_modes()];
};

void WetDeposition::init(const AeroConfig &aero_config,
                         const Config &wed_dep_config) {
  config_ = wed_dep_config;
  const int nlev = config_.nlev;
  Kokkos::resize(isprx, nlev);
  Kokkos::resize(cldv, nlev);
  Kokkos::resize(cldvcu, nlev);
  Kokkos::resize(cldvst, nlev);
  Kokkos::resize(rain, nlev);
  Kokkos::resize(cldcu, nlev);
  Kokkos::resize(cldt, nlev);
  Kokkos::resize(evapc, nlev);
  Kokkos::resize(cmfdqr, nlev);
  Kokkos::resize(prain, nlev);
  Kokkos::resize(conicw, nlev);
  Kokkos::resize(totcond, nlev);
  Kokkos::resize(f_act_conv_coarse, nlev);
  Kokkos::resize(f_act_conv_coarse_dust, nlev);
  Kokkos::resize(f_act_conv_coarse_nacl, nlev);
  Kokkos::resize(scavcoefnum, nlev);
  Kokkos::resize(scavcoefvol, nlev);
  Kokkos::resize(sol_facti, nlev);
  Kokkos::resize(sol_factic, nlev);
  Kokkos::resize(sol_factb, nlev);
  Kokkos::resize(f_act_conv, nlev);
  Kokkos::resize(qsrflx_mzaer2cnvpr, aero_model::pcnst, 2);
  Kokkos::resize(qqcw_sav, nlev, aero_model::maxd_aspectype + 2,
                 aero_model::pcnst);

  Kokkos::resize(scavt, nlev);
  Kokkos::resize(bcscavt, nlev);
  Kokkos::resize(rcscavt, nlev);

  Kokkos::resize(rtscavt_sv, aero_model::pcnst);
  Kokkos::deep_copy(rtscavt_sv, 0.0);

  const int num_modes = AeroConfig::num_modes();
  Real dgnum_amode[num_modes];
  Real sigmag_amode[num_modes];
  Real aerosol_dry_density[num_modes];
  for (int i = 0; i < num_modes; ++i) {
    dgnum_amode[i] = modes(i).nom_diameter;
    sigmag_amode[i] = modes(i).mean_std_dev;
  }
  // Note: Original code uses the following aerosol densities.
  // sulfate, sulfate, dust, p-organic
  aerosol_dry_density[0] = mam4::mam4_density_so4;
  aerosol_dry_density[1] = mam4::mam4_density_so4;
  aerosol_dry_density[2] = mam4::mam4_density_dst;
  aerosol_dry_density[3] = mam4::mam4_density_pom;
  aero_model::modal_aero_bcscavcoef_init(dgnum_amode, sigmag_amode,
                                         aerosol_dry_density, scavimptblnum,
                                         scavimptblvol);
}

// compute_tendencies -- computes tendencies and updates diagnostics
// NOTE: that both diags and tends are const below--this means their views
// NOTE: are fixed, but the data in those views is allowed to vary.
KOKKOS_INLINE_FUNCTION
void WetDeposition::compute_tendencies(
    const AeroConfig &config, const ThreadTeam &team, Real t, Real dt,
    const Atmosphere &atm, const Surface &sfc, const Prognostics &progs,
    const Diagnostics &diags, const Tendencies &tends) const {
  const int nlev = config_.nlev;
  static constexpr int pcnst = aero_model::pcnst;

  // change mode order as mmode_loop_aa loops in a different order
  static constexpr int mode_order_change[4] = {0, 1, 3, 2};

  const int jaerowater = 2;

  haero::ConstColumnView temperature = atm.temperature;
  haero::ConstColumnView pmid = atm.pressure;
  haero::ConstColumnView pdel = atm.hydrostatic_dp;
  haero::ConstColumnView q_liq = atm.liquid_mixing_ratio;
  haero::ConstColumnView q_ice = atm.ice_mixing_ratio;

  // calculate some variables needed in wetdepa_v2
  ColumnView dp_frac = diags.deep_convective_cloud_fraction;
  ColumnView sh_frac = diags.shallow_convective_cloud_fraction;
  ColumnView dp_ccf = diags.deep_convective_cloud_fraction;
  ColumnView sh_ccf = diags.shallow_convective_cloud_fraction;
  ColumnView evapcdp = diags.deep_convective_precipitation_evaporation;
  ColumnView evapcsh = diags.shallow_convective_precipitation_evaporation;
  ColumnView cldst = diags.stratiform_cloud_fraction;
  ColumnView evapr = diags.evaporation_of_falling_precipitation;
  ColumnView rprddp = diags.deep_convective_precipitation_production;
  ColumnView rprdsh = diags.shallow_convective_precipitation_production;
  ColumnView icwmrdp = diags.deep_convective_cloud_condensate;
  ColumnView icwmrsh = diags.shallow_convective_cloud_condensate;

  // FIXME: move values from prog to state_q
  Diagnostics::ColumnTracerView state_q = diags.tracer_mixing_ratio;
  Diagnostics::ColumnTracerView ptend_q = diags.d_tracer_mixing_ratio_dt;

  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, nlev), [&](int kk) {
    // copy data from prog to stateq
    const auto state_q_kk = ekat::subview(state_q, kk);
    utils::extract_stateq_from_prognostics(progs, atm, state_q_kk.data(), kk);
  });
  team.team_barrier();

  ColumnView aerdepwetis = diags.aerosol_wet_deposition_interstitial;
  ColumnView aerdepwetcw = diags.aerosol_wet_deposition_cloud_water;

  Real rprdshsum = 0, rprddpsum = 0, evapcdpsum = 0, evapcshsum = 0;
  Kokkos::parallel_reduce(
      Kokkos::TeamVectorRange(team, nlev),
      [&](int kk, Real &suma) { suma += rprdsh[kk]; }, rprdshsum);
  Kokkos::parallel_reduce(
      Kokkos::TeamVectorRange(team, nlev),
      [&](int kk, Real &suma) { suma += rprddp[kk]; }, rprddpsum);
  Kokkos::parallel_reduce(
      Kokkos::TeamVectorRange(team, nlev),
      [&](int kk, Real &suma) { suma += evapcdp[kk]; }, evapcdpsum);
  Kokkos::parallel_reduce(
      Kokkos::TeamVectorRange(team, nlev),
      [&](int kk, Real &suma) { suma += evapcsh[kk]; }, evapcshsum);

  // cumulus cloud fraction =  dp_frac + sh_frac
  wetdep::sum_values(team, cldcu, dp_frac, sh_frac, nlev);
  // total cloud fraction [fraction] = dp_ccf + sh_ccf
  wetdep::sum_values(team, cldt, dp_ccf, sh_ccf, nlev);
  // evaporation from convection (deep + shallow)
  wetdep::sum_values(team, evapc, evapcsh, evapcdp, nlev);
  // dq/dt due to convective cmfdqr =  rprddp + rprdsh
  wetdep::sum_values(team, cmfdqr, rprddp, rprdsh, nlev);
  // rate of conversion of condensate to precipitation [kg/kg/s].
  wetdep::sum_values(team, prain, icwmrdp, icwmrsh, nlev);
  // total condensate (ice+liq) [kg/kg] = q_liq + q_ice
  wetdep::sum_values(team, totcond, q_liq, q_ice, nlev);
  // sum deep and shallow convection contributions
  wetdep::sum_deep_and_shallow(team, conicw, icwmrdp, dp_frac, icwmrsh, sh_frac,
                               nlev);

  team.team_barrier(); // for cldcu

  // Estimate the cloudy volume which is occupied by rain or cloud water
  wetdep::cloud_diagnostics(team, temperature, pmid, pdel, cmfdqr, evapc, cldt,
                            cldcu, cldst, evapr, prain, cldv, cldvcu, cldvst,
                            rain, nlev);
  team.team_barrier();
  ColumnView dlf = diags.total_convective_detrainment;

  wetdep::zero_values(team, aerdepwetis, pcnst);
  wetdep::zero_values(team, aerdepwetcw, pcnst);

  // set the mass-weighted sol_factic for coarse mode species.
  wetdep::set_f_act(team, isprx, f_act_conv_coarse, f_act_conv_coarse_dust,
                    f_act_conv_coarse_nacl, pdel, prain, cmfdqr, evapr, state_q,
                    ptend_q, dt, nlev);
  team.team_barrier();

  // main loop over aerosol modes
  for (int mtmp = 0; mtmp < AeroConfig::num_modes(); ++mtmp) {
    // for mam4, do accum, aitken, pcarbon, then coarse
    // so change the order of 2 and 3 here
    // for mam4:
    // do   accum = 0,
    // then aitken = 1,
    // then pcarbon - 3,
    // then coarse = 2
    const int imode = mode_order_change[mtmp];

    // loop over interstitial (1) and cloud-borne (2) forms
    // BSINGH (09/12/2014):Do cloudborne first for unified convection
    // scheme so that the resuspension of cloudborne can be saved then
    // applied to interstitial (RCE)

    // do cloudborne (2) first then interstitial (1)
    for (int lphase = 2; 1 <= lphase; --lphase) {

      if (lphase == 1) { // interstial aerosol
        // Computes lookup table for aerosol impaction/interception scavenging
        // rates
        wetdep::modal_aero_bcscavcoef_get(team, diags, isprx, scavimptblvol,
                                          scavimptblnum, scavcoefnum,
                                          scavcoefvol, imode, nlev);
      }
      // define sol_factb and sol_facti values, and f_act_conv
      wetdep::define_act_frac(team, sol_facti, sol_factic, sol_factb,
                              f_act_conv, lphase, imode, nlev);
      team.team_barrier();

      // REASTER 08/12/2015 - changed ordering (mass then number) for
      // prevap resuspend to coarse loop over number + chem constituents +
      // water index for aerosol number / chem-mass / water-mass
      for (int lspec = 0; lspec < num_species_mode(imode) + 2; ++lspec) {

        int mm, jnv, jnummaswtr;
        aero_model::index_ordering(lspec, imode, lphase, mm, jnv, jnummaswtr);
        // bypass wet aerosols
        if (0 <= mm && jnummaswtr != jaerowater) {

#if 0
          wetdep::compute_q_tendencies( // tendencies are in scavt
              team, f_act_conv, f_act_conv_coarse, f_act_conv_coarse_dust,
              f_act_conv_coarse_nacl, scavcoefnum, scavcoefvol, totcond, cmfdqr,
              conicw, evapc, evapr, prain, dlf, cldt, cldcu, cldst, cldvst,
              cldvcu, sol_facti, sol_factic, sol_factb, scavt, bcscavt, rcscavt,
              rtscavt_sv, state_q, ptend_q, qqcw_sav, pdel, dt, jnummaswtr, jnv,
              mm, lphase, imode, lspec);
          team.team_barrier();

#endif
          // Update ptend_q from the tendency, scavt
          wetdep::update_q_tendencies(team, ptend_q, scavt, mm, nlev);

          if (lphase == 1)
            aerdepwetis[mm] =
                aero_model::calc_sfc_flux(team, scavt, pdel, nlev);
          else // if (lphase == 2)
            aerdepwetcw[mm] =
                aero_model::calc_sfc_flux(team, scavt, pdel, nlev);
          const Real sflxbc =
              aero_model::calc_sfc_flux(team, bcscavt, pdel, nlev);
          const Real sflxec =
              aero_model::calc_sfc_flux(team, rcscavt, pdel, nlev);

          // apportion convective surface fluxes to deep and shallow
          // conv this could be done more accurately in subr wetdepa
          // since deep and shallow rarely occur simultaneously, and
          // these fields are just diagnostics, this approximate method
          // is adequate only do this for interstitial aerosol, because
          // conv clouds to not affect the stratiform-cloudborne
          // aerosol.
          Real sflxbcdp, sflxecdp;
          aero_model::apportion_sfc_flux_deep(rprddpsum, rprdshsum, evapcdpsum,
                                              evapcshsum, sflxbc, sflxec,
                                              sflxbcdp, sflxecdp);

          // when ma_convproc_intr is used, convective in-cloud wet
          // removal is done there the convective (total and deep)
          // precip-evap-resuspension includes in- and below-cloud
          // contributions, so pass the below-cloud contribution to
          // ma_convproc_intr
          //
          // NOTE: ma_convproc_intr no longer uses these
          qsrflx_mzaer2cnvpr(mm, 0) = sflxec;
          qsrflx_mzaer2cnvpr(mm, 1) = sflxecdp;
        }
      }
      team.team_barrier();
    }
  }
  team.team_barrier();
}
} // namespace mam4

#endif
