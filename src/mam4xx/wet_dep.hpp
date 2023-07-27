// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MAM4XX_WET_DEPOSITION_HPP
#define MAM4XX_WET_DEPOSITION_HPP

#include <haero/atmosphere.hpp>
#include <haero/constants.hpp>
#include <haero/math.hpp>
#include <mam4xx/aero_config.hpp>
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
KOKKOS_INLINE_FUNCTION
void local_precip_production(/* cont int ncol, */ const Real *pdel,
                             const Real *source_term, const Real *sink_term,
                             Real *lprec, const Atmosphere &atm) {
  const int pver = atm.num_levels();
  for (int i = 0; i < pver; i++) {
    lprec[i] = (pdel[i] / Constants::gravity) * (source_term[i] - sink_term[i]);
  }
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
 * @param[out] sumppr_all Sum of precipitation rate above each layer, for calling rain_mix_ratio use [kg/m2/s]
 *
 * @pre cld, lprec, cldv and sumppr_all are all an array
 *      of size pver == atm.num_levels().
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
KOKKOS_INLINE_FUNCTION
void calculate_cloudy_volume(/* cont int ncol, */ const Real *cld, const Real *lprec,
                             const bool is_tot_cld, Real *cldv, Real *sumppr_all,
                             const Atmosphere &atm) {
  const int pver = atm.num_levels();
  Real sumppr = 0.0; // Precipitation rate [kg/m2/s]
  Real cldv1 = 0.0; // Precip weighted cloud fraction from above [kg/m2/s]
  Real sumpppr = 1e-36; // Sum of positive precips from above

  Real lprecp = 0.0; // Local production rate of precip [kg/m2/s] if positive

  for (int i = 0; i < pver; i++) {
      if (is_tot_cld) {
          cldv[i] = haero::max( haero::min(1.0, cldv1 / sumpppr) * sumppr / sumpppr, cld[i]);
      }
      else {
          // For convective and stratiform precipitation volume at the top 
          // interface of each layer. Neglect the current layer.
          cldv[i] = haero::max( haero::min(1.0, cldv1 / sumpppr) * (sumppr / sumpppr), 0.0);
      }
      lprecp = haero::max(lprec[i], 1e-30);
      cldv1 += cld[i] * lprecp;
      sumppr += lprec[i];
      sumppr_all[i] = sumppr;
      sumpppr += lprecp;
  }
}

/**
 * @brief Calculate rain mixing ratio [kg/kg] from the local precipitation rate
 * 
 * @param[in] temperature Temperature [K].
 * @param[in] pmid Pressure at layer midpoints [Pa].
 * @param[in] sumppr Sum of precipitation rate above each layer [kg/m2/s]
 * @param[in] atm Atmosphere object (used for number of levels).
 * 
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
void rain_mix_ratio(/* cont int ncol, */ const Real *temperature, const Real *pmid,
                             const Real *sumppr, Real *rain, const Atmosphere &atm) {
  const int pver = atm.num_levels();
  Real convfw = 0.0; // Falling velocity at air density = 1 kg/m3 [m/s * sqrt(rho)].
  Real rho = 0.0; // Air density [kg/m3].
  Real vfall = 0.0; // Falling velocity [m/s].

  // Define constant convfw from cldwat.F90
  // Reference: Tripoli and Cotton (1980)
  convfw = 1.94 * 2.13 * haero::sqrt(Constants::density_h2o * Constants::gravity * 2.7e-4);

  for (int i = 0; i < pver; i++) {
      rain[i] = 0.0;
      if( temperature[i] > Constants::melting_pt_h2o ) {
          rho = pmid[i] / (Constants::r_gas_dry_air * temperature[i]);
          vfall = convfw / haero::sqrt(rho);
          rain[i] = sumppr[i] / (rho * vfall);
          if (rain[i] < 1e-14) {
              rain[i] = 0.0;
          }
      }
  }
}

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
 * @param[in] evapc Evaporation rate of convective precipitation ( >= 0 ) [kg/kg/s].
 * @param[in] cldt Total cloud fraction [fraction, unitless].
 * @param[in] cldcu Cumulus cloud fraction [fraction, unitless].
 * @param[in] clst Stratus cloud fraction [fraction, unitless].
 * @param[in] evapr rate of evaporation of falling precipitation [kg/kg/s].
 * @param[in] prain rate of conversion of condensate to precipitation [kg/kg/s].
 * @param[in] atm Atmosphere object (used for number of levels).
 * 
 * @param[out] cldv Fraction occupied by rain or cloud water [fraction, unitless].
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
void clddiag(const Real* temperature, const Real* pmid, const Real* pdel,
             const Real* cmfdqr, const Real* evapc, const Real* cldt,
             const Real* cldcu, const Real* cldst, const Real* evapr,
             const Real* prain, Real* cldv, Real* cldvcu, Real* cldvst,
             Real* rain, const Atmosphere &atm)
{
  // Calculate local precipitation production rate
  // In src/chemistry/aerosol/wetdep.F90, (prain + cmfdqr) is used for source_term
  // This is just a temporary array that contains the sum of the two vectors...
  const int pver = atm.num_levels();

  // Have to use stack memory since pver is non-const
  auto source_term = new Real[pver];
  auto lprec = new Real[pver];
  auto lprec_st = new Real[pver];
  auto lprec_cu = new Real[pver];
  auto sumppr_all = new Real[pver];
  auto sumppr_all_cu = new Real[pver];
  auto sumppr_all_st = new Real[pver];
 
  for (int i = 0; i < pver; i++) {
    source_term[i] = prain[i] + cmfdqr[i];
  }

  // ...and then we pass the temporary array to local_precip_production
  // TODO - !FIXME: Possible bug: why there is no evapc in lprec calculation?
  local_precip_production(/* ncol, */ pdel, source_term, evapc, lprec, atm);
  local_precip_production(/* ncol, */ pdel, cmfdqr, evapr, lprec_cu, atm);
  local_precip_production(/* ncol, */ pdel, prain, evapr, lprec_st, atm);

  // Calculate cloudy volume which is occupied by rain or cloud water
  // Total
  calculate_cloudy_volume(/* ncol, */ cldt, lprec, true, cldv, sumppr_all, atm);
  // Convective
  calculate_cloudy_volume(/* ncol, */ cldcu, lprec_cu, false, cldvcu, sumppr_all_cu, atm);
  // Stratiform
  calculate_cloudy_volume(/* ncol, */ cldst, lprec_st, false, cldvst, sumppr_all_st, atm);

  // Calculate rain mixing ratio
  rain_mix_ratio(/* ncol, */ temperature, pmid, sumppr_all, rain, atm);

  delete[] source_term;
  delete[] lprec;
  delete[] lprec_st;
  delete[] lprec_cu;
  delete[] sumppr_all;
  delete[] sumppr_all_cu;
  delete[] sumppr_all_st;
}



// ==============================================================================
KOKKOS_INLINE_FUNCTION
void update_scavenging(
  const int mam_prevap_resusp_optcc,   
  const Real pdel_ik,              
  const Real omsm,   
  const Real srcc,   
  const Real srcs,      
  const Real srct,    
  const Real fins,   
  const Real finc,
  const Real fracev_st, 
  const Real fracev_cu,      
  const Real resusp_c,   
  const Real resusp_s, 
  const Real precs_ik,  
  const Real evaps_ik,       
  const Real cmfdqr_ik,  
  const Real evapc_ik,  
  Real &scavt_ik,  
  Real &iscavt_ik,      
  Real &icscavt_ik, 
  Real &isscavt_ik, 
  Real &bcscavt_ik,
  Real &bsscavt_ik,     
  Real &rcscavt_ik, 
  Real &rsscavt_ik,
  Real &scavabs,   
  Real &scavabc,        
  Real &precabc,    
  Real &precabs)
{
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

} // namespace wetdep

/// @class WedDeposition
/// Wet Deposition process for MAM4 aerosol model.
class WetDeposition {
public:
  struct Config {
    Config() = default;
    Config(const Config &) = default;
    ~Config() = default;
    Config &operator=(const Config &) = default;
  };

  const char *name() const { return "MAM4 Wet Deposition"; }

  void init(const AeroConfig &aero_config,
            const Config &wed_dep_config = Config()) {
    config_ = wed_dep_config;
  }

private:
  Config config_;
};

} // namespace mam4

#endif
