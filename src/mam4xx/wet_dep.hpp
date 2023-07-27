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
