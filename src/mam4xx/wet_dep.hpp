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
 * @param[in] is_tot_cld Flag for ???
 *      -   TODO!! - unsure what this really means.
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
      }
  }
}

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
