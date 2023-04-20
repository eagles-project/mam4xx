// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MAM4XX_WET_DEPOSITION_HPP
#define MAM4XX_WET_DEPOSITION_HPP

#include <haero/atmosphere.hpp>
#include <haero/constants.hpp>
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

private:
  Config config_;

public:
  const char *name() const { return "MAM4 Wet Deposition"; }

  void init(const AeroConfig &aero_config,
            const Config &wed_dep_config = Config()) {
    config_ = wed_dep_config;
  }
};

} // namespace mam4

#endif
