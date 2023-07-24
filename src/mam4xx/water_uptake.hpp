// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MAM4XX_WATER_UPTAKE_HPP
#define MAM4XX_WATER_UPTAKE_HPP

#include <haero/atmosphere.hpp>
#include <haero/surface.hpp>
#include <mam4xx/aero_config.hpp>

namespace mam4 {
class Water_Uptake {

public:
  struct Config {

    Config(){};

    Config(const Config &) = default;
    ~Config() = default;
    Config &operator=(const Config &) = default;
  };

private:
  Config config_;

public:
  // name -- unique name of the process implemented by this class
  const char *name() const { return "MAM4 wet deposition"; }

  // init -- initializes the implementation with MAM4's configuration
  void init(const AeroConfig &aero_config,
            const Config &process_config = Config());

  // validate -- validates the given atmospheric state and prognostics against
  // assumptions made by this implementation, returning true if the states are
  // valid, false if not
  KOKKOS_INLINE_FUNCTION
  bool validate(const AeroConfig &config, const ThreadTeam &team,
                const Atmosphere &atm, const Surface &sfc,
                const Prognostics &progs) const;

  // compute_tendencies -- computes tendencies and updates diagnostics
  // NOTE: that both diags and tends are const below--this means their views
  // NOTE: are fixed, but the data in those views is allowed to vary.
  KOKKOS_INLINE_FUNCTION
  void compute_tendencies(const AeroConfig &config, const ThreadTeam &team,
                          Real t, Real dt, const Atmosphere &atm,
                          const Surface &sfc, const Prognostics &progs,
                          const Diagnostics &diags,
                          const Tendencies &tends) const;
};

namespace water_uptake {
KOKKOS_INLINE_FUNCTION
void modal_aero_wateruptake_wetdens(
    const Real wetvol[AeroConfig::num_modes()],
    const Real wtrvol[AeroConfig::num_modes()],
    const Real drymass[AeroConfig::num_modes()],
    const Real specdens_1[AeroConfig::num_modes()],
    Real wetdens[AeroConfig::num_modes()]) {

  // compute aerosol wet density (kg/m3)
  // looping over densities
  for (int imode = 0; imode < AeroConfig::num_modes(); ++imode) {
    static constexpr Real small_value_30 = 1.0e-30; // (BAD CONSTANT)
    if (wetvol[imode] > small_value_30) {
      // ! wet density
      wetdens[imode] =
          (drymass[imode] + Constants::density_h2o * wtrvol[imode]) /
          wetvol[imode];
    } else {
      // dry density
      wetdens[imode] = specdens_1[imode];
    }
  }
}
}; // namespace water_uptake
} // namespace mam4

#endif