// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MAM4XX_DRYDEP_HPP
#define MAM4XX_DRYDEP_HPP

#include <haero/atmosphere.hpp>
#include <mam4xx/aero_config.hpp>

namespace mam4 {

class DryDep {

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
  const char *name() const { return "MAM4 dry deposition"; }

  // init -- initializes the implementation with MAM4's configuration
  void init(const AeroConfig &aero_config,
            const Config &process_config = Config());

  // validate -- validates the given atmospheric state and prognostics against
  // assumptions made by this implementation, returning true if the states are
  // valid, false if not
  KOKKOS_INLINE_FUNCTION
  bool validate(const AeroConfig &config, const ThreadTeam &team,
                const Atmosphere &atm, const Prognostics &progs) const;

  // compute_tendencies -- computes tendencies and updates diagnostics
  // NOTE: that both diags and tends are const below--this means their views
  // NOTE: are fixed, but the data in those views is allowed to vary.
  KOKKOS_INLINE_FUNCTION
  void compute_tendencies(const AeroConfig &config, const ThreadTeam &team,
                          Real t, Real dt, const Atmosphere &atm,
                          const Prognostics &progs, const Diagnostics &diags,
                          const Tendencies &tends) const;
};

namespace hetfrz {

KOKKOS_INLINE_FUNCTION
Real gravit_settling_velocity(const Real particle_radius,
                              const Real particle_density,
                              const Real slip_correction,
                              const Real dynamic_viscosity,
                              const Real particle_sig) {

  // Calculate terminal velocity following, e.g.,
  //  -  Seinfeld and Pandis (1997),  p. 466
  //  - Zhang L. et al. (2001), DOI: 10.1016/S1352-2310(00)00326-5, Eq. 2.

  const Real gravit_settling_velocity =
      (4.0 / 18.0) * particle_radius * particle_radius * particle_density *
      Constants::gravity * slip_correction / dynamic_viscosity;

  // Account for size distribution (i.e., we are calculating the bulk velocity
  // for a particle population instead of a single particle).

  const Real lnsig = haero::log(particle_sig);
  const Real dispersion = haero::exp(2.0 * lnsig * lnsig);

  return gravit_settling_velocity * dispersion;
}

} // namespace hetfrz

} // namespace mam4

#endif