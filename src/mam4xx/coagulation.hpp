#ifndef MAM4XX_COAGULATION_HPP
#define MAM4XX_COAGULATION_HPP

#include <mam4xx/aero_config.hpp>
#include <mam4xx/mam4_types.hpp>

#include <Kokkos_Array.hpp>
#include <haero/atmosphere.hpp>
#include <haero/constants.hpp>
#include <haero/haero.hpp>
#include <iomanip>
#include <iostream>

namespace mam4 {

/// @class Coagulation
/// This class implements MAM4's gas/aersol exchange  parameterization. Its
/// structure is defined by the usage of the impl_ member in the AeroProcess
/// class in
/// ../aero_process.hpp.
class Coagulation {
public:
  // process-specific configuration data (if any)
  struct Config {
    Config() {}
    Config(const Config &) = default;
    ~Config() = default;
    Config &operator=(const Config &) = default;
  };

  // name -- unique name of the process implemented by this class
  const char *name() const { return "MAM4 Coagulation"; }

  // init -- initializes the implementation with MAM4's configuration
  void init(const AeroConfig &aero_config,
            const Config &process_config = Config());

  // validate -- validates the given atmospheric state and prognostics against
  // assumptions made by this implementation, returning true if the states are
  // valid, false if not
  KOKKOS_INLINE_FUNCTION
  bool validate(const AeroConfig &config, const ThreadTeam &team,
                const Atmosphere &atm, const Prognostics &progs) const {
    // TODO
    return true;
  }

  // compute_tendencies -- computes tendencies and updates diagnostics
  // NOTE: that both diags and tends are const below--this means their views
  // NOTE: are fixed, but the data in those views is allowed to vary.
  KOKKOS_INLINE_FUNCTION
  void compute_tendencies(const AeroConfig &config, const ThreadTeam &team,
                          Real t, Real dt, const Atmosphere &atm,
                          const Prognostics &progs, const Diagnostics &diags,
                          const Tendencies &tends) const;

private:
  // Gas-Aerosol-Exchange-specific configuration
  Config config_;
};

namespace coagulation {


} // namespace coagulation

// init -- initializes the implementation with MAM4's configuration
inline void Coagulation::init(const AeroConfig &aero_config,
                             const Config &process_config) {
  // TODO
  config_ = process_config;
}

// compute_tendencies -- computes tendencies and updates diagnostics
// NOTE: that both diags and tends are const below--this means their views
// NOTE: are fixed, but the data in those views is allowed to vary.
KOKKOS_INLINE_FUNCTION
void Coagulation::compute_tendencies(const AeroConfig &config,
                                    const ThreadTeam &team, Real t, Real dt,
                                    const Atmosphere &atm,
                                    const Prognostics &progs,
                                    const Diagnostics &diags,
                                    const Tendencies &tends) const 
{
    printf("Hello!\n");
    // TODO
}
} // namespace mam4

#endif