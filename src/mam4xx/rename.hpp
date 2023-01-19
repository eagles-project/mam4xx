#ifndef MAM4XX_RENAME_HPP
#define MAM4XX_RENAME_HPP

#include <haero/atmosphere.hpp>
#include <haero/math.hpp>

#include <mam4xx/aero_config.hpp>
#include <mam4xx/conversions.hpp>
#include <mam4xx/mam4_types.hpp>

namespace mam4 {

using haero::max;
using haero::min;
using haero::sqrt;

namespace rename {} // namespace rename

/// @class Rename
/// This class implements MAM4's rename parameterization.
class Rename {
public:
  // nucleation-specific configuration
  struct Config {
    // default constructor -- sets default values for parameters

    Config() {}

    Config(const Config &) = default;
    ~Config() = default;
    Config &operator=(const Config &) = default;
  };

private:
  Config config_;

public:
  // name -- unique name of the process implemented by this class
  const char *name() const { return "MAM4 rename"; }

  // init -- initializes the implementation with MAM4's configuration and with
  // a process-specific configuration.
  void init(const AeroConfig &aero_config,
            const Config &rename_config = Config()) {
    // Set nucleation-specific config parameters.
    config_ = rename_config;

  } // end(init)

  KOKKOS_INLINE_FUNCTION
  void compute_tendencies(const AeroConfig &config, const ThreadTeam &team,
                          Real t, Real dt, const Atmosphere &atmosphere,
                          const Prognostics &prognostics,
                          const Diagnostics &diagnostics,
                          const Tendencies &tendencies) const {}

private:
};

} // namespace mam4

#endif
