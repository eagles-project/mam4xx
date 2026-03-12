// Copyright (c) 2021, National Technology & Engineering Solutions of Sandia,
// LLC (NTESS). Copyright (c) 2022, Battelle Memorial Institute
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MAM4XX_AERO_PROCESS_HPP
#define MAM4XX_AERO_PROCESS_HPP

#include <cstring>
#include <mam4xx/atmosphere.hpp>
#include <mam4xx/surface.hpp>
#include <memory>
#include <type_traits>

namespace mam4 {

/// @class AeroProcess
/// This type defines the interface for a specific process in the aerosol
/// lifecycle, backed by a specific implementation, the structure of which is
/// defined by a specific "aerosol configuration".
template <typename AerosolConfig, typename AerosolProcessImpl>
class AeroProcess final {
public:
  // Types derived from template parameters.
  using AeroConfig = AerosolConfig;
  using Prognostics = typename AerosolConfig::Prognostics;
  using Diagnostics = typename AerosolConfig::Diagnostics;
  using Tendencies = typename AerosolConfig::Tendencies;
  using ProcessImpl = AerosolProcessImpl;
  using ProcessConfig = typename ProcessImpl::Config;

  // Tendencies type must be the same as that for Prognostics.
  static_assert(std::is_same<Tendencies, Prognostics>::value,
                "Tendencies and Prognostics types must be identical!");

  /// Constructs an instance of an aerosol process with the given name,
  /// associated with the given aerosol configuration.
  /// @param [in] aero_config The aerosol configuration for this process
  /// @param [in] process_config Any process-specific information required by
  ///                            this process's implementation.
  AeroProcess(const AeroConfig &aero_config,
              const ProcessConfig &process_config = ProcessConfig())
      : aero_config_(aero_config), process_config_(process_config),
        process_impl_() {
    // Set the name of this process.
    const char *name = process_impl_.name();
    std::strncpy(name_, name, sizeof(name_));
    // Pass the configuration data to the implementation to initialize it.
    process_impl_.init(aero_config_, process_config_);
  }

  /// Destructor.
  KOKKOS_INLINE_FUNCTION ~AeroProcess() {}

  // Copy construction is required for host -> device dispatches
  KOKKOS_INLINE_FUNCTION
  AeroProcess(const AeroProcess &) = default;

  /// Default constructor is disabled.
  AeroProcess() = delete;

  // Deep copies are not allowed.
  AeroProcess &operator=(const AeroProcess &) = delete;

  //------------------------------------------------------------------------
  //                          Accessors (host only)
  //------------------------------------------------------------------------

  /// On host: returns the name of this process.
  std::string name() const { return std::string(name_); }

  /// On host: returns the aerosol configuration (metadata) associated with
  /// this process.
  const AeroConfig &aero_config() const { return aero_config_; }

  /// On host: returns any process-specific configuration data.
  const ProcessConfig &process_config() const { return process_config_; }

  //------------------------------------------------------------------------
  //                            Public Interface
  //------------------------------------------------------------------------

  /// On host: (re-)initializes the process with the given configuration
  void init(const ProcessConfig &config) {
    process_config_ = config;
    process_impl_.init(aero_config_, process_config_);
  }

  /// On host or device: Validates input aerosol and atmosphere data, returning
  /// true if all data is physically consistent (whatever that means), and false
  /// if not.
  /// @param [in] team The Kokkos team used to run this process in a parallel
  ///                  dispatch.
  /// @param [in] atmosphere Atmosphere state variables with which to validate.
  /// @param [in] prognostics A collection of aerosol prognostic variables to be
  ///                         validated.
  KOKKOS_INLINE_FUNCTION
  bool validate(const ThreadTeam &team, const Atmosphere &atmosphere,
                const Surface &surface, const Prognostics &prognostics) const {
    return process_impl_.validate(aero_config_, team, atmosphere, surface,
                                  prognostics);
  }

  /// On host or device: runs the aerosol process at a given time with the given
  /// data.
  /// @param [in]    team The Kokkos team used to run this process in a parallel
  ///                     dispatch.
  /// @param [in]    t The simulation time at which this process is being
  ///                  invoked (in seconds).
  /// @param [in]    dt The simulation time interval ("timestep size") over
  ///                   which this process occurs.
  /// @param [in]    atmosphere The atmosphere state variables used by this
  ///                           process.
  /// @param [in]    prognostics An array containing aerosol tracer data to be
  ///                            evolved.
  /// @param [inout] diagnostics An array that can store aerosol diagnostic
  ///                            data computed or updated by this process.
  /// @param [out]   tendencies An array analogous to prognostics that
  ///                           stores computed tendencies.
  KOKKOS_INLINE_FUNCTION
  void compute_tendencies(const ThreadTeam &team, Real t, Real dt,
                          const Atmosphere &atmosphere, const Surface &surface,
                          const Prognostics &prognostics,
                          const Diagnostics &diagnostics,
                          const Tendencies &tendencies) const {
    process_impl_.compute_tendencies(aero_config_, team, t, dt, atmosphere,
                                     surface, prognostics, diagnostics,
                                     tendencies);
  }

private:
  char name_[256];
  AeroConfig aero_config_;
  ProcessConfig process_config_;
  ProcessImpl process_impl_;
};

} // namespace mam4

#endif
