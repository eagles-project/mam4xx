// Copyright (c) 2021, National Technology & Engineering Solutions of Sandia,
// LLC (NTESS). Copyright (c) 2022, Battelle Memorial Institute
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MAM4XX_AERO_UTILS_HPP
#define MAM4XX_AERO_UTILS_HPP

#include "gas_species.hpp"

#include <functional>

namespace mam4 {

/// @struct AeroUtils
/// This is just a grab bag of utility functions that work with an aerosol
/// package with the given configuration type.
template <typename AeroConfig> struct AeroUtils final {

  // Types derived from template parameters.
  using Config = AeroConfig;
  using AeroMD = typename AeroConfig::AeroMetadata;

  // You can't create one of these classes--it's just a templated namespace.
  AeroUtils() = delete;
  AeroUtils(const AeroUtils &) = delete;
  ~AeroUtils() = delete;

  /// On host: calls a given function, passing it each of the mass mixing ratios
  /// for the aerosols in the related configuration.
  /// @param [in] config The aerosol configuration for which mass mixing ratios
  ///                    are to be manipulated.
  /// @param [in] f The function to be called with the given arguments. Usually
  ///               f is a lambda that is allowed to capture values from the
  ///               call site.
  static void foreach_aero_mmr(const Config &config,
                               std::function<void(const AeroMD &)> f) {
    config.foreach_aero_mmr(f);
  }

  /// On host: calls a given function, passing it each of the number mixing
  /// ratios for the aerosols in the related configuration.
  /// @param [in] config The aerosol configuration for which number mixing
  ///                    ratios are to be manipulated.
  /// @param [in] f The function to be called with the given arguments. Usually
  ///               f is a lambda that is allowed to capture values from the
  ///               call site.
  static void foreach_aero_nmr(const Config &config,
                               std::function<void(const AeroMD &)> f) {
    config.foreach_aero_nmr(f);
  }

  /// On host: calls a given function, passing it each of the gases
  /// for the related aerosol configuration.
  /// @param [in] config The aerosol configuration for which gas species are to
  ///                    be manipulated.
  /// @param [in] f The function to be called with the given arguments. Usually
  ///               f is a lambda that is allowed to capture values from the
  ///               call site.
  static void foreach_gas(const Config &config,
                          std::function<void(const GasSpecies &)> f) {
    config.foreach_gas(f);
  }
};

} // namespace mam4

#endif
