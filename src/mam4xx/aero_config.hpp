// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MAM4XX_AERO_CONFIG_HPP
#define MAM4XX_AERO_CONFIG_HPP

#include <mam4xx/aero_modes.hpp>

#include <algorithm>
#include <map>
#include <numeric>

namespace mam4 {

class Prognostics; // fwd decl
class Diagnostics; // fwd decl
// Tendencies are identical in structure to prognostics.
using Tendencies = Prognostics; // fwd decl

/// @struct MAM4::AeroConfig: for use with all MAM4 process implementations
class AeroConfig final {
public:
  // Types.
  using Prognostics = ::mam4::Prognostics;
  using Diagnostics = ::mam4::Diagnostics;
  using Tendencies = ::mam4::Tendencies;

  bool calculate_gas_uptake_coefficient = false;
  int number_gauss_points_for_integration = 2;
  // Default constructor.
  AeroConfig() {}

  // Copy constructor.
  AeroConfig(const AeroConfig &) = default;

  // Destructor.
  ~AeroConfig() = default;

  // Assignment operator.
  AeroConfig &operator=(const AeroConfig &) = default;

  // Comparison operators.
  inline bool operator==(const AeroConfig &other) const {
    return true; // all MAM4 configs are equivalent
  }
  inline bool operator!=(const AeroConfig &other) const {
    return false; // all MAM4 configs are equivalent
  }

  /// Returns the number of aerosol modes.
  static constexpr int num_modes() { return 4; }

  /// Returns the number of aerosol ids. This is the number of enums in
  /// mam4::AeroId.
  static constexpr int num_aerosol_ids() { return 7; }

  /// Returns the number of gas ids. This is the number of enums in mam4::GasId.
  static constexpr int num_gas_ids() { return 3; }

  /// Returns the number of aging pairs
  static constexpr int max_agepair() { return 1; }
};

/// MAM4 column-wise prognostic aerosol fields (also used for tendencies).
class Prognostics final {
public:
  using ColumnView = haero::ColumnView;
  using ThreadTeam = haero::ThreadTeam;

  /// Creates a container for prognostic variables on the specified number of
  /// vertical levels.
  explicit Prognostics(int num_levels) : nlev_(num_levels) {
    for (int mode = 0; mode < AeroConfig::num_modes(); ++mode) {
      n_mode_i[mode] = ColumnView("n_mode_i", num_levels);
      n_mode_c[mode] = ColumnView("n_mode_c", num_levels);
      Kokkos::deep_copy(n_mode_i[mode], 0.0);
      Kokkos::deep_copy(n_mode_c[mode], 0.0);
      for (int spec = 0; spec < AeroConfig::num_aerosol_ids(); ++spec) {
        q_aero_i[mode][spec] = ColumnView("q_aero_i", num_levels);
        q_aero_c[mode][spec] = ColumnView("q_aero_c", num_levels);
        Kokkos::deep_copy(q_aero_i[mode][spec], 0.0);
        Kokkos::deep_copy(q_aero_c[mode][spec], 0.0);
      }
    }
    for (int gas = 0; gas < AeroConfig::num_gas_ids(); ++gas) {
      q_gas[gas] = ColumnView("q_gas", num_levels);
      q_gas_avg[gas] = ColumnView("q_gas_avg", num_levels);
      Kokkos::deep_copy(q_gas[gas], 0.0);
      Kokkos::deep_copy(q_gas_avg[gas], 0.0);
      for (int mode = 0; mode < AeroConfig::num_modes(); ++mode) {
        uptkaer[gas][mode] = ColumnView("uptake_rate", num_levels);
        Kokkos::deep_copy(uptkaer[gas][mode], 0.0);
      }
    }
  }

  Prognostics() = default; // Careful! Only for creating placeholders in views
  Prognostics(const Prognostics &) = default;
  ~Prognostics() = default;
  Prognostics &operator=(const Prognostics &) = default;

  ///  modal interstitial aerosol number mixing ratios (see aero_mode.hpp for
  ///  indexing)
  ColumnView n_mode_i[AeroConfig::num_modes()];

  /// modal cloudborne aerosol number mixing ratios (see aero_mode.hpp for
  /// indexing)
  ColumnView n_mode_c[AeroConfig::num_modes()];

  /// interstitial aerosol mass mixing ratios within each mode
  /// (see aero_mode.hpp for indexing)
  ColumnView q_aero_i[AeroConfig::num_modes()][AeroConfig::num_aerosol_ids()];

  /// cloudborne aerosol mass mixing ratios within each mode
  /// (see aero_mode.hpp for indexing)
  ColumnView q_aero_c[AeroConfig::num_modes()][AeroConfig::num_aerosol_ids()];

  /// gas mass mixing ratios (see aero_mode.hpp for indexing)
  ColumnView q_gas[AeroConfig::num_gas_ids()];

  /// time average of the gas mix ratios over the time step of
  /// integration (see aero_mode.hpp for indexing)
  ColumnView q_gas_avg[AeroConfig::num_gas_ids()];

  /// Uptate Rate for each gas species and each mode.
  /// i.e. Gas to aerosol mass transfer rate (1/s)
  ColumnView uptkaer[AeroConfig::num_gas_ids()][AeroConfig::num_modes()];

  KOKKOS_INLINE_FUNCTION
  int num_levels() const { return nlev_; }

  /// Returns true iff all prognostic quantities are nonnegative, using the
  /// given thread team to parallelize the check.
  KOKKOS_INLINE_FUNCTION
  bool quantities_nonnegative(const ThreadTeam &team) const {
    const int nk = num_levels();
    int violations = 0;
    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team, nk),
        KOKKOS_CLASS_LAMBDA(int k, int &violation) {
          for (int mode = 0; mode < AeroConfig::num_modes();
               ++mode) { // check mode mmrs
            if ((n_mode_i[mode](k) < 0) || (n_mode_c[mode](k) < 0)) {
              ++violation;
            } else {
              for (int spec = 0; spec < AeroConfig::num_aerosol_ids();
                   ++spec) { // check aerosol mmrs
                if ((q_aero_i[mode][spec](k) < 0) ||
                    (q_aero_c[mode][spec](k) < 0)) {
                  ++violation;
                  break;
                }
              }
            }
            if (violation > 0)
              break;
          }
          if (violation == 0) {
            for (int gas = 0; gas < AeroConfig::num_gas_ids();
                 ++gas) { // check gas mmrs
              if (q_gas[gas](k) < 0)
                ++violation;
              if (q_gas_avg[gas](k) < 0)
                ++violation;
            }
          }
        },
        violations);
    return (violations == 0);
  }

private:
  int nlev_;
};

/// MAM4 column-wise diagnostic aerosol fields.
class Diagnostics final {
public:
  using ColumnView = haero::ColumnView;

  explicit Diagnostics(int num_levels) : nlev_(num_levels) {
    for (int mode = 0; mode < AeroConfig::num_modes(); ++mode) {
      hygroscopicity[mode] = ColumnView("hygroscopicity", num_levels);
      Kokkos::deep_copy(hygroscopicity[mode], 0.0);
      dry_geometric_mean_diameter_i[mode] =
          ColumnView("dry_geometric_mean_diameter_interstitial", num_levels);
      dry_geometric_mean_diameter_c[mode] =
          ColumnView("dry_geometric_mean_diameter_cloudborne", num_levels);
      dry_geometric_mean_diameter_total[mode] =
          ColumnView("dry_geometric_mean_diameter_total", num_levels);
      Kokkos::deep_copy(dry_geometric_mean_diameter_i[mode], 0.0);
      Kokkos::deep_copy(dry_geometric_mean_diameter_c[mode], 0.0);
      Kokkos::deep_copy(dry_geometric_mean_diameter_total[mode], 0.0);
      wet_geometric_mean_diameter_i[mode] =
          ColumnView("wet_geometric_mean_diameter_interstitial", num_levels);
      wet_geometric_mean_diameter_c[mode] =
          ColumnView("wet_geometric_mean_diameter_cloudborne", num_levels);
      Kokkos::deep_copy(wet_geometric_mean_diameter_i[mode], 0.0);
      Kokkos::deep_copy(wet_geometric_mean_diameter_c[mode], 0.0);

      wet_density[mode] = ColumnView("wet_density", num_levels);
      Kokkos::deep_copy(wet_density[mode], 0.0);
    }
    uptkrate_h2so4 = ColumnView("uptkrate_h2so4", num_levels);
    Kokkos::deep_copy(uptkrate_h2so4, 0.0);
    g0_soa_out = ColumnView("g0_soa_out", num_levels);
    Kokkos::deep_copy(g0_soa_out, 0.0);
    iscloudy = haero::DeviceType::view_1d<bool>("is_cloudy_bool", num_levels);
    Kokkos::deep_copy(iscloudy, false);
    num_substeps = haero::DeviceType::view_1d<int>("num_substeps", num_levels);
    Kokkos::deep_copy(num_substeps, 0);
  }
  Diagnostics() = default; // Careful! Only for creating placeholders in views
  Diagnostics(const Diagnostics &) = default;
  ~Diagnostics() = default;
  Diagnostics &operator=(const Diagnostics &) = default;

  int num_levels() const { return nlev_; }

  /// Hygroscopicity is a modal mass-weighted average over all species
  /// in a mode
  ColumnView hygroscopicity[AeroConfig::num_modes()];

  /// Total dry particle diameter is a modal mass-weighted average over
  /// all species of interstitial AND cloudborne aerosols in a mode
  ColumnView dry_geometric_mean_diameter_total[AeroConfig::num_modes()];

  /// Dry particle diameter is a modal mass-weighted average over all species
  /// of interstitial aerosols in a mode
  ColumnView dry_geometric_mean_diameter_i[AeroConfig::num_modes()];

  /// Dry particle diameter is a modal mass-weighted average over all species
  /// of cloudborne aerosols in a mode
  ColumnView dry_geometric_mean_diameter_c[AeroConfig::num_modes()];

  /// Wet particle diameter is a modal mass-weighted average over all species
  /// of interstitial aerosols in a mode
  ColumnView wet_geometric_mean_diameter_i[AeroConfig::num_modes()];

  /// Wet particle diameter is a modal mass-weighted average over all species
  /// of cloudborne in a mode
  ColumnView wet_geometric_mean_diameter_c[AeroConfig::num_modes()];

  // Aerosol wet density
  ColumnView wet_density[AeroConfig::num_modes()];

  /// For gas-aerosol exchange process
  /// Uptake rate coefficient of H2SO4 gas, summed over all modes
  ColumnView uptkrate_h2so4;
  /// Ambient SOA gas equilib mixing rate (mol/mol at actual mw)
  ColumnView g0_soa_out;

  /// boolean indicating whether cloudborne aerosols are present in a cell
  haero::DeviceType::view_1d<bool> iscloudy;

  /// Number of time substeps needed to converge in mam_soaexch_advance_in_time
  haero::DeviceType::view_1d<int> num_substeps;

private:
  int nlev_;
};

} // namespace mam4

#endif
