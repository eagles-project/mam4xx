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
  // number of vertical levels
  int nlev_;

public:
  using ColumnView = haero::ColumnView;
  using ThreadTeam = haero::ThreadTeam;

  /// Creates a container for prognostic variables on the specified number of
  /// vertical levels. All views must be set manually.
  explicit Prognostics(int num_levels) : nlev_(num_levels) {}

  Prognostics() = default; // use only for creating containers of Prognostics!
  ~Prognostics() = default;

  // these are used to populate containers of Prognostics objects
  Prognostics(const Prognostics &rhs) = default;
  Prognostics &operator=(const Prognostics &rhs) = default;

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
};

/// MAM4 column-wise diagnostic aerosol fields.
class Diagnostics final {
  // number of vertical levels
  int nlev_;

public:
  using ColumnView = haero::ColumnView;

  /// Creates a container for diagnostic variables on the specified number of
  /// vertical levels. All views must be set manually.
  explicit Diagnostics(int num_levels) : nlev_(num_levels) {}

  Diagnostics() = default; // use only for creating containers of Diagnostics!
  ~Diagnostics() = default;

  // these are used to populate containers of Diagnostics objects
  Diagnostics(const Diagnostics &rhs) = default;
  Diagnostics &operator=(const Diagnostics &rhs) = default;

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
  haero::DeviceType::view_1d<bool> is_cloudy;

  /// Number of time substeps needed to converge in mam_soaexch_advance_in_time
  haero::DeviceType::view_1d<int> num_substeps;

  // Output variables for nucleate_ice process:
  // Ask experts for better names for: icenuc_num_hetfrz, icenuc_num_immfrz,
  // nihf
  // icenuc_num_depnuc,  icenuc_num_meydep output number conc of ice nuclei due
  // to heterogeneous freezing [1/m3]
  ColumnView icenuc_num_hetfrz;
  // niimm
  //  output number conc of ice nuclei due to immersion freezing (hetero nuc)
  //  [1/m3]
  ColumnView icenuc_num_immfrz;
  // nidep
  // output number conc of ice nuclei due to deposition nucleation (hetero nuc)
  // [1/m3]
  ColumnView icenuc_num_depnuc;
  // nimey
  // !output number conc of ice nuclei due to meyers deposition [1/m3]
  ColumnView icenuc_num_meydep;
  // number of activated aerosol for ice nucleation (homogeneous freezing only)
  // [#/kg]
  ColumnView num_act_aerosol_ice_nucle_hom;
  // number of activated aerosol for ice nucleation [#/kg]
  ColumnView num_act_aerosol_ice_nucle;
};

} // namespace mam4

#endif
