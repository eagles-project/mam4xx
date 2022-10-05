#ifndef MAM4XX_AERO_CONFIG_HPP
#define MAM4XX_AERO_CONFIG_HPP

#include <algorithm>
#include <map>
#include <numeric>

#include "aero_modes.hpp"
#include "haero/view_pack_helpers.hpp"

namespace mam4 {

/// MAM4 column-wise prognostic aerosol fields (also used for tendencies).
class Prognostics final {
 public:
  using PackInfo = haero::PackInfo;
  using ColumnView = haero::ColumnView;
  using ThreadTeam = haero::ThreadTeam;

  /// Creates a container for prognostic variables on the specified number of
  /// vertical levels.
  explicit Prognostics(int num_levels) : nlev_(num_levels) {
    const int nk = PackInfo::num_packs(num_levels);
    for (int mode = 0; mode < 4; ++mode) {
      n_mode[mode] = ColumnView("n_mode", nk);
      haero::zero_init(n_mode[mode], num_levels);
      for (int spec = 0; spec < 7; ++spec) {
        q_aero[mode][spec] = ColumnView("q_aero", nk);
        haero::zero_init(q_aero[mode][spec], num_levels);
      }
    }
    for (int gas = 0; gas < 13; ++gas) {
      q_gas[gas] = ColumnView("q_gas", nk);
      haero::zero_init(q_gas[gas], num_levels);
      for (int mode = 0; mode < 4; ++mode) {
        uptkaer[gas][mode] = ColumnView("uptake_rate", nk);
        haero::zero_init(uptkaer[gas][mode], num_levels);
      }
    }
  }

  Prognostics() = default;  // Careful! Only for creating placeholders in views
  Prognostics(const Prognostics&) = default;
  ~Prognostics() = default;
  Prognostics& operator=(const Prognostics&) = default;

  /// modal aerosol number mixing ratios (see aero_mode.hpp for indexing)
  ColumnView n_mode[4];

  /// aerosol mass mixing ratios within each mode (see aero_mode.hpp for
  /// indexing)
  ColumnView q_aero[4][7];

  /// gas mass mixing ratios (see aero_mode.hpp for indexing)
  ColumnView q_gas[13];

  ColumnView uptkaer[13][4];

  KOKKOS_INLINE_FUNCTION
  int num_levels() const { return nlev_; }

  /// Returns true iff all prognostic quantities are nonnegative, using the
  /// given thread team to parallelize the check.
  KOKKOS_INLINE_FUNCTION
  bool quantities_nonnegative(const ThreadTeam& team) const {
    const int nk = PackInfo::num_packs(num_levels());
    int violations = 0;
    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team, nk),
        KOKKOS_CLASS_LAMBDA(int k, int& violation) {
          for (int mode = 0; mode < 4; ++mode) {  // check mode mmrs
            if ((n_mode[mode](k) < 0).any()) {
              ++violation;
            } else {
              for (int spec = 0; spec < 7; ++spec) {  // check aerosol mmrs
                if ((q_aero[mode][spec](k) < 0).any()) {
                  ++violation;
                  break;
                }
              }
            }
            if (violation > 0) break;
          }
          if (violation == 0) {
            for (int gas = 0; gas < 13; ++gas) {  // check gas mmrs
              if ((q_gas[gas](k) < 0).any()) ++violation;
            }
          }
        },
        violations);
    return (violations == 0);
  }

 private:
  int nlev_;
};

// Tendencies are identical in structure to prognostics.
using Tendencies = Prognostics;

/// MAM4 column-wise diagnostic aerosol fields.
class Diagnostics final {
 public:
  using ColumnView = haero::ColumnView;
  using PackInfo = haero::PackInfo;

  explicit Diagnostics(int num_levels) : nlev_(num_levels) {
    const int nk = PackInfo::num_packs(num_levels);
    for (int mode = 0; mode < 4; ++mode) {
      dry_geometric_mean_diameter[mode] =
          ColumnView("dry_geometric_mean_diameter", nk);
      haero::zero_init(dry_geometric_mean_diameter[mode], num_levels);
      wet_geometric_mean_diameter[mode] =
          ColumnView("wet_geometric_mean_diameter", nk);
      haero::zero_init(wet_geometric_mean_diameter[mode], num_levels);
    }
  }
  Diagnostics() = default;  // Careful! Only for creating placeholders in views
  Diagnostics(const Diagnostics&) = default;
  ~Diagnostics() = default;
  Diagnostics& operator=(const Diagnostics&) = default;

  int num_levels() const { return nlev_; }

  ColumnView dry_geometric_mean_diameter[4];
  ColumnView wet_geometric_mean_diameter[4];

  /// For gas-aerosol exchange process
  ColumnView uptkrate_h2so4;

 private:
  int nlev_;
};

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
  AeroConfig(const AeroConfig&) = default;

  // Destructor.
  ~AeroConfig() = default;

  // Assignment operator.
  AeroConfig& operator=(const AeroConfig&) = default;

  // Comparison operators.
  inline bool operator==(const AeroConfig& other) const {
    return true;  // all MAM4 configs are equivalent
  }
  inline bool operator!=(const AeroConfig& other) const {
    return false;  // all MAM4 configs are equivalent
  }

  /// Returns the number of aerosol modes.
  static constexpr int num_modes() { return 4; }

  /// Returns the number of aerosol ids. This is the number of enums in
  /// mam4::AeroId.
  static constexpr int num_aerosol_ids() { return 7; }

  /// Returns the number of gas ids. This is the number of enums in mam4::GasId.
  static constexpr int num_gas_ids() { return 13; }
};

}  // namespace mam4

#endif
