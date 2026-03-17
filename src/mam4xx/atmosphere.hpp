// Copyright (c) 2021, National Technology & Engineering Solutions of Sandia,
// LLC (NTESS). Copyright (c) 2022, Battelle Memorial Institute
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MAM4XX_ATMOSPHERE_HPP
#define MAM4XX_ATMOSPHERE_HPP

#include <mam4xx/aero_config.hpp>

#include <ekat_kernel_assert.hpp>

namespace mam4 {

/// @class Atmosphere
/// This type stores atmospheric state variables inherited from a host model.
class Atmosphere final {
  // number of vertical levels
  int num_levels_;

public:
  /// Constructs an Atmosphere object holding the state for a single atmospheric
  /// column with the given planetary boundary layer height.
  /// All views must be set manually elsewhere or provided by a host model.
  KOKKOS_INLINE_FUNCTION
  Atmosphere(int num_levels, const ConstColumnView T, const ConstColumnView p,
             const ConstColumnView qv, const ConstColumnView qc,
             const ConstColumnView nqc, const ConstColumnView qi,
             const ConstColumnView nqi, const ConstColumnView z,
             const ConstColumnView hdp, const ConstColumnView intp,
             const ConstColumnView cf, const ConstColumnView w, const Real pblh)
      : num_levels_(num_levels), temperature(T), pressure(p),
        vapor_mixing_ratio(qv), liquid_mixing_ratio(qc),
        cloud_liquid_number_mixing_ratio(nqc), ice_mixing_ratio(qi),
        cloud_ice_number_mixing_ratio(nqi), height(z), hydrostatic_dp(hdp),
        interface_pressure(intp), cloud_fraction(cf),
        updraft_vel_ice_nucleation(w), planetary_boundary_layer_height(pblh) {

#ifndef NDEBUG
    auto nlev = static_cast<std::size_t>(num_levels_);
#endif
    EKAT_KERNEL_ASSERT(T.extent(0) >= nlev);
    EKAT_KERNEL_ASSERT(p.extent(0) >= nlev);
    EKAT_KERNEL_ASSERT(qv.extent(0) >= nlev);
    EKAT_KERNEL_ASSERT(qc.extent(0) >= nlev);
    EKAT_KERNEL_ASSERT(nqc.extent(0) >= nlev);
    EKAT_KERNEL_ASSERT(qi.extent(0) >= nlev);
    EKAT_KERNEL_ASSERT(nqi.extent(0) >= nlev);
    EKAT_KERNEL_ASSERT(z.extent(0) >= nlev);
    EKAT_KERNEL_ASSERT(hdp.extent(0) >= nlev);
    EKAT_KERNEL_ASSERT(cf.extent(0) >= nlev);
    EKAT_KERNEL_ASSERT(w.extent(0) >= nlev);

    EKAT_KERNEL_ASSERT(pblh >= 0.0);
  }

  // use only for creating containers of Atmospheres!
  KOKKOS_INLINE_FUNCTION
  Atmosphere() = default;

  // these are supported for initializing containers of Atmospheres
  KOKKOS_INLINE_FUNCTION
  Atmosphere(const Atmosphere &rhs) = default;
  KOKKOS_INLINE_FUNCTION
  Atmosphere &operator=(const Atmosphere &rhs) = default;

  /// destructor, valid on both host and device
  KOKKOS_INLINE_FUNCTION
  ~Atmosphere() {}

  // views storing atmospheric state data for a single vertical column

  /// temperature [K]
  ConstColumnView temperature;

  /// pressure [Pa]
  ConstColumnView pressure;

  /// water vapor mass mixing ratio [kg vapor/kg dry air]
  ConstColumnView vapor_mixing_ratio;

  /// liquid water mass mixing ratio [kg vapor/kg dry air]
  ConstColumnView liquid_mixing_ratio;

  /// grid box averaged cloud liquid number mixing ratio [#/kg dry air]
  ConstColumnView cloud_liquid_number_mixing_ratio;

  /// ice water mass mixing ratio [kg vapor/kg dry air]
  ConstColumnView ice_mixing_ratio;

  // grid box averaged cloud ice number mixing ratio [#/kg dry air]
  ConstColumnView cloud_ice_number_mixing_ratio;

  /// height at the midpoint of each vertical level [m]
  ConstColumnView height;

  /// hydroѕtatic "pressure thickness" defined as the difference in hydrostatic
  /// pressure levels between the interfaces bounding a vertical level [Pa]
  ConstColumnView hydrostatic_dp;

  /// Pressure at the level interfaces. Length is number of 1 + the
  /// number of levels.  [Pa]
  ConstColumnView interface_pressure;

  /// cloud fraction [-]
  ConstColumnView cloud_fraction;

  /// vertical updraft velocity used for ice nucleation [m/s]
  ConstColumnView updraft_vel_ice_nucleation;

  // column-specific planetary boundary layer height [m]
  Real planetary_boundary_layer_height;

  /// returns the number of vertical levels per column in the system
  KOKKOS_INLINE_FUNCTION
  int num_levels() const { return num_levels_; }

  /// Returns true iff all atmospheric quantities are nonnegative, using the
  /// given thread team to parallelize the check.
  KOKKOS_INLINE_FUNCTION
  bool quantities_nonnegative(const ThreadTeam &team) const {
    const int nk = num_levels();
    int violations = 0;
    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team, nk),
        KOKKOS_CLASS_LAMBDA(int k, int &violation) {
          if ((temperature(k) < 0) || (pressure(k) < 0) ||
              (vapor_mixing_ratio(k) < 0) || (liquid_mixing_ratio(k) < 0) ||
              (ice_mixing_ratio(k) < 0) ||
              (cloud_liquid_number_mixing_ratio(k) < 0) ||
              (cloud_ice_number_mixing_ratio(k) < 0)) {
            violation = 1;
          }
        },
        violations);
    return (violations == 0);
  }
};

} // namespace mam4

#endif
