// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include "testing.hpp"

#include <ekat_fpe.hpp>
#include <ekat_kokkos_session.hpp>

// the testing namespace contains functions that are useful only within tests,
// not to be used in production code
namespace mam4::testing {

// A simple memory allocation pool for standalone ColumnViews to be used in
// (e.g.) unit tests. A ColumnPool manages a number of ColumnViews with a fixed
// number of vertical levels.
class ColumnPool {
  size_t num_levels_;          // number of vertical levels per column (fixed)
  size_t num_cols_;            // number of allocated columns
  std::vector<int> col_used_;  // columns that are being used already
  std::vector<Real *> memory_; // per-column memory itself (allocated on device)
public:
  // constructs a column pool with the given initial number of columns, each
  // with the given number of vertical levels.`
  ColumnPool(size_t num_vertical_levels, size_t initial_num_columns = 64)
      : num_levels_(num_vertical_levels), num_cols_(initial_num_columns),
        col_used_(initial_num_columns, 0),
        memory_(initial_num_columns, nullptr) {
    for (size_t i = 0; i < num_cols_; ++i) {
      memory_[i] = reinterpret_cast<Real *>(Kokkos::kokkos_malloc(
          "Column pool", sizeof(Real) * num_vertical_levels));
    }
  }

  // no copying of the pool

  // destructor
  ~ColumnPool() {
    for (size_t i = 0; i < num_cols_; ++i) {
      Kokkos::kokkos_free(memory_[i]);
    }
  }

  // returns a "fresh" (unused) ColumnView from the ColumnPool, marking it as
  // used, and allocating additional memory if needed)
  ColumnView column_view() {
    // find the first unused column
    size_t i;
    for (i = 0; i < num_cols_; ++i) {
      if (!col_used_[i])
        break;
    }
    if (i == num_cols_) { // all columns in the pool are in use!
      // double the number of allocated columns in the pool
      size_t new_num_cols = 2 * num_cols_;
      col_used_.resize(new_num_cols, 0);
      memory_.resize(new_num_cols, nullptr);
      for (size_t i = num_cols_; i < new_num_cols; ++i) {
        memory_[i] = reinterpret_cast<Real *>(
            Kokkos::kokkos_malloc(sizeof(Real) * num_levels_));
      }
      num_cols_ = new_num_cols;
    }

    col_used_[i] = 1;
    return ColumnView(memory_[i], num_levels_);
  }
};

// column pools, organized by column resolution
std::map<size_t, std::unique_ptr<ColumnPool>> pools_{};

mam4::Atmosphere create_atmosphere(int num_levels, mam4::Real pblh) {
  auto temperature = create_column_view(num_levels);
  auto pressure = create_column_view(num_levels);
  auto vapor_mixing_ratio = create_column_view(num_levels);
  auto liquid_mixing_ratio = create_column_view(num_levels);
  auto cloud_liquid_number_mixing_ratio = create_column_view(num_levels);
  auto ice_mixing_ratio = create_column_view(num_levels);
  auto cloud_ice_number_mixing_ratio = create_column_view(num_levels);
  auto height = create_column_view(num_levels);
  auto hydrostatic_dp = create_column_view(num_levels);
  auto interface_pressure = create_column_view(num_levels + 1);
  auto cloud_fraction = create_column_view(num_levels);
  auto updraft_vel_ice_nucleation = create_column_view(num_levels);
  return Atmosphere(num_levels, temperature, pressure, vapor_mixing_ratio,
                    liquid_mixing_ratio, cloud_liquid_number_mixing_ratio,
                    ice_mixing_ratio, cloud_ice_number_mixing_ratio, height,
                    hydrostatic_dp, interface_pressure, cloud_fraction,
                    updraft_vel_ice_nucleation, pblh);
}

ColumnView create_column_view(int num_levels) {
  // find a column pool for the given number of vertical levels
  auto iter = pools_.find(num_levels);
  if (iter == pools_.end()) {
    auto result = pools_.emplace(
        num_levels, std::unique_ptr<ColumnPool>(new ColumnPool(num_levels)));
    iter = result.first;
  }
  return iter->second->column_view();
}

void finalize() { pools_.clear(); }

Surface create_surface() { return Surface(); }

Prognostics create_prognostics(int num_levels) {
  Prognostics p(num_levels);
  for (int mode = 0; mode < AeroConfig::num_modes(); ++mode) {
    p.n_mode_i[mode] = create_column_view(num_levels);
    p.n_mode_c[mode] = create_column_view(num_levels);
    Kokkos::deep_copy(p.n_mode_i[mode], 0.0);
    Kokkos::deep_copy(p.n_mode_c[mode], 0.0);
    for (int spec = 0; spec < AeroConfig::num_aerosol_ids(); ++spec) {
      p.q_aero_i[mode][spec] = create_column_view(num_levels);
      p.q_aero_c[mode][spec] = create_column_view(num_levels);
      Kokkos::deep_copy(p.q_aero_i[mode][spec], 0.0);
      Kokkos::deep_copy(p.q_aero_c[mode][spec], 0.0);
    }
  }
  for (int gas = 0; gas < AeroConfig::num_gas_ids(); ++gas) {
    p.q_gas[gas] = create_column_view(num_levels);
    p.q_gas_avg[gas] = create_column_view(num_levels);
    Kokkos::deep_copy(p.q_gas[gas], 0.0);
    Kokkos::deep_copy(p.q_gas_avg[gas], 0.0);
    for (int mode = 0; mode < AeroConfig::num_modes(); ++mode) {
      p.uptkaer[gas][mode] = create_column_view(num_levels);
      Kokkos::deep_copy(p.uptkaer[gas][mode], 0.0);
    }
  }
  return p;
}

Diagnostics create_diagnostics(int num_levels) {
  Diagnostics d(num_levels);
  for (int mode = 0; mode < AeroConfig::num_modes(); ++mode) {
    d.hygroscopicity[mode] = create_column_view(num_levels);
    Kokkos::deep_copy(d.hygroscopicity[mode], 0.0);
    d.dry_geometric_mean_diameter_i[mode] = create_column_view(num_levels);
    d.dry_geometric_mean_diameter_c[mode] = create_column_view(num_levels);
    d.dry_geometric_mean_diameter_total[mode] = create_column_view(num_levels);
    Kokkos::deep_copy(d.dry_geometric_mean_diameter_i[mode], 0.0);
    Kokkos::deep_copy(d.dry_geometric_mean_diameter_c[mode], 0.0);
    Kokkos::deep_copy(d.dry_geometric_mean_diameter_total[mode], 0.0);
    d.wet_geometric_mean_diameter_i[mode] = create_column_view(num_levels);
    d.wet_geometric_mean_diameter_c[mode] = create_column_view(num_levels);
    Kokkos::deep_copy(d.wet_geometric_mean_diameter_i[mode], 0.0);
    Kokkos::deep_copy(d.wet_geometric_mean_diameter_c[mode], 0.0);

    d.wet_density[mode] = create_column_view(num_levels);
    Kokkos::deep_copy(d.wet_density[mode], 0.0);

    d.uptkrate_h2so4 = create_column_view(num_levels);
    Kokkos::deep_copy(d.uptkrate_h2so4, 0.0);

    d.g0_soa_out = create_column_view(num_levels);
    Kokkos::deep_copy(d.g0_soa_out, 0.0);

    d.is_cloudy = DeviceType::view_1d<bool>("is_cloudy_bool", num_levels);
    Kokkos::deep_copy(d.is_cloudy, false);
    d.num_substeps = DeviceType::view_1d<int>("num_substeps", num_levels);
    Kokkos::deep_copy(d.num_substeps, 0);

    d.icenuc_num_hetfrz = create_column_view(num_levels);
    d.icenuc_num_immfrz = create_column_view(num_levels);
    d.icenuc_num_depnuc = create_column_view(num_levels);
    d.icenuc_num_meydep = create_column_view(num_levels);
    Kokkos::deep_copy(d.icenuc_num_hetfrz, 0.0);
    Kokkos::deep_copy(d.icenuc_num_immfrz, 0.0);
    Kokkos::deep_copy(d.icenuc_num_depnuc, 0.0);
    Kokkos::deep_copy(d.icenuc_num_meydep, 0.0);

    d.num_act_aerosol_ice_nucle_hom = create_column_view(num_levels);
    d.num_act_aerosol_ice_nucle = create_column_view(num_levels);
    Kokkos::deep_copy(d.num_act_aerosol_ice_nucle_hom, 0.0);
    Kokkos::deep_copy(d.num_act_aerosol_ice_nucle, 0.0);

    d.activation_fraction[mode] = create_column_view(num_levels);
    Kokkos::deep_copy(d.activation_fraction[mode], 0.0);

    d.hetfrz_immersion_nucleation_tend = create_column_view(num_levels);
    Kokkos::deep_copy(d.hetfrz_immersion_nucleation_tend, 0.0);
    d.hetfrz_contact_nucleation_tend = create_column_view(num_levels);
    Kokkos::deep_copy(d.hetfrz_contact_nucleation_tend, 0.0);
    d.hetfrz_depostion_nucleation_tend = create_column_view(num_levels);
    Kokkos::deep_copy(d.hetfrz_depostion_nucleation_tend, 0.0);

    d.hetfrz = Diagnostics::View2D(
        "DiagnosticMiscDiags", Diagnostics::number_of_hetfrz_diag, num_levels);
    Kokkos::deep_copy(d.hetfrz, 0.0);
  }
  return d;
}

Tendencies create_tendencies(int num_levels) {
  return create_prognostics(num_levels);
}

} // namespace mam4::testing

//------------------------------------------------------------------------
// EKAT test session initialization and finalization overrides
//------------------------------------------------------------------------
// The following functions are called at the beginning and the end of an
// EKAT test session. When calling EkatCreateUnitTest, you must specify the
// option EXCLUDE_TEST_SESSION which prevents EKAT from using its own
// default implementations.
//------------------------------------------------------------------------

// This implementation of ekat_initialize_test_session is identical to the
// default provided by EKAT.
void ekat_initialize_test_session(int argc, char **argv,
                                  const bool print_config) {
  ekat::initialize_kokkos_session(argc, argv, print_config);
  ekat::enable_fpes(mam4::testing::default_fpes);
}

// This implementation of ekat_finalize_test_session calls
// mam4::testing::finalize() to deallocate all ColumnView pools.
void ekat_finalize_test_session() {
  mam4::testing::finalize();
  ekat::finalize_kokkos_session();
}
