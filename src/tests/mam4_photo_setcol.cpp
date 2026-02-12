// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include "atmosphere_utils.hpp"
#include "testing.hpp"
#include <catch2/catch.hpp>
#include <ekat_logger.hpp>
#include <mam4xx/mam4.hpp>

// Precision-dependent tolerance traits
template <typename T> struct PrecisionTolerance;

template <> struct PrecisionTolerance<float> {
  static constexpr float tol = 1e-5f; // Single precision tolerance
};

template <> struct PrecisionTolerance<double> {
  static constexpr double tol = 1e-12; // Double precision tolerance
};

template <typename ViewType1D>
Real compute_l2_norm(const ViewType1D &x, const ViewType1D &y) {
  // Sanity check: views must have the same length
  const auto n = x.extent(0);
  assert(n == y.extent(0) && "x and y must have the same length");

  // Accumulator for sum of squared differences
  Real sum_sq = 0.0;

  // Parallel reduction over i=0..n-1: sum_sq += (x(i)-y(i))²
  Kokkos::parallel_reduce(
      "l2_norm_reduce", Kokkos::RangePolicy<>(0, n),
      KOKKOS_LAMBDA(const int i, Real &local_sum) {
        Real d = x(i) - y(i);
        local_sum += d * d;
      },
      sum_sq);
  Kokkos::fence();

  // Return the square root
  return std::sqrt(sum_sq);
}

// Compute the arithmetic mean of a 1D view
template <typename ViewType1D> Real compute_mean(const ViewType1D &x) {
  const auto n = x.extent(0);
  Real sum = 0.0;

  // Parallel reduction to sum all elements
  Kokkos::parallel_reduce(
      "mean_reduce", Kokkos::RangePolicy<>(0, n),
      KOKKOS_LAMBDA(const int i, Real &local_sum) { local_sum += x(i); }, sum);
  Kokkos::fence();

  // Return average
  return sum / n;
}

// Compute relative L2 norm: ||x - y||₂ / (mean(x) + ε)
template <typename ViewType1D>
Real compute_relative_l2_norm(const ViewType1D &x, const ViewType1D &y) {
  const Real num = compute_l2_norm(x, y);
  const Real den = compute_mean(x);
  const Real eps = 1e-16; // Small value to prevent division by zero
  return num / (den + eps);
}

using namespace mam4;
using namespace haero;

// Test that mo_photo::setcol produces the same results in serial and parallel
TEST_CASE("setcol_serial_vs_parallel", "mo_photo") {

  using View2D = DeviceType::view_2d<Real>;
  using View2DHost = typename HostType::view_2d<Real>;

  // Initialize random number generator for reproducibility
  std::default_random_engine generator(12345);
  std::uniform_real_distribution<double> unif_dist(0.0, 1.0);

  // Logger setup
  ekat::Comm comm;
  ekat::logger::Logger<> logger("nucleate_ice unit tests",
                                ekat::logger::LogLevel::debug, comm);
  std::ostringstream ss;

  // Test configuration
  constexpr Real tol = PrecisionTolerance<Real>::tol;
  constexpr int ncol = 3;          // Number of columns
  constexpr int pver = mam4::nlev; // Number of vertical levels

  // Host views for input (column deltas) and reference output
  View2DHost col_delta_host("col_delta_host", ncol, pver + 1);
  View2DHost col_dens_host_ref("col_dens_host_ref", ncol, pver);

  // Initialize column deltas with random values
  for (int i = 0; i < ncol; ++i) {
    for (int j = 0; j < pver + 1; ++j) {
      col_delta_host(i, j) = unif_dist(generator);
    }
  }

  // Serial reference implementation: compute column densities
  // First level: col_dens[0] = col_delta[0] + 0.5 * col_delta[1]
  for (size_t i = 0; i < ncol; ++i) {
    col_dens_host_ref(i, 0) = col_delta_host(i, 0) + 0.5 * col_delta_host(i, 1);
  }

  // Subsequent levels: col_dens[k] = col_dens[k-1] + 0.5*(col_delta[k] +
  // col_delta[k+1])
  for (int kk = 1; kk < pver; ++kk) {
    int km1 = kk - 1;
    for (size_t i = 0; i < ncol; ++i) {
      col_dens_host_ref(i, kk) =
          col_dens_host_ref(i, km1) +
          0.5 * (col_delta_host(i, km1 + 1) + col_delta_host(i, kk + 1));
    }
  }

  // Copy input data to device
  View2D col_delta("col_delta", ncol, pver + 1);
  Kokkos::deep_copy(col_delta, col_delta_host);

  // Device view for parallel output
  View2D col_dens("col_dens", ncol, pver);
  auto team_policy = ThreadTeamPolicy(ncol, Kokkos::AUTO);

  // Parallel implementation using mo_photo::setcol
  Kokkos::parallel_for(
      team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
        const int icol = team.league_rank(); // Column index
        auto col_dens_icol = ekat::subview(col_dens, icol);
        auto col_delta_icol = ekat::subview(col_delta, icol);
        mo_photo::setcol(team, col_delta_icol, col_dens_icol);
      });

  // Copy results back to host for verification
  auto col_dens_host = Kokkos::create_mirror_view(col_dens);
  Kokkos::deep_copy(col_dens_host, col_dens);

  // Compare serial reference with parallel implementation
  for (int i = 0; i < ncol; ++i) {
    for (int j = 0; j < pver; ++j) {
      const Real diff = std::abs(col_dens_host(i, j) - col_dens_host_ref(i, j));
      if (diff >= tol) {
        std::cout << diff << " " << col_dens_host(i, j) << " "
                  << col_dens_host_ref(i, j) << "\n";
        logger.debug(ss.str());
      }
      REQUIRE(diff <= tol);
    }
  }
}

// Test compute_o3_column_density: compare old implementation vs new refactored
// version
TEST_CASE("compute_o3_column_density", "mo_photo") {
  using View1D = DeviceType::view_1d<Real>;
  using View1DHost = typename HostType::view_1d<Real>;

  auto team_policy = ThreadTeamPolicy(1, Kokkos::AUTO);
  auto team_policy_single = ThreadTeamPolicy(1, 1);
  constexpr int pver = mam4::nlev;

  // Atmospheric profile parameters (humid atmosphere with RH ~32%-98%)
  Real pblh = 1000;         // Planetary boundary layer height [m]
  const Real Tv0 = 300;     // Reference virtual temperature [K]
  const Real Gammav = 0.01; // Virtual temperature lapse rate [K/m]
  const Real qv0 = 0.015;   // Surface specific humidity [kg/kg]
  const Real qv1 = 7.5e-4;  // Specific humidity lapse rate [1/m]
  Atmosphere atm =
      mam4::init_atm_const_tv_lapse_rate(pver, pblh, Tv0, Gammav, qv0, qv1);

  // Gas-phase constituent setup
  constexpr int num_gas_aerosol_constituents = mam4::gas_chemistry::gas_pcnst;
  View1D o3_col_dens_i("o3_col_dens_i", pver); // Old implementation output
  mam4::Prognostics progs = mam4::testing::create_prognostics(pver);

  // Convert molecular weights from g/mol to kg/mol
  Real adv_mass_kg_per_moles[num_gas_aerosol_constituents];
  for (int i = 0; i < num_gas_aerosol_constituents; ++i) {
    adv_mass_kg_per_moles[i] = mam4::gas_chemistry::adv_mass[i] / 1e3;
  }

  // Initialize ozone mass mixing ratio with random values
  const int o3_idx = 0;
  const auto &mmr_o3 = progs.q_gas[o3_idx];
  std::default_random_engine generator(12345);
  std::uniform_real_distribution<double> unif_dist(0.0, 1);
  View1DHost mmr_o3_host("mmr_o3_host", pver);
  for (int j = 0; j < pver; ++j) {
    mmr_o3_host(j) = unif_dist(generator);
  }
  Kokkos::deep_copy(mmr_o3, mmr_o3_host);

  // Reference value for ozone column delta at top of atmosphere (from
  // validation data)
  const Real o3_col_deltas_0 = 1.6e15;

  // ===== OLD IMPLEMENTATION: Manual extraction and accumulation =====
  Kokkos::parallel_for(
      team_policy_single, KOKKOS_LAMBDA(const ThreadTeam &team) {
        constexpr int gas_pcnst = mam4::gas_chemistry::gas_pcnst;
        constexpr int offset_aerosol = mam4::utils::gasses_start_ind();
        constexpr int o3_idx = mam4::gas_chemistry::o3_idx;

        // Array to store column density increments at each level
        Real o3_col_deltas[pver + 1] = {};
        o3_col_deltas[0] = o3_col_deltas_0;

        // Compute column density increments for each level
        Kokkos::parallel_for(
            Kokkos::TeamVectorRange(team, pver), [&](const int k) {
              const Real pdel =
                  atm.hydrostatic_dp(k); // Pressure thickness [Pa]

              // Extract aerosol state variables from prognostics
              Real q[gas_pcnst] = {};
              Real state_q[pcnst] = {};
              mam4::utils::extract_stateq_from_prognostics(progs, atm, state_q,
                                                           k);

              for (int i = offset_aerosol; i < pcnst; ++i) {
                q[i - offset_aerosol] = state_q[i];
              }

              // Convert mass mixing ratios (MMR) to volume mixing ratios (VMR)
              Real vmr[gas_pcnst] = {};
              mam4::microphysics::mmr2vmr(q, adv_mass_kg_per_moles, vmr);

              // Compute the column density increment for this level
              o3_col_deltas[k + 1] =
                  mam4::mo_photo::set_ub_col(vmr[o3_idx], pdel);
            });

        team.team_barrier();

        // Accumulate column density increments into total column densities
        mam4::mo_photo::setcol(team, o3_col_deltas, o3_col_dens_i);
      });
  Kokkos::fence();

  // ===== NEW IMPLEMENTATION: Refactored convenience function =====
  const Real mw_o3 =
      adv_mass_kg_per_moles[0]; // Ozone molecular weight [kg/mol]
  View1D o3_col_dens_i2("o3_col_dens_i2", pver);
  const auto pdel = atm.hydrostatic_dp; // Pressure thickness array [Pa]

  Kokkos::parallel_for(
      "compute_o3_column", team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
        mam4::microphysics::compute_o3_column_density(
            team,
            pdel,            // Pressure thickness array [nlev]
            mmr_o3,          // Ozone mass mixing ratio [nlev]
            o3_col_deltas_0, // Top-of-atmosphere column density [1/cm²]
            mw_o3,           // Ozone molecular weight [kg/mol]
            o3_col_dens_i2   // Output: column density [1/cm²] [nlev]
        );
      });
  Kokkos::fence();

  // Compare old vs new implementation
  const Real error = compute_relative_l2_norm(o3_col_dens_i, o3_col_dens_i2);
  constexpr Real tol = PrecisionTolerance<Real>::tol;
  REQUIRE(error <= tol);
}
