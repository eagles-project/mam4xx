// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include "atmosphere_utils.hpp"
#include "testing.hpp"
#include <catch2/catch.hpp>
#include <ekat_logger.hpp>
#include <mam4xx/mam4.hpp>

template <typename T> struct PrecisionTolerance;

template <> struct PrecisionTolerance<float> {
  static constexpr float tol = 1e-5f; // Single precision
};

template <> struct PrecisionTolerance<double> {
  static constexpr double tol = 1e-12; // Double precision
};

template <typename ViewType1D>
Real compute_l2_norm(const ViewType1D &x, const ViewType1D &y) {
  // sanity check
  const auto n = x.extent(0);
  assert(n == y.extent(0) && "x and y must have the same length");

  // accumulator for sum of squared diffs
  Real sum_sq = 0.0;

  // parallel reduction over i=0..n-1: sum_sq += (x(i)-y(i))^2
  Kokkos::parallel_reduce(
      "l2_norm_reduce", Kokkos::RangePolicy<>(0, n),
      KOKKOS_LAMBDA(const int i, Real &local_sum) {
        Real d = x(i) - y(i);
        local_sum += d * d;
      },
      sum_sq);
  Kokkos::fence();

  // take square root on host
  return std::sqrt(sum_sq);
}

using namespace mam4;
using namespace haero;

TEST_CASE("setcol_serial_vs_parallel", "mo_photo") {

  using View2D = DeviceType::view_2d<Real>;
  using View2DHost = typename HostType::view_2d<Real>;
  // Initialize random number generator using std::default_random_engine
  std::default_random_engine generator(12345); // Seed for reproducibility
  std::uniform_real_distribution<double> unif_dist(0.0, 1.0); // Uniform
  ekat::Comm comm;
  ekat::logger::Logger<> logger("nucleate_ice unit tests",
                                ekat::logger::LogLevel::debug, comm);
  std::ostringstream ss;

  // initialization
  constexpr Real tol = PrecisionTolerance<Real>::tol;
  constexpr int ncol = 3;
  constexpr int pver = mam4::nlev;

  View2DHost col_delta_host("col_delta_host", ncol, pver + 1);
  View2DHost col_dens_host_ref("col_dens_host_ref", ncol, pver);

  for (int i = 0; i < ncol; ++i) {
    for (int j = 0; j < pver + 1; ++j) {
      col_delta_host(i, j) = unif_dist(generator); // Generate random number
    }
  }

  // serial implementation
  for (size_t i = 0; i < ncol; ++i) {
    col_dens_host_ref(i, 0) = col_delta_host(i, 0) + 0.5 * col_delta_host(i, 1);
  }

  for (int kk = 1; kk < pver; ++kk) {
    int km1 = kk - 1;
    for (size_t i = 0; i < ncol; ++i) {
      col_dens_host_ref(i, kk) =
          col_dens_host_ref(i, km1) +
          0.5 * (col_delta_host(i, km1 + 1) + col_delta_host(i, kk + 1));
    }
  }

  View2D col_delta("col_delta", ncol, pver + 1);
  Kokkos::deep_copy(col_delta, col_delta_host);

  View2D col_dens("col_dens", ncol, pver);
  auto team_policy = ThreadTeamPolicy(ncol, Kokkos::AUTO);

  Kokkos::parallel_for(
      team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
        const int icol = team.league_rank(); // column index
        auto col_dens_icol = ekat::subview(col_dens, icol);
        auto col_delta_icol = ekat::subview(col_delta, icol);
        mo_photo::setcol(team, col_delta_icol, col_dens_icol);
      });
  auto col_dens_host = Kokkos::create_mirror_view(col_dens);
  Kokkos::deep_copy(col_dens_host, col_dens);

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

TEST_CASE("compute_o3_column_density", "mo_photo") {
  using View1D = DeviceType::view_1d<Real>;
  using View1DHost = typename HostType::view_1d<Real>;

  auto team_policy = ThreadTeamPolicy(1, Kokkos::AUTO);
  int nlev = 72;
  Real pblh = 1000;
  // these values correspond to a humid atmosphere with relative humidity
  // values approximately between 32% and 98%
  const Real Tv0 = 300;     // reference virtual temperature [K]
  const Real Gammav = 0.01; // virtual temperature lapse rate [K/m]
  const Real qv0 =
      0.015; // specific humidity at surface [kg h2o / kg moist air]
  const Real qv1 = 7.5e-4; // specific humidity lapse rate [1 / m]
  Atmosphere atm =
      mam4::init_atm_const_tv_lapse_rate(nlev, pblh, Tv0, Gammav, qv0, qv1);

  constexpr int num_gas_aerosol_constituents = mam4::gas_chemistry::gas_pcnst;
  View1D o3_col_dens_i("o3_col_dens_i", nlev);
  mam4::Prognostics progs = mam4::testing::create_prognostics(nlev);
  Real adv_mass_kg_per_moles[num_gas_aerosol_constituents];
  for (int i = 0; i < num_gas_aerosol_constituents; ++i) {
    adv_mass_kg_per_moles[i] = mam4::gas_chemistry::adv_mass[i] / 1e3;
  }

  const int o3_idx = 0;
  const auto &mmr_o3 = progs.q_gas[o3_idx];
  std::default_random_engine generator(12345); // Seed for reproducibility
  std::uniform_real_distribution<double> unif_dist(0.0, 1.0); // Uniform
  View1DHost mmr_o3_host("mmr_o3_host", nlev);
  for (int j = 0; j < nlev + 1; ++j) {
    mmr_o3_host(j) = unif_dist(generator); // Generate random number
  }
  Kokkos::deep_copy(mmr_o3, mmr_o3_host);

  Kokkos::parallel_for(
      team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
        mam4::microphysics::compute_o3_column_density(
            team, atm, progs,      // in
            adv_mass_kg_per_moles, // in
            o3_col_dens_i);        // out
      });
  Kokkos::fence();

  const Real o3_col_deltas_0 = 0.0;            // example scaling
  const Real mw_o3 = adv_mass_kg_per_moles[0]; // molecular weight of ozone
  View1D o3_col_dens_i2("o3_col_dens_i2", nlev);
  const auto pdel = atm.hydrostatic_dp;
  std::cout << "adv_mass_kg_per_moles[0]" << mw_o3 << "\n";

  // one team, with enough threads to cover 'nlev' if needed
  Kokkos::parallel_for(
      "compute_o3_column", team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
        mam4::microphysics::compute_o3_column_density(
            team,
            pdel,            // pressure-thickness array [nlev]
            mmr_o3,          // ozone mass‐mixing ratio [nlev]
            o3_col_deltas_0, // constant factor
            mw_o3,           // ozone molecular weight
            o3_col_dens_i2   // output column‐density [nlev]
        );
      });

  const Real error = compute_l2_norm(o3_col_dens_i, o3_col_dens_i2);
  constexpr Real tol = PrecisionTolerance<Real>::tol;
  REQUIRE(error <= tol);
}
