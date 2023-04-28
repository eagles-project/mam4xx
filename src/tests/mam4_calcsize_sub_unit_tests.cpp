// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include "testing.hpp"
#include <mam4xx/mam4.hpp>

#include <catch2/catch.hpp>
#include <ekat/ekat_pack_kokkos.hpp>
#include <ekat/logging/ekat_logger.hpp>
#include <ekat/mpi/ekat_comm.hpp>

// if you need something from the data/ directory
// std::string data_file = MAM4_TEST_DATA_DIR;
// #include <mam4_test_config.hpp>

using namespace haero;

TEST_CASE("test_constructor", "mam4_calcsize_sub_process") {
  mam4::AeroConfig mam4_config;
  mam4::CalcSizeSubProcess process(mam4_config);
  REQUIRE(process.name() == "MAM4 calcsize_sub");
  REQUIRE(process.aero_config() == mam4_config);
}

TEST_CASE("test_compute_tendencies", "mam4_calcsize_sub_process") {
  ekat::Comm comm;

  ekat::logger::Logger<> logger("calcsize_sub unit tests",
                                ekat::logger::LogLevel::debug, comm);

  int nlev = 1;
  Real pblh = 1000;
  Atmosphere atm = mam4::testing::create_atmosphere(nlev, pblh);
  mam4::Prognostics progs = mam4::testing::create_prognostics(nlev);
  mam4::Diagnostics diags = mam4::testing::create_diagnostics(nlev);
  mam4::Tendencies tends = mam4::testing::create_tendencies(nlev);

  mam4::AeroConfig mam4_config;
  mam4::CalcSizeSubProcess process(mam4_config);

  const int ncol = 1;
  // Single-column dispatch.
  auto team_policy = ThreadTeamPolicy(ncol, Kokkos::AUTO);
  Real t = 0.0, dt = 30.0;
  Kokkos::parallel_for(
      team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
        process.compute_tendencies(team, t, dt, atm, progs, diags, tends);
      });
}

TEST_CASE("test_set_initial_sz_and_volumes", "mam4_calcsize_sub_process") {
  ekat::Comm comm;
  ekat::logger::Logger<> logger("calcsize_sub unit tests",
                                ekat::logger::LogLevel::debug, comm);

  const int nmodes = mam4::AeroConfig::num_modes();
  const int top_lev = 0; // input
  const int nlev = 1;    // input
  const Real dry_geometric_mean_diameter[nmodes] = {
      0.1100000000e-06, 0.2600000000e-07, 0.2000000000e-05, 0.5000000058e-07};

  ColumnView dgncur_view[nmodes];
  ColumnView v2ncur_view[nmodes];
  ColumnView dryvol_view = testing::create_column_view(nlev);
  for (int imode = 0; imode < nmodes; ++imode) {
    dgncur_view[imode] = testing::create_column_view(nlev);
    v2ncur_view[imode] = testing::create_column_view(nlev);
  }
  const int ncol = 1;
  // Single-column dispatch.
  auto team_policy = ThreadTeamPolicy(ncol, Kokkos::AUTO);
  Kokkos::parallel_for(
      team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
        Real dgncur[nlev][nmodes] = {}; // output length nlev
        Real v2ncur[nlev][nmodes] = {}; // output length nlev
        Real dryvol[nlev] = {};         // output length nlev
        for (int imode = 0; imode < nmodes; ++imode) {
          mam4::calcsize_sub::set_initial_sz_and_volumes(
              top_lev, nlev, imode, dry_geometric_mean_diameter, dgncur, v2ncur,
              dryvol);
          dgncur_view[imode](top_lev) = dgncur[top_lev][imode];
          v2ncur_view[imode](top_lev) = v2ncur[top_lev][imode];
          dryvol_view(top_lev) = dryvol[top_lev];
        }
      });

  const Real dgncur_test[nmodes] = {0.1100000000e-06, 0.2600000000e-07,
                                    0.2000000000e-05, 0.5000000058e-07};
  for (int imode = 0; imode < nmodes; ++imode) {
    auto dgncur_host = Kokkos::create_mirror_view(dgncur_view[imode]);
    Kokkos::deep_copy(dgncur_host, dgncur_view[imode]);
    REQUIRE(dgncur_host(0) == Approx(dgncur_test[imode]));
  }

  const Real v2ncur_test[nmodes] = {0.3031219160e+21, 0.4021279287e+23,
                                    0.5043190877e+17, 0.5654239825e+22};
  for (int imode = 0; imode < nmodes; ++imode) {
    auto v2ncur_host = Kokkos::create_mirror_view(v2ncur_view[imode]);
    Kokkos::deep_copy(v2ncur_host, v2ncur_view[imode]);
    REQUIRE(v2ncur_host(0) == Approx(v2ncur_test[imode]));
  }

  auto dryvol_host = Kokkos::create_mirror_view(dryvol_view);
  Kokkos::deep_copy(dryvol_host, dryvol_view);
  REQUIRE(dryvol_host(0) == 0);
}
