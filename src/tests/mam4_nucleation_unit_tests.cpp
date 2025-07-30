// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include "atmosphere_utils.hpp"
#include "testing.hpp"
#include <mam4xx/mam4.hpp>

#include <catch2/catch.hpp>
#include <ekat_comm.hpp>
#include <ekat_logger.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>

using namespace haero;

TEST_CASE("test_constructor", "mam4_nucleation_process") {
  mam4::AeroConfig mam4_config;
  mam4::NucleationProcess process(mam4_config);
  REQUIRE(process.name() == "MAM4 nucleation");
  REQUIRE(process.aero_config() == mam4_config);
}

TEST_CASE("test_compute_tendencies", "mam4_nucleation_process") {
  ekat::Comm comm;

  ekat::logger::Logger<> logger("nucleation unit tests",
                                ekat::logger::LogLevel::debug, comm);

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

  Surface sfc = mam4::testing::create_surface();
  mam4::Prognostics progs = mam4::testing::create_prognostics(nlev);
  mam4::Diagnostics diags = mam4::testing::create_diagnostics(nlev);
  mam4::Tendencies tends = mam4::testing::create_tendencies(nlev);

  mam4::AeroConfig mam4_config;
  mam4::NucleationProcess process(mam4_config);

  const auto prog_qgas0 = progs.q_gas[0];
  const auto tend_qgas0 = tends.q_gas[0];
  auto h_prog_qgas0 = Kokkos::create_mirror_view(prog_qgas0);
  auto h_tend_qgas0 = Kokkos::create_mirror_view(tend_qgas0);
  Kokkos::deep_copy(h_prog_qgas0, prog_qgas0);
  Kokkos::deep_copy(h_tend_qgas0, tend_qgas0);

  std::ostringstream ss;
  ss << "prog_qgas0 [in]: [ ";
  for (int k = 0; k < nlev; ++k) {
    ss << h_prog_qgas0(k) << " ";
  }
  ss << "]";
  logger.debug(ss.str());
  ss.str("");
  ss << "tend_qgas0 [in]: [ ";
  for (int k = 0; k < nlev; ++k) {
    ss << h_tend_qgas0(k) << " ";
  }
  ss << "]";
  logger.debug(ss.str());
  ss.str("");

  for (int k = 0; k < nlev; ++k) {
    CHECK(!isnan(h_prog_qgas0(k)));
    CHECK(!isnan(h_tend_qgas0(k)));
  }
  // Single-column dispatch.
  auto team_policy = ThreadTeamPolicy(1u, Kokkos::AUTO);
  Real t = 0.0, dt = 30.0;
  Kokkos::parallel_for(
      team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
        process.compute_tendencies(team, t, dt, atm, sfc, progs, diags, tends);
      });
  Kokkos::deep_copy(h_prog_qgas0, prog_qgas0);
  Kokkos::deep_copy(h_tend_qgas0, tend_qgas0);

  ss << "prog_qgas0 [out]: [ ";
  for (int k = 0; k < nlev; ++k) {
    ss << h_prog_qgas0(k) << " ";
  }
  ss << "]";
  logger.debug(ss.str());
  ss.str("");
  ss << "tend_qgas0 [out]: [ ";
  for (int k = 0; k < nlev; ++k) {
    ss << h_tend_qgas0(k) << " ";
  }
  ss << "]";
  logger.debug(ss.str());
  ss.str("");

  for (int k = 0; k < nlev; ++k) {
    CHECK(!isnan(h_prog_qgas0(k)));
    CHECK(!isnan(h_tend_qgas0(k)));
  }
}

TEST_CASE("test_multicol_compute_tendencies", "mam4_nucleation_process") {
  // Now we process multiple columns within a single dispatch (mc means
  // "multi-column").
  int ncol = 8;
  DeviceType::view_1d<Atmosphere> mc_atm("mc_progs", ncol);
  DeviceType::view_1d<Surface> mc_sfc("mc_sfc", ncol);
  DeviceType::view_1d<mam4::Prognostics> mc_progs("mc_atm", ncol);
  DeviceType::view_1d<mam4::Diagnostics> mc_diags("mc_diags", ncol);
  DeviceType::view_1d<mam4::Tendencies> mc_tends("mc_tends", ncol);
  int nlev = 72;
  Real pblh = 1000;
  // these values correspond to a humid atmosphere with relative humidity
  // values approximately between 32% and 98%
  const Real Tv0 = 300;     // reference virtual temperature [K]
  const Real Gammav = 0.01; // virtual temperature lapse rate [K/m]
  const Real qv0 =
      0.015; // specific humidity at surface [kg h2o / kg moist air]
  const Real qv1 = 7.5e-4; // specific humidity lapse rate [1 / m]
  Atmosphere atmosphere =
      mam4::init_atm_const_tv_lapse_rate(nlev, pblh, Tv0, Gammav, qv0, qv1);

  Surface surface = mam4::testing::create_surface();
  mam4::Prognostics prognostics = mam4::testing::create_prognostics(nlev);
  mam4::Diagnostics diagnostics = mam4::testing::create_diagnostics(nlev);
  mam4::Tendencies tendencies = mam4::testing::create_tendencies(nlev);
  for (int icol = 0; icol < ncol; ++icol) {
    Kokkos::parallel_for(
        "Load multi-column views", 1, KOKKOS_LAMBDA(const int) {
          mc_atm(icol) = atmosphere;
          mc_sfc(icol) = surface;
          mc_progs(icol) = prognostics;
          mc_diags(icol) = diagnostics;
          mc_tends(icol) = tendencies;
        });
  }

  mam4::AeroConfig mam4_config;
  mam4::NucleationProcess process(mam4_config);

  // Dispatch over all the above columns.
  auto team_policy = ThreadTeamPolicy(ncol, Kokkos::AUTO);
  Real t = 0.0, dt = 30.0;
  Kokkos::parallel_for(
      team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
        const int icol = team.league_rank();
        process.compute_tendencies(team, t, dt, mc_atm(icol), mc_sfc(icol),
                                   mc_progs(icol), mc_diags(icol),
                                   mc_tends(icol));
      });
}
