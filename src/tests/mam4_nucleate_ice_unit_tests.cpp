// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include "testing.hpp"
#include <mam4xx/mam4.hpp>
#include <mam4xx/utils.hpp>
// #include <mam4xx/wv_sat_methods.hpp>

#include <catch2/catch.hpp>
#include <ekat/ekat_pack_kokkos.hpp>
#include <ekat/logging/ekat_logger.hpp>
#include <ekat/mpi/ekat_comm.hpp>

// if you need something from the data/ directory
// std::string data_file = MAM4_TEST_DATA_DIR;
// #include <mam4_test_config.hpp>

using namespace haero;

TEST_CASE("test_constructor", "mam4_nucleate_ice_process") {
  mam4::AeroConfig mam4_config;
  mam4::NucleateIceProcess process(mam4_config);
  REQUIRE(process.name() == "MAM4 nucleate_ice");
  REQUIRE(process.aero_config() == mam4_config);
}

TEST_CASE("test_wv_sat_svp_to_qsat", "mam4_nucleate_ice_process") {
  ekat::Comm comm;
  ekat::logger::Logger<> logger("nucleate_ice unit tests",
                                ekat::logger::LogLevel::debug, comm);

  std::ostringstream ss;

  mam4::AeroConfig mam4_config;
  mam4::NucleateIceProcess process(mam4_config);
  Real es, p;
  // FIXME: do these numbers make sense?
  es = 101325.0 + 1;
  p = 101325.0;
  Real qs = mam4::wv_sat_methods::wv_sat_svp_to_qsat(es, p);
  ss << "qs [out]: [ ";
  ss << qs << " ";
  ss << "]";
  logger.debug(ss.str());
  ss.str("");
  // function requires this is bounded between 0 and 1
  REQUIRE(qs > 0.0);
  REQUIRE(qs <= 1.0);

  es -= 500;
  qs = mam4::wv_sat_methods::wv_sat_svp_to_qsat(es, p);
  ss << "qs [out]: [ ";
  ss << qs << " ";
  ss << "]";
  logger.debug(ss.str());
  // function requires this is bounded between 0 and 1
  REQUIRE(qs > 0.0);
  REQUIRE(qs <= 1.0);
}

TEST_CASE("test_wv_sat_qsat_water", "mam4_nucleate_ice_process") {
  ekat::Comm comm;
  ekat::logger::Logger<> logger("nucleate_ice unit tests",
                                ekat::logger::LogLevel::debug, comm);

  std::ostringstream ss;

  // NOTE: qs output comes directly from wv_sat_svp_to_qsat() (above), so we
  // won't test those values here

  mam4::AeroConfig mam4_config;
  mam4::NucleateIceProcess process(mam4_config);
  Real es, p, qs;
  Real t[4] = {273, 298, 323, 373};
  // for temperature \in [273, 373]K (~[0, 100]C) es is \in [615, 1.009e5]
  // and final comparison is min(es, p),
  // so here we essentially test GoffGratch_svp_water().
  // es is immediately calculated by svp_water() and qs by wv_sat_svp_to_qsat(),
  // so no need to initialize.
  // we make p artificially high to test es calculation
  p = 2.0e5;
  for (int i = 0; i < 4; ++i) {
    mam4::wv_sat_methods::wv_sat_qsat_water(t[i], p, es, qs);
    ss << "temperature = " << t[i];
    logger.debug(ss.str());
    ss.str("");
    ss << "es [out]: [ ";
    ss << es << " ";
    ss << "]";
    logger.debug(ss.str());
    ss.str("");
    REQUIRE(es > 200.0);
    REQUIRE(es < 1.01e5);
  }

  // make p artificially low to test min(es, p)
  p = 10;
  for (int i = 0; i < 4; ++i) {
    mam4::wv_sat_methods::wv_sat_qsat_water(t[i], p, es, qs);
    ss << "temperature = " << t[i];
    logger.debug(ss.str());
    ss.str("");
    ss << "es [out]: [ ";
    ss << es << " ";
    ss << "]";
    logger.debug(ss.str());
    // for p > es (pressure greater than boiling point of water at 1 atm) es is
    // set to p
    REQUIRE(es == p);
  }
}

TEST_CASE("test_GoffGratch_svp_ice", "mam4_nucleate_ice_process") {
  ekat::Comm comm;
  ekat::logger::Logger<> logger("nucleate_ice unit tests",
                                ekat::logger::LogLevel::debug, comm);

  std::ostringstream ss;

  mam4::AeroConfig mam4_config;
  mam4::NucleateIceProcess process(mam4_config);
  Real t[4] = {173, 198, 223, 273};
  Real es;
  // for temperature \in [173, 273]K (~[-100, 0]C) es is \in [1.0e-3, 604]
  for (int i = 0; i < 4; ++i) {
    es = mam4::wv_sat_methods::GoffGratch_svp_ice(t[i]);
    ss << "temperature = " << t[i];
    logger.debug(ss.str());
    ss.str("");
    ss << "es [out]: [ ";
    ss << es << " ";
    ss << "]";
    logger.debug(ss.str());
    ss.str("");
    REQUIRE(es > 1.0e-3);
    REQUIRE(es < 604.0);
  }
}

TEST_CASE("test_calculate_regm_nucleati", "mam4_nucleate_ice_process") {
  ekat::Comm comm;
  ekat::logger::Logger<> logger("nucleate_ice unit tests",
                                ekat::logger::LogLevel::debug, comm);

  std::ostringstream ss;

  mam4::AeroConfig mam4_config;
  mam4::NucleateIceProcess process(mam4_config);
  Real w_vlc, Na, regm;
  // FIXME: do these numbers make sense?
  w_vlc = 0.2;
  Na = 58;
  mam4::nucleate_ice::calculate_regm_nucleati(w_vlc, Na, regm);
  ss << "regm [out]: [ ";
  ss << regm << " ";
  ss << "]";
  logger.debug(ss.str());
  ss.str("");
  // regm is a threshold temperature [C], so giving it a wide range of values
  // that seem reasonable for the atmosphere
  REQUIRE(regm > -200.0);
  REQUIRE(regm < 50.0);
}

TEST_CASE("test_calculate_RHw_hf", "mam4_nucleate_ice_process") {
  ekat::Comm comm;
  ekat::logger::Logger<> logger("nucleate_ice unit tests",
                                ekat::logger::LogLevel::debug, comm);

  std::ostringstream ss;

  mam4::AeroConfig mam4_config;
  mam4::NucleateIceProcess process(mam4_config);
  Real temperature, lnw, RHw;
  // FIXME: do these numbers make sense?
  temperature = -40.0;
  lnw = 1.0;
  mam4::nucleate_ice::calculate_RHw_hf(temperature, lnw, RHw);
  ss << "RHw [out]: [ ";
  ss << RHw << " ";
  ss << "]";
  logger.debug(ss.str());
  ss.str("");
  // these bounds come from the rough approximations in the source code
  REQUIRE(RHw > 0.9);
  REQUIRE(RHw < 1.6);

  temperature = -100.0;
  lnw = 2.0;
  mam4::nucleate_ice::calculate_RHw_hf(temperature, lnw, RHw);
  ss << "RHw [out]: [ ";
  ss << RHw << " ";
  ss << "]";
  logger.debug(ss.str());
  ss.str("");
  REQUIRE(RHw > 0.9);
  REQUIRE(RHw < 1.6);
}

TEST_CASE("test_compute_tendencies", "mam4_nucleate_ice_process") {
  ekat::Comm comm;

  ekat::logger::Logger<> logger("nucleate_ice unit tests",
                                ekat::logger::LogLevel::debug, comm);
  int nlev = 72;
  Real pblh = 1000;
  Atmosphere atm = mam4::testing::create_atmosphere(nlev, pblh);
  mam4::Prognostics progs = mam4::testing::create_prognostics(nlev);
  mam4::Diagnostics diags = mam4::testing::create_diagnostics(nlev);
  mam4::Tendencies tends = mam4::testing::create_tendencies(nlev);

  mam4::AeroConfig mam4_config;
  mam4::NucleateIceProcess process(mam4_config);

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
        process.compute_tendencies(team, t, dt, atm, progs, diags, tends);
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

TEST_CASE("test_multicol_compute_tendencies", "mam4_nucleateIce_process") {
  // Now we process multiple columns within a single dispatch (mc means
  // "multi-column").
  int ncol = 8;
  DeviceType::view_1d<Atmosphere> mc_atm("mc_progs", ncol);
  DeviceType::view_1d<mam4::Prognostics> mc_progs("mc_atm", ncol);
  DeviceType::view_1d<mam4::Diagnostics> mc_diags("mc_diags", ncol);
  DeviceType::view_1d<mam4::Tendencies> mc_tends("mc_tends", ncol);
  int nlev = 72;
  Real pblh = 1000;
  Atmosphere atm = mam4::testing::create_atmosphere(nlev, pblh);
  mam4::Prognostics progs = mam4::testing::create_prognostics(nlev);
  mam4::Diagnostics diags = mam4::testing::create_diagnostics(nlev);
  mam4::Tendencies tends = mam4::testing::create_tendencies(nlev);
  for (int icol = 0; icol < ncol; ++icol) {
    Kokkos::parallel_for(
        "Load multi-column views", 1, KOKKOS_LAMBDA(const int) {
          mc_atm(icol) = atm;
          mc_progs(icol) = progs;
          mc_diags(icol) = diags;
          mc_tends(icol) = tends;
        });
  }

  mam4::AeroConfig mam4_config;
  mam4::NucleateIceProcess process(mam4_config);

  // Dispatch over all the above columns.
  auto team_policy = ThreadTeamPolicy(ncol, Kokkos::AUTO);
  Real t = 0.0, dt = 30.0;
  Kokkos::parallel_for(
      team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
        const int icol = team.league_rank();
        process.compute_tendencies(team, t, dt, mc_atm(icol), mc_progs(icol),
                                   mc_diags(icol), mc_tends(icol));
      });
}
