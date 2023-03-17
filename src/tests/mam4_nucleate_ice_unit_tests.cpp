// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

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
  // REQUIRE(qs <= 1.0);
  // REQUIRE(qs > 0.0);

  es -= 500;
  qs = mam4::wv_sat_methods::wv_sat_svp_to_qsat(es, p);
  ss << "qs [out]: [ ";
  ss << qs << " ";
  ss << "]";
  logger.debug(ss.str());
  // REQUIRE(qs <= 1.0);
  // REQUIRE(qs > 0.0);
}

TEST_CASE("test_wv_sat_qsat_water", "mam4_nucleate_ice_process") {
  ekat::Comm comm;
  ekat::logger::Logger<> logger("nucleate_ice unit tests",
                                ekat::logger::LogLevel::debug, comm);

  std::ostringstream ss;

  mam4::AeroConfig mam4_config;
  mam4::NucleateIceProcess process(mam4_config);
  Real es, p, t, qs;
  // FIXME: do these numbers make sense?
  es = 101325.0 + 1;
  p = 300.0;
  t = 288;
  qs = 0.8;
  mam4::wv_sat_methods::wv_sat_qsat_water(t, p, es, qs);
  ss << "es [out]: [ ";
  ss << es << " ";
  ss << "]";
  logger.debug(ss.str());
  ss.str("");
  // REQUIRE(es > 273.0);

  p = 700;
  mam4::wv_sat_methods::wv_sat_qsat_water(t, p, es, qs);
  ss << "es [out]: [ ";
  ss << es << " ";
  ss << "]";
  logger.debug(ss.str());
  // REQUIRE(es == p);
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
  // see if the returned threshold temperature is in a reasonable(?) range
  // FIXME: do these numbers make sense?
  // REQUIRE(regm > -100.0);
  // REQUIRE(regm < 100.0);
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
  // REQUIRE(RHw > 0.9529);
  // REQUIRE(RHw < 1.3850);

  temperature = -100.0;
  lnw = 2.0;
  mam4::nucleate_ice::calculate_RHw_hf(temperature, lnw, RHw);
  ss << "RHw [out]: [ ";
  ss << RHw << " ";
  ss << "]";
  logger.debug(ss.str());
  ss.str("");
  // REQUIRE(RHw > 0.9529);
  // REQUIRE(RHw < 1.3850);
}

TEST_CASE("test_compute_tendencies", "mam4_nucleate_ice_process") {
  ekat::Comm comm;

  ekat::logger::Logger<> logger("nucleate_ice unit tests",
                                ekat::logger::LogLevel::debug, comm);
  int nlev = 72;
  Real pblh = 1000;
  Atmosphere atm(nlev, pblh);
  mam4::Prognostics progs(nlev);
  mam4::Diagnostics diags(nlev);
  mam4::Tendencies tends(nlev);

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
  // Now we process multiple columns within a single dіspatch (mc means
  // "multi-column").
  int ncol = 8;
  DeviceType::view_1d<Atmosphere> mc_atm("mc_progs", ncol);
  DeviceType::view_1d<mam4::Prognostics> mc_progs("mc_atm", ncol);
  DeviceType::view_1d<mam4::Diagnostics> mc_diags("mc_diags", ncol);
  DeviceType::view_1d<mam4::Tendencies> mc_tends("mc_tends", ncol);
  int nlev = 72;
  Real pblh = 1000;
  Atmosphere atmosphere(nlev, pblh);
  mam4::Prognostics prognostics(nlev);
  mam4::Diagnostics diagnostics(nlev);
  mam4::Tendencies tendencies(nlev);
  for (int icol = 0; icol < ncol; ++icol) {
    Kokkos::parallel_for(
        "Load multi-column views", 1, KOKKOS_LAMBDA(const int) {
          mc_atm(icol) = atmosphere;
          mc_progs(icol) = prognostics;
          mc_diags(icol) = diagnostics;
          mc_tends(icol) = tendencies;
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
