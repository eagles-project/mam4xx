// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include "testing.hpp"
#include <mam4xx/mam4.hpp>

#include <ekat/ekat_type_traits.hpp>
#include <ekat/logging/ekat_logger.hpp>
#include <ekat/mpi/ekat_comm.hpp>

#include <catch2/catch.hpp>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>

using namespace haero;
using namespace mam4;
TEST_CASE("test_constructor", "mam4_coagulation_process") {
  mam4::AeroConfig mam4_config;
  mam4::CoagulationProcess::ProcessConfig process_config;
  mam4::CoagulationProcess process(mam4_config, process_config);
  REQUIRE(process.name() == "MAM4 Coagulation");
  REQUIRE(process.aero_config() == mam4_config);
}

TEST_CASE("test_aging_pairs", "mam4_aging_pairs") {
  // mam4 coagulation assumes that max_agepair is 1
  mam4::AeroConfig mam4_config;
  REQUIRE(mam4_config.max_agepair() == 1);
}

TEST_CASE("bm0ij_data", "mam4_cagulation_process") {

  // Here we test a few values returned directly from fortran with those
  // as defined in the C++ code.

  const Real threshold_error = std::numeric_limits<float>::epsilon();

  int n1 = 0;
  int n2a = 1;
  int n2n = 2;
  Real bm0ij_f = 0.674432;
  Real bm0ij_c = coagulation::bm0ij_data(n1, n2a, n2n);
  REQUIRE(haero::abs(bm0ij_f - bm0ij_c) < threshold_error);

  n1 = 1;
  n2a = 4;
  n2n = 0;
  bm0ij_f = 0.739575;
  bm0ij_c = coagulation::bm0ij_data(n1, n2a, n2n);
  REQUIRE(haero::abs(bm0ij_f - bm0ij_c) < threshold_error);
}

TEST_CASE("bm3ij_data", "mam4_coagulation_process") {

  const Real threshold_error = std::numeric_limits<float>::epsilon();

  int n1 = 0;
  int n2a = 1;
  int n2n = 2;
  Real bm3i_f = 0.74927;
  Real bm3i_c = coagulation::bm3i_data(n1, n2a, n2n);
  REQUIRE(haero::abs(bm3i_f - bm3i_c) < threshold_error);

  n1 = 1;
  n2a = 4;
  n2n = 0;
  bm3i_f = 0.91886;
  bm3i_c = coagulation::bm3i_data(n1, n2a, n2n);
  REQUIRE(haero::abs(bm3i_f - bm3i_c) < threshold_error);
}

TEST_CASE("intra_coag_rate_for_0th_moment", "mam4_coagulation_process") {

  Real a_const = 0.0;
  Real knc = 0.0;

  Real kngxx = 0.0;
  Real kfmxx = 0.0;
  Real sqdgxx = 0.0;

  Real esxx04 = 0.0;
  Real esxx08 = 0.0;
  Real esxx20 = 0.0;

  Real esxx01 = 0.0;
  Real esxx05 = 0.0;
  Real esxx25 = 0.0;

  int n2x = 1;
  Real qnxx = 0.0;

  coagulation::intramodal_coag_rate_for_0th_moment(
      a_const, knc, kngxx, kfmxx, sqdgxx, esxx04, esxx08, esxx20, esxx01,
      esxx05, esxx25, n2x, qnxx);
}

TEST_CASE("test_compute_tendencies", "mam4_coagulation_process") {

  ekat::Comm comm;
  ekat::logger::Logger<> logger("aging unit tests",
                                ekat::logger::LogLevel::debug, comm);
  int nlev = 72;
  Real pblh = 1000;
  Atmosphere atm = mam4::testing::create_atmosphere(nlev, pblh);
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
