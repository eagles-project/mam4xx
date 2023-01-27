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

TEST_CASE("intermodal_coag_rate_for_0th_moment", "mam4_coagulation_process"){}

TEST_CASE("intermodal_coag_rate_for_3rd_moment", "mam4_coagulation_process") {}

TEST_CASE("intra_coag_rate_for_0th_moment", "mam4_coagulation_process") {

  Real one = 0.0;
  Real two = 0.0;
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
      one, two, a_const, knc, kngxx, kfmxx, sqdgxx, esxx04, esxx08, esxx20,
      esxx01, esxx05, esxx25, n2x, qnxx);
}

TEST_CASE("test_compute_tendencies", "mam4_coagulation_process") {

  // Kokkos::parallel_for(
  //     team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
  //       process.compute_tendencies(team, t, dt, atm, progs, diags, tends);
  //     });
}
