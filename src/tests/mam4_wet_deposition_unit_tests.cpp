// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/mam4.hpp>
#include "testing.hpp"

#include <haero/constants.hpp>

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

TEST_CASE("test_constructor", "mam4_wet_deposition_process") {
  ekat::Comm comm;
  ekat::logger::Logger<> logger("wet deposition constructor test",
                                ekat::logger::LogLevel::debug, comm);
  mam4::AeroConfig mam4_config;
  mam4::WetDepositionProcess::ProcessConfig process_config;
  mam4::WetDepositionProcess process(mam4_config, process_config);
  REQUIRE(process.name() == "MAM4 Wet Deposition");
  REQUIRE(process.aero_config() == mam4_config);
}

// While this isn't a high level function/part of the namespace, we should still test it
TEST_CASE("test_local_precip_production", "mam4_wet_deposition_process") {
  ekat::Comm comm;
  // TODO - figure out how to print this logging information...
  ekat::logger::Logger<> logger("wet deposition local precip production test",
                                ekat::logger::LogLevel::debug, comm);
  int nlev = 72;
  Real pblh = 1000;
  Atmosphere atm = mam4::testing::create_atmosphere(nlev, pblh);

  const int pver = atm.num_levels();

  Real *pdel = new Real[pver];
  Real *source_term = new Real[pver]; 
  Real *sink_term = new Real[pver];
  Real *lprec = new Real[pver];

  std::fill_n(pdel, pver, 1.0);
  std::fill_n(source_term, pver, 2.0);
  std::fill_n(sink_term, pver, 1.5);
  std::fill_n(lprec, pver, 0.0);

  mam4::wetdep::local_precip_production(pdel, source_term, sink_term, lprec, atm);

  for (int i = 0; i < pver; i++)
  {
    REQUIRE(pdel[i] == 1.0);
    REQUIRE(source_term[i] == 2.0);
    REQUIRE(sink_term[i] == 1.5);
    REQUIRE(lprec[i] == pdel[i] / Constants::gravity * (source_term[i] - sink_term[i]));
  }
}
