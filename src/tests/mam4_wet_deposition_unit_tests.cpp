// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include "testing.hpp"
#include <mam4xx/mam4.hpp>

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

  // TODO - call this from in a Kokkos loop to test device code
  mam4::wetdep::local_precip_production(pdel, source_term, sink_term, lprec,
                                        atm);

  // TODO - generate/use real validation data
  for (int i = 0; i < pver; i++) {
    REQUIRE(pdel[i] == 1.0);
    REQUIRE(source_term[i] == 2.0);
    REQUIRE(sink_term[i] == 1.5);
    REQUIRE(lprec[i] ==
            pdel[i] / Constants::gravity * (source_term[i] - sink_term[i]));
  }

  delete[] pdel;
  delete[] source_term;
  delete[] sink_term;
  delete[] lprec;
}

TEST_CASE("test_calculate_cloudy_volume", "mam4_wet_deposition_process") {
  ekat::Comm comm;
  // TODO - figure out how to print this logging information...
  ekat::logger::Logger<> logger("wet deposition calculate cloudy volume test",
                                ekat::logger::LogLevel::debug, comm);
  int nlev = 72;
  Real pblh = 1000;
  Atmosphere atm = mam4::testing::create_atmosphere(nlev, pblh);

  const int pver = atm.num_levels();

  Real *cld = new Real[pver];
  Real *lprec = new Real[pver];
  Real *cldv = new Real[pver];
  Real *sumppr_all = new Real[pver];

  std::fill_n(cld, pver, 1.0);
  std::fill_n(lprec, pver, 2.0);
  std::fill_n(cldv, pver, 1.5);
  std::fill_n(sumppr_all, pver, 0.0);

  // Pass true to flag
  // TODO - call this from in a Kokkos loop to test device code
  mam4::wetdep::calculate_cloudy_volume(cld, lprec, true, cldv, sumppr_all,
                                        atm);

  // TODO - generate/use real validation data
  for (int i = 0; i < pver; i++) {
    REQUIRE(cld[i] == 1.0);
    REQUIRE(lprec[i] == 2.0);
    // REQUIRE(cldv[i] == 1.5);
    // REQUIRE(sumppr_all[i] == 0.0);
  }

  std::fill_n(cld, pver, 1.0);
  std::fill_n(lprec, pver, 2.0);
  std::fill_n(cldv, pver, 1.5);
  std::fill_n(sumppr_all, pver, 0.0);

  // Pass false to flag
  // TODO - call this from in a Kokkos loop to test device code
  mam4::wetdep::calculate_cloudy_volume(cld, lprec, false, cldv, sumppr_all,
                                        atm);

  // TODO - generate/use real validation data
  for (int i = 0; i < pver; i++) {
    REQUIRE(cld[i] == 1.0);
    REQUIRE(lprec[i] == 2.0);
    // REQUIRE(cldv[i] == 1.5);
    // REQUIRE(sumppr_all[i] == 0.0);
  }

  delete[] cld;
  delete[] lprec;
  delete[] cldv;
  delete[] sumppr_all;
}

TEST_CASE("test_rain_mix_ratio", "mam4_wet_deposition_process") {
  ekat::Comm comm;
  // TODO - figure out how to print this logging information...
  ekat::logger::Logger<> logger("rain mixing ratio test",
                                ekat::logger::LogLevel::debug, comm);
  int nlev = 72;
  Real pblh = 1000;
  Atmosphere atm = mam4::testing::create_atmosphere(nlev, pblh);

  const int pver = atm.num_levels();

  Real *temperature = new Real[pver];
  Real *pmid = new Real[pver];
  Real *sumppr = new Real[pver];
  Real *rain = new Real[pver];

  std::fill_n(temperature, pver, 1.0);
  std::fill_n(pmid, pver, 2.0);
  std::fill_n(sumppr, pver, 1.5);
  std::fill_n(rain, pver, 0.0);

  // TODO - call this from in a Kokkos loop to test device code
  mam4::wetdep::rain_mix_ratio(temperature, pmid, sumppr, rain, atm);

  // TODO - generate/use real validation data
  for (int i = 0; i < pver; i++) {
    REQUIRE(temperature[i] == 1.0);
    REQUIRE(pmid[i] == 2.0);
    REQUIRE(sumppr[i] == 1.5);
    // REQUIRE(atm[i] == 0.0);
  }

  delete[] temperature;
  delete[] pmid;
  delete[] sumppr;
  delete[] rain;
}
