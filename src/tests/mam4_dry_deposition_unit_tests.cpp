// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include "mam4xx/aero_modes.hpp"
#include "testing.hpp"
#include <mam4xx/mam4.hpp>

#include <haero/constants.hpp>

#include <ekat_type_traits.hpp>
#include <ekat_logger.hpp>
#include <ekat_comm.hpp>

#include <catch2/catch.hpp>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>

using namespace haero;
using namespace mam4;

Real tol = 1e-8;

TEST_CASE("test_constructor", "mam4_dry_deposition_process") {
  ekat::Comm comm;
  ekat::logger::Logger<> logger("dry deposition constructor test",
                                ekat::logger::LogLevel::debug, comm);
  mam4::AeroConfig mam4_config;
  mam4::DryDepositionProcess::ProcessConfig process_config;
  mam4::DryDepositionProcess process(mam4_config, process_config);
  REQUIRE(process.name() == "MAM4 Dry Deposition");
  REQUIRE(process.aero_config() == mam4_config);
}

TEST_CASE("test_calcram", "mam4_dry_deposition_process") {
  ekat::Comm comm;
  ekat::logger::Logger<> logger("dry deposition calcram test",
                                ekat::logger::LogLevel::debug, comm);

  const Real icefrac = 0.6;
  const Real ocnfrac = 0.10000000000000000e+01;
  const Real ram1_in = 0;
  const Real tair = 0.28487570148166026e+03;
  const Real pmid = 0.10006244767430093e+06;
  const Real pdel = 0.30135299181891605e+03;
  const Real fv_in = 0;
  typedef typename Kokkos::MinMax<Real>::value_type Calcram;
  Calcram cal_cram;
  Kokkos::parallel_reduce(
      "test_local_precip_production", 1,
      KOKKOS_LAMBDA(const int, Calcram &vlc) {
        const Real landfrac = 0.1;
        const Real obklen = -0.20723257141035126e+03;
        const Real ustar = 0.39900396673305327e+00;
        Real ram1_out = 0, fv_out = 0;
        mam4::drydep::calcram(landfrac, icefrac, ocnfrac, obklen, ustar, tair,
                              pmid, pdel, ram1_in, fv_in, ram1_out, fv_out);
        vlc.min_val = ram1_out;
        vlc.max_val = fv_out;
      },
      Kokkos::MinMax<Real>(cal_cram));

  REQUIRE(cal_cram.min_val == Approx(0.0));
  REQUIRE(cal_cram.max_val == Approx(1.0e-12));

  Kokkos::parallel_reduce(
      "test_local_precip_production", 1,
      KOKKOS_LAMBDA(const int, Calcram &vlc) {
        const Real landfrac = 0.0;
        const Real obklen = 0.20723257141035126e+03;
        const Real ustar = 0.39900396673305327e+00;
        Real ram1_out = 0, fv_out = 0;
        mam4::drydep::calcram(landfrac, icefrac, ocnfrac, obklen, ustar, tair,
                              pmid, pdel, ram1_in, fv_in, ram1_out, fv_out);
        vlc.min_val = ram1_out;
        vlc.max_val = fv_out;
      },
      Kokkos::MinMax<Real>(cal_cram));
  REQUIRE(cal_cram.min_val == Approx(37.800534979));
  REQUIRE(cal_cram.max_val == Approx(0.399003966733));

  Kokkos::parallel_reduce(
      "test_local_precip_production", 1,
      KOKKOS_LAMBDA(const int, Calcram &vlc) {
        const Real landfrac = 0.0;
        const Real obklen = -0.20723257141035126e+03;
        const Real ustar = 0;
        Real ram1_out = 0, fv_out = 0;
        mam4::drydep::calcram(landfrac, icefrac, ocnfrac, obklen, ustar, tair,
                              pmid, pdel, ram1_in, fv_in, ram1_out, fv_out);
        vlc.min_val = ram1_out;
        vlc.max_val = fv_out;
      },
      Kokkos::MinMax<Real>(cal_cram));
  REQUIRE(cal_cram.min_val == Approx(0.0));
  REQUIRE(cal_cram.max_val == Approx(1.0e-12));
}
