// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/mam4.hpp>
#include <mam4xx/utils.hpp>

#include <catch2/catch.hpp>
#include <ekat_comm.hpp>
#include <ekat_logger.hpp>
#include <ekat_pack_kokkos.hpp>

// if you need something from the data/ directory
// std::string data_file = MAM4_TEST_DATA_DIR;
// #include <mam4_test_config.hpp>

using mam4::Real;

TEST_CASE("test_compute_before_growth_dryvol_and_num", "mam4_rename_process") {
  // AeroConfig mam4_config;
  mam4::rename::compute_before_growth_dryvol_and_num();
  REQUIRE(1 == 1);
}

TEST_CASE("test_total_interstitial_and_cloudborne", "mam4_rename_process") {
  // AeroConfig mam4_config;
  Real outvar = mam4::rename::total_interstitial_and_cloudborne();
  REQUIRE(outvar == outvar);
}

TEST_CASE("test_compute_tail_fraction", "mam4_rename_process") {
  // AeroConfig mam4_config;
  // test to see if it runs
  Real log_dia_tail_fac = 1.5;
  Real tail_fraction = 0.0;
  mam4::rename::compute_tail_fraction(1.0e-3, 3.0e-3, 2.0, log_dia_tail_fac,
                                      tail_fraction);
  CHECK(!mam4::isnan(tail_fraction));
}
