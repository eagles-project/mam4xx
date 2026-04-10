// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/mam4.hpp>

#include <ekat_comm.hpp>
#include <ekat_logger.hpp>
#include <ekat_type_traits.hpp>

#include <catch2/catch.hpp>

using mam4::Real;

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
  Real bm0ij_c = mam4::coagulation::bm0ij_data(n1, n2a, n2n);
  REQUIRE(mam4::abs(bm0ij_f - bm0ij_c) < threshold_error);

  n1 = 1;
  n2a = 4;
  n2n = 0;
  bm0ij_f = 0.739575;
  bm0ij_c = mam4::coagulation::bm0ij_data(n1, n2a, n2n);
  REQUIRE(mam4::abs(bm0ij_f - bm0ij_c) < threshold_error);
}

TEST_CASE("bm3ij_data", "mam4_coagulation_process") {
  const Real threshold_error = std::numeric_limits<float>::epsilon();

  int n1 = 0;
  int n2a = 1;
  int n2n = 2;
  Real bm3i_f = 0.74927;
  Real bm3i_c = mam4::coagulation::bm3i_data(n1, n2a, n2n);
  REQUIRE(mam4::abs(bm3i_f - bm3i_c) < threshold_error);

  n1 = 1;
  n2a = 4;
  n2n = 0;
  bm3i_f = 0.91886;
  bm3i_c = mam4::coagulation::bm3i_data(n1, n2a, n2n);
  REQUIRE(mam4::abs(bm3i_f - bm3i_c) < threshold_error);
}

TEST_CASE("intra_coag_rate_for_0th_moment", "mam4_coagulation_process") {

  Real a_const = 1.0;
  Real knc = 1.0;

  Real kngxx = 1.0;
  Real kfmxx = 1.0;
  Real sqdgxx = 1.0;

  Real esxx04 = 1.0;
  Real esxx08 = 1.0;
  Real esxx20 = 1.0;

  Real esxx01 = 1.0;
  Real esxx05 = 1.0;
  Real esxx25 = 1.0;

  int n2x = 1;
  Real qnxx = 0.0;

  mam4::coagulation::intramodal_coag_rate_for_0th_moment(
      a_const, knc, kngxx, kfmxx, sqdgxx, esxx04, esxx08, esxx20, esxx01,
      esxx05, esxx25, n2x, qnxx);
}
