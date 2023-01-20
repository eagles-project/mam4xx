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

TEST_CASE("test_constructor", "mam4_aging_process") {
  mam4::AeroConfig mam4_config;
  mam4::AgingProcess process(mam4_config);
  REQUIRE(process.name() == "MAM4 aging");
  REQUIRE(process.aero_config() == mam4_config);
}

TEST_CASE("test_compute_tendencies", "mam4_aging_process") {}

TEST_CASE("test_cond_coag_mass_to_accum", "mam4_aging_process") {}

TEST_CASE("transfer_aged_pcarbon_to_accum", "mam4_aging_process") {
  const int nsrc = static_cast<int>(ModeIndex::PrimaryCarbon);
  const int ndest = static_cast<int>(ModeIndex::Accumulation);

  Real xferfrac_pcage = 0.5;
  Real frac_cond = 0.25;
  Real frac_coag = 0.75;

  std::vector<Real> qaer_cur(AeroConfig::num_modes(), 0.0);
  std::vector<Real> qaer_del_cond(AeroConfig::num_modes(), 0.0);
  std::vector<Real> qaer_del_coag(AeroConfig::num_modes(), 0.0);

  qaer_cur[nsrc] = 0.0;

  aging::transfer_aged_pcarbon_to_accum(
      nsrc, ndest, xferfrac_pcage, frac_cond, frac_coag, qaer_cur.data(),
      qaer_del_cond.data(), qaer_del_coag.data());

  for (int m = 0; m < AeroConfig::num_modes(); ++m) {
    REQUIRE(qaer_cur[m] == 0.0);
    REQUIRE(qaer_del_cond[m] == 0.0);
    REQUIRE(qaer_del_coag[m] == 0.0);
  }

  qaer_cur[nsrc] = 1.0;
  aging::transfer_aged_pcarbon_to_accum(
      nsrc, ndest, xferfrac_pcage, frac_cond, frac_coag, qaer_cur.data(),
      qaer_del_cond.data(), qaer_del_coag.data());

  Real sum_for_conservation = 0.0;
  for (int m = 0; m < AeroConfig::num_modes(); ++m) {
    if (m == nsrc) {
      REQUIRE(qaer_cur[m] == 0.5);
    } else if (m == ndest) {
      REQUIRE(qaer_cur[m] == 0.5);
      REQUIRE(qaer_del_cond[m] == 0.125);
      REQUIRE(qaer_del_coag[m] == 0.375);
    } else {
      REQUIRE(qaer_cur[m] == 0.0);
      REQUIRE(qaer_del_coag[m] == 0.0);
      REQUIRE(qaer_del_cond[m] == 0.0);
    }
    sum_for_conservation += qaer_cur[m] + qaer_del_cond[m] + qaer_del_coag[m];
  }

  // Check for conservation
  REQUIRE(sum_for_conservation == 1.0);
}

TEST_CASE("mam4_pcarbon_aging_1subarea", "mam4_aging_process") {}