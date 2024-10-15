#include "atmosphere_utils.hpp"
#include "testing.hpp"

#include <haero/floating_point.hpp>
#include <haero/haero.hpp>

#include <catch2/catch.hpp>
#include <ekat/logging/ekat_logger.hpp>
#include <ekat/mpi/ekat_comm.hpp>

#include <mam4xx/aero_model_emissions.hpp>
#include <mam4xx/mam4.hpp>

using namespace mam4;

TEST_CASE("test_init_dust_dmt_vwr", "mam4_aero_emissions_unit_tests") {
  ekat::Comm comm;
  ekat::logger::Logger<> logger(
      "aero_model_emissions unit tests: test_init_dust_dmt_vwr",
      ekat::logger::LogLevel::debug, comm);

  const int dust_nbin = aero_model_emissions::dust_nbin;
  const Real dust_dmt_grd[dust_nbin + 1] = {1.0e-7, 1.0e-6, 1.0e-5};
  Real dust_dmt_vwr[dust_nbin];
  Real dust_dmt_vwr_ref[dust_nbin] = {0.78056703442146215e-6,
                                      0.38983341139985417e-5};

  aero_model_emissions::init_dust_dmt_vwr(dust_dmt_grd, dust_dmt_vwr);

  for (int i = 0; i < dust_nbin; ++i) {
    logger.debug("computed value of dust_dmt_vwr[{}] = {}", i, dust_dmt_vwr[i]);
    logger.debug("reference value of dust_dmt_vwr[{}] = {}", i,
                 dust_dmt_vwr_ref[i]);
  }

  for (int i = 0; i < dust_nbin; ++i) {
    REQUIRE(FloatingPoint<Real>::equiv(dust_dmt_vwr[i], dust_dmt_vwr_ref[i]));
  }
}
