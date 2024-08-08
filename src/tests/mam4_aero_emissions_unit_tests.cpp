#include "atmosphere_utils.hpp"
#include "testing.hpp"

#include <haero/floating_point.hpp>
#include <haero/haero.hpp>

#include <catch2/catch.hpp>
#include <ekat/logging/ekat_logger.hpp>
#include <ekat/mpi/ekat_comm.hpp>
#include <mam4xx/mam4.hpp>

using namespace mam4;

TEST_CASE("test_aero_emissions_create_struct",
          "mam4_aero_emissions_unit_tests") {
  ekat::Comm comm;
  ekat::logger::Logger<> logger(
      "mo_aero_emissions unit tests: test_aero_emissions_create_struct",
      ekat::logger::LogLevel::debug, comm);
  logger.debug("");

  logger.debug("***TODO***");
}
