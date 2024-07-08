#include "atmosphere_utils.hpp"
#include "testing.hpp"

// #include "mam4xx/aero_modes.hpp"
// #include "mam4xx/conversions.hpp"
// #include <mam4xx/mode_dry_particle_size.hpp>
// #include <mam4xx/aero_config.hpp>

// #include <haero/constants.hpp>
#include <haero/floating_point.hpp>
#include <haero/haero.hpp>

// #include "mam4xx/conversions.hpp"

#include <catch2/catch.hpp>
#include <ekat/logging/ekat_logger.hpp>
#include <ekat/mpi/ekat_comm.hpp>
#include <mam4xx/mam4.hpp>

// using namespace haero;
using namespace mam4;
// // using namespace mam4::conversions;
// const int nmodes = AeroConfig::num_modes();
// const int loffset = 9;
// const mam4::mo_setsox::Config setsox_config_;

TEST_CASE("test_srf_emissions_create_struct", "mam4_srf_emissions_unit_tests") {
  ekat::Comm comm;
  ekat::logger::Logger<> logger(
      "mo_srf_emissions unit tests: test_srf_emissions_create_struct",
      ekat::logger::LogLevel::debug, comm);
  logger.debug("");



  logger.debug("***TESTING***");

}
