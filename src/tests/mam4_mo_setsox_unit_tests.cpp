#include "atmosphere_utils.hpp"
#include "testing.hpp"

// #include "mam4xx/aero_modes.hpp"
// #include "mam4xx/conversions.hpp"
// #include <mam4xx/mode_dry_particle_size.hpp>

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
// using namespace mam4::conversions;

// NOTE: other than the nonnegative-ish requirements, this test is basically
// vacuous, since it mostly does the same thing as the function, but I suppose
// it'll let us know if the function changes while not having to worry about
// new/globally-defined constants
TEST_CASE("test_mo_setsox_init", "mam4_mo_setsox_unit_tests") {
  ekat::Comm comm;
  ekat::logger::Logger<> logger("mo_setsox unit tests",
                                ekat::logger::LogLevel::debug, comm);

  Real a = 1.0;
  mam4::mo_setsox::sox_init(a);
  REQUIRE(FloatingPoint<Real>::equiv(a, 1.0));
}
