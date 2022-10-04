#include <catch2/catch.hpp>
#include <map>
#include <sstream>
#include "ekat/logging/ekat_logger.hpp"
#include "ekat/mpi/ekat_comm.hpp"
#include "mam4.hpp"
#include "aero_modes.hpp"

using namespace mam4;

TEST_CASE("aero_modes_test", "") {

  ekat::Comm comm;

  ekat::logger::Logger<> logger("nucleation unit tests",
    ekat::logger::LogLevel::debug, comm);

  const bool mode_has_spec[4][7] = {
    {true,  true,  true,  true,  true,  true,  true},
    {true,  false, true,  false, false, true,  true},
    {true,  true,  true,  true,  true,  true,  true},
    {false, true,  false, true,  false, false, true}};

  for (int m=0; m<4; ++m) {
    for (int s=0; s<7; ++s) {
      const int aero_idx = aerosol_index_for_mode(
        static_cast<ModeIndex>(m), static_cast<AeroId>(s));
      logger.debug("m = {} s = {} aero_idx = {}", m, s, aero_idx);
      if (aero_idx >= 0) {
        REQUIRE( mode_has_spec[m][s] );
      }
      else {
        REQUIRE( !mode_has_spec[m][s] );
      }
    }
  }

}
