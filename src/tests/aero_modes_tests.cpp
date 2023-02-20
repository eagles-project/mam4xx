// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/aero_modes.hpp>
#include <mam4xx/mam4.hpp>

#include <catch2/catch.hpp>
#include <ekat/logging/ekat_logger.hpp>
#include <ekat/mpi/ekat_comm.hpp>

#include <map>
#include <sstream>

using namespace mam4;

TEST_CASE("aero_modes_test", "") {

  ekat::Comm comm;

  ekat::logger::Logger<> logger("nucleation unit tests",
                                ekat::logger::LogLevel::info, comm);

  const bool mode_has_spec[4][7] = {
      {true, true, true, true, true, true, true},
      {true, true, false, false, true, false, true},
      {true, true, true, true, true, true, true},
      {false, false, true, true, false, false, true}};

  for (int m = 0; m < 4; ++m) {
    for (int s = 0; s < 7; ++s) {
      const int aero_idx = aerosol_index_for_mode(static_cast<ModeIndex>(m),
                                                  static_cast<AeroId>(s));
      logger.debug("m = {} s = {} aero_idx = {}", m, s, aero_idx);
      if (aero_idx >= 0) {
        logger.info("{} mode contains aerosol species \"{}\".",
                    mode_str(static_cast<ModeIndex>(m)),
                    aero_id_str(static_cast<AeroId>(s)));
        REQUIRE(mode_has_spec[m][s]);
      } else {
        logger.info("{} mode does not contain aerosol species \"{}\".",
                    mode_str(static_cast<ModeIndex>(m)),
                    aero_id_str(static_cast<AeroId>(s)));
        REQUIRE(!mode_has_spec[m][s]);
      }
    }
  }
}
