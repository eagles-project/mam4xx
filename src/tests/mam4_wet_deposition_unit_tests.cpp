// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

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

TEST_CASE("test_constructor", "mam4_wet_deposition_process") {
  mam4::AeroConfig mam4_config;
  mam4::WetDepositionProcess::ProcessConfig process_config;
  mam4::WetDepositionProcess process(mam4_config, process_config);
  REQUIRE(process.name() == "MAM4 Wet Deposition");
  REQUIRE(process.aero_config() == mam4_config);
}
