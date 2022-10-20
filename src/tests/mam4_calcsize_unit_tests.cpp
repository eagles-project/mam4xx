#include <mam4xx/mam4.hpp>

#include <catch2/catch.hpp>
#include <ekat/ekat_pack_kokkos.hpp>
#include <ekat/logging/ekat_logger.hpp>
#include <ekat/mpi/ekat_comm.hpp>

// #include <cmath>
// #include <iomanip>
// #include <iostream>
// #include <limits>
// #include <sstream>

// if you need something from the data/ directory
// std::string data_file = MAM4_TEST_DATA_DIR;
// #include <mam4_test_config.hpp>

using namespace haero;

TEST_CASE("test_constructor", "mam4_nucleation_process") {
  mam4::AeroConfig mam4_config;
  mam4::CalcSizeProcess process(mam4_config);
  REQUIRE(process.name() == "MAM4 calcsize");
  REQUIRE(process.aero_config() == mam4_config);
}
