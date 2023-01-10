#include <mam4xx/mam4.hpp>
#include <catch2/catch.hpp>
#include <ekat/logging/ekat_logger.hpp>
#include <ekat/mpi/ekat_comm.hpp>

TEST_CASE("test_constructor", "mam4_aging_process") {
  mam4::AeroConfig mam4_config;
  mam4::AgingProcess process(mam4_config);
  REQUIRE(process.name() == "MAM4 aging");
  REQUIRE(process.aero_config() == mam4_config);
}

TEST_CASE("test_compute_tendencies", "mam4_aging_process") {


  ekat::Comm comm;
  ekat::logger::Logger<> logger("aging unit tests",
                            ekat::logger::LogLevel::debug, comm);
  std::ostringstream ss;


  mam4::AeroConfig mam4_config;
  mam4::AgingProcess process(mam4_config);


    ss << "\n aging compute tendencies";
    logger.debug(ss.str());
    ss.str("");  


}