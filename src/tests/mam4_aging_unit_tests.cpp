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

TEST_CASE("test_compute_tendencies", "mam4_aging_process") {




}

TEST_CASE("test_cond_coag_mass_to_accum", "mam4_aging_process"){




}

TEST_CASE("transfer_aged_pcarbon_to_accum", "mam4_aging_process"){




}

TEST_CASE("mam4_pcarbon_aging_1subarea", "mam4_aging_process"){

  ekat::Comm comm;
  ekat::logger::Logger<> logger("aging unit tests",
                            ekat::logger::LogLevel::debug, comm);
  std::ostringstream ss;


  mam4::AeroConfig mam4_config;
  mam4::AgingProcess process(mam4_config);

  const auto naero = mam4::AeroConfig::num_aerosol_ids();
  const auto nmodes = mam4::AeroConfig::num_modes();
  
  Real dgn_a[nmodes];  
  Real qnum_cur[nmodes];
  Real qnum_del_cond[nmodes];
  Real qnum_del_coag[naero][nmodes];
  Real qaer_cur[naero][nmodes];
  Real qaer_del_cond[naero][nmodes];
  Real qaer_del_coag_in[naero][nmodes];


  mam4::aging::mam_pcarbon_aging_1subarea(dgn_a, qnum_cur, qnum_del_cond,
  qnum_del_coag, qaer_cur, qaer_del_cond, qaer_del_coag_in);

  ss << "\n aging compute tendencies";
  logger.debug(ss.str());
  ss.str("");  



}