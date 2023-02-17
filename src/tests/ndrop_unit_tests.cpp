#include "atmosphere_utils.hpp"

#include "mam4xx/aero_modes.hpp"
#include "mam4xx/conversions.hpp"

#include <haero/atmosphere.hpp>
#include <haero/constants.hpp>
#include <haero/floating_point.hpp>
#include <haero/haero.hpp>

#include "mam4xx/conversions.hpp"

#include <catch2/catch.hpp>
#include <ekat/ekat_pack_kokkos.hpp>
#include <ekat/logging/ekat_logger.hpp>
#include <ekat/mpi/ekat_comm.hpp>
#include <mam4xx/mam4.hpp>

// if you need something from the data/ directory
// std::string data_file = MAM4_TEST_DATA_DIR;
// #include <mam4_test_config.hpp>

// using namespace haero;
using namespace mam4;
using namespace mam4::conversions;

TEST_CASE("test_get_aer_num", "mam4_ndrop") {
  ekat::Comm comm;

  ekat::logger::Logger<> logger("ndrop unit tests",
                                ekat::logger::LogLevel::debug, comm);

  int nlev = 1;
  Real pblh = 1000;
  Atmosphere atm(nlev, pblh);

  // initialize a hydrostatically balanced moist air column
  // using constant lapse rate in virtual temperature to manufacture
  // exact solutions.
  //
  // these values correspond to a humid atmosphere with relative humidity
  // values approximately between 32% and 98%
  const Real Tv0 = 300;     // reference virtual temperature [K]
  const Real Gammav = 0.01; // virtual temperature lapse rate [K/m]
  const Real qv0 =
      0.015; // specific humidity at surface [kg h2o / kg moist air]
  const Real qv1 = 7.5e-4; // specific humidity lapse rate [1 / m]
  init_atm_const_tv_lapse_rate(atm, Tv0, Gammav, qv0, qv1);

  mam4::Prognostics progs(nlev);
  mam4::Diagnostics diags(nlev);

  mam4::AeroConfig mam4_config;

  const auto nmodes = mam4::AeroConfig::num_modes();

  Real naerosol[nmodes];

  for (int idx; idx <= nmodes; idx++) {
    for (int k = 0; k < nlev; ++k) {
      // logger.info("diags.dry_geometric_mean_diameter_total[idx](k) = {}",
      // diags.dry_geometric_mean_diameter_total[idx](k));
      // logger.info("progs.n_mode_i[mode_idx](k) + progs.n_mode_c[mode_idx](k)
      // = {}", progs.n_mode_i[idx](k) + progs.n_mode_c[idx](k));

      logger.info("interst = {}", progs.n_mode_i[idx](k));
      logger.info("cloud = {}", progs.n_mode_c[idx](k));
      logger.info("rho = {}", conversions::density_of_ideal_gas(
                                  atm.temperature(k), atm.pressure(k)));

      mam4::get_aer_num(diags, progs, atm, idx, k, naerosol);
      logger.info("naerosol[{}] = {}", idx, naerosol[idx]);
    }
  }
}
