#include "atmosphere_utils.hpp"

#include "mam4xx/aero_modes.hpp"
#include "mam4xx/conversions.hpp"
#include <mam4xx/mode_dry_particle_size.hpp>

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

  logger.info("starting ndrop test");
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

  const auto T = atm.temperature;
  const auto P = atm.pressure;

  auto h_T = Kokkos::create_mirror_view(T);
  auto h_P = Kokkos::create_mirror_view(P);

  Kokkos::deep_copy(h_T, T);
  Kokkos::deep_copy(h_P, P);

  logger.info("initialized atmosphere");
  logger.info("atm.temp = {}", h_T(0));
  logger.info("atm.pressure = {}", h_P(0));
  mam4::Prognostics progs(nlev);
  mam4::Diagnostics diags(nlev);

  mam4::AeroConfig mam4_config;

  const auto nmodes = mam4::AeroConfig::num_modes();

  // initialize progs
  const Real number_mixing_ratio = 2e7;
  const Real mass_mixing_ratio = 3e-8;
  for (int m = 0; m < nmodes; m++) {
    Kokkos::deep_copy(progs.n_mode_i[m], number_mixing_ratio);
    Kokkos::deep_copy(progs.n_mode_c[m], number_mixing_ratio);
    for (int aid = 0; aid < 7; ++aid) {
      const int s = aerosol_index_for_mode(static_cast<ModeIndex>(m),
                                           static_cast<AeroId>(aid));
      if (s >= 0) {
        auto h_q_view_i = Kokkos::create_mirror_view(progs.q_aero_i[m][s]);
        auto h_q_view_c = Kokkos::create_mirror_view(progs.q_aero_c[m][s]);
        for (int k = 0; k < nlev; ++k) {
          h_q_view_i(k) = mass_mixing_ratio;
          h_q_view_c(k) = mass_mixing_ratio;
        }
        Kokkos::deep_copy(progs.q_aero_i[m][s], h_q_view_i);
        Kokkos::deep_copy(progs.q_aero_c[m][s], h_q_view_c);
      }
    }
  }

  logger.info("initialized prognostics");
  // initialize diags
  for (int m = 0; m < nmodes; m++) {
    Real dry_vol = 0.0;
    for (int aid = 0; aid < 7; ++aid) {
      const int s = aerosol_index_for_mode(static_cast<ModeIndex>(m),
                                           static_cast<AeroId>(aid));
      if (s >= 0) {
        dry_vol += mass_mixing_ratio / aero_species(s).density;
      }
    }
  }

  logger.info("initialized diagnostics");
  // Call mam4xx kernel across all modes, levels
  Kokkos::parallel_for(
      "compute_dry_particle_size", nlev, KOKKOS_LAMBDA(const int i) {
        mode_avg_dry_particle_diam(diags, progs, i);
      });

  logger.info("finished initialization");

  mam4::Real naerosol[nmodes];

  // iterate over modes and columns to calculate aer num
  for (int m = 0; m < nmodes; m++) {
    auto h_diam_total = Kokkos::create_mirror_view(diags.dry_geometric_mean_diameter_total[m]);
    Kokkos::deep_copy(h_diam_total, diags.dry_geometric_mean_diameter_total[m]); 

    Kokkos::parallel_for(
        "get_aer_num", nlev,  KOKKOS_LAMBDA(int i) {
          get_aer_num(diags, progs, atm, m,  i, naerosol);
        });

    for (int k = 0; k < nlev; k++) {
      logger.info("k = {}", k);
      logger.info("h_diam_total({}) = {}", k, h_diam_total(k));
      Real vaerosol = conversions::mean_particle_volume_from_diameter(
          h_diam_total(k),
          modes(m).mean_std_dev);

      logger.info("calcing num2vol ratios");
      Real num2vol_ratio_min =
          1.0 / conversions::mean_particle_volume_from_diameter(
                    modes(m).min_diameter, modes(m).mean_std_dev);
      Real num2vol_ratio_max =
          1.0 / conversions::mean_particle_volume_from_diameter(
                    modes(m).max_diameter, modes(m).mean_std_dev);
      Real min_bound = vaerosol * num2vol_ratio_max;
      Real max_bound = vaerosol * num2vol_ratio_min;

      logger.info("atm.temp = {}", h_T(k));
      logger.info("atm.pressure = {}", h_P(k)); 
      Real middle = (number_mixing_ratio + number_mixing_ratio) *
                    conversions::density_of_ideal_gas(h_T(k), h_P(k));
                    
      logger.info("going into get aer num");
      //mam4::get_aer_num(diags, progs, atm, m, k, naerosol);
      logger.info("naerosol[{}] = {}", m, naerosol[m]);

      logger.info("min bound = {}", min_bound);
      logger.info("max bound = {}", max_bound);
      logger.info("progs calc = {}", middle);

      //REQUIRE(
      //    FloatingPoint<Real>::in_bounds(naerosol[m], min_bound, max_bound));
      //bool check_calc = FloatingPoint<Real>::equiv(naerosol[m], min_bound) ||
      //                  FloatingPoint<Real>::equiv(naerosol[m], middle) ||
      //                  FloatingPoint<Real>::equiv(naerosol[m], max_bound);
      //REQUIRE(check_calc);
    }
  }
}
