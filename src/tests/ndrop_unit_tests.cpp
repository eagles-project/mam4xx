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

  // initialize diags
  for (int m = 0; m < 4; ++m) {
    Real dry_vol = 0.0;
    for (int aid = 0; aid < 7; ++aid) {
      const int s = aerosol_index_for_mode(static_cast<ModeIndex>(m),
                                           static_cast<AeroId>(aid));
      if (s >= 0) {
        dry_vol += mass_mixing_ratio / aero_species(s).density;
      }
    }
  }

  // Call mam4xx kernel across all modes, levels
  Kokkos::parallel_for(
      "compute_dry_particle_size", nlev, KOKKOS_LAMBDA(const int i) {
        mode_avg_dry_particle_diam(diags, progs, i);
      });

  // Copy kernel results from device to host
  for (int m = 0; m < 4; ++m) {
    auto h_diam_i =
        Kokkos::create_mirror_view(diags.dry_geometric_mean_diameter_i[m]);
    auto h_diam_c =
        Kokkos::create_mirror_view(diags.dry_geometric_mean_diameter_c[m]);
    auto h_diam_total =
        Kokkos::create_mirror_view(diags.dry_geometric_mean_diameter_total[m]);

    Kokkos::deep_copy(h_diam_i, diags.dry_geometric_mean_diameter_i[m]);
    Kokkos::deep_copy(h_diam_c, diags.dry_geometric_mean_diameter_c[m]);
    Kokkos::deep_copy(h_diam_total, diags.dry_geometric_mean_diameter_total[m]);
  }

  Real naerosol[nmodes];

  // iterate over modes and columns to calculate aer num
  for (int idx = 0; idx < nmodes; idx++) {
    for (int k = 0; k < nlev; ++k) {
      Real vaerosol = conversions::mean_particle_volume_from_diameter(
          diags.dry_geometric_mean_diameter_total[idx](k),
          modes(idx).mean_std_dev);
      Real num2vol_ratio_min =
          1.0 / conversions::mean_particle_volume_from_diameter(
                    modes(idx).min_diameter, modes(idx).mean_std_dev);
      Real num2vol_ratio_max =
          1.0 / conversions::mean_particle_volume_from_diameter(
                    modes(idx).max_diameter, modes(idx).mean_std_dev);
      Real min_bound = vaerosol * num2vol_ratio_max;
      Real max_bound = vaerosol * num2vol_ratio_min;
      Real middle = (progs.n_mode_i[idx](k) + progs.n_mode_c[idx](k)) *
                    conversions::density_of_ideal_gas(atm.temperature(k),
                                                      atm.pressure(k));

      mam4::get_aer_num(diags, progs, atm, idx, k, naerosol);
      logger.info("naerosol[{}] = {}", idx, naerosol[idx]);

      logger.info("min bound = {}", min_bound);
      logger.info("max bound = {}", max_bound);
      logger.info("progs calc = {}", middle);

      REQUIRE(
          FloatingPoint<Real>::in_bounds(naerosol[idx], min_bound, max_bound));
      bool check_calc = FloatingPoint<Real>::equiv(naerosol[idx], min_bound) ||
                        FloatingPoint<Real>::equiv(naerosol[idx], middle) ||
                        FloatingPoint<Real>::equiv(naerosol[idx], max_bound);
      REQUIRE(check_calc);
    }
  }
}
