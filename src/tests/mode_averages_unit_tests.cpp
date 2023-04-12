// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include "atmosphere_utils.hpp"
#include "testing.hpp"

#include <mam4xx/aero_config.hpp>
#include <mam4xx/aero_modes.hpp>
#include <mam4xx/conversions.hpp>
#include <mam4xx/mam4_types.hpp>
#include <mam4xx/mode_dry_particle_size.hpp>
#include <mam4xx/mode_hygroscopicity.hpp>
#include <mam4xx/mode_wet_particle_size.hpp>

#include <haero/atmosphere.hpp>
#include <haero/floating_point.hpp>

#include <catch2/catch.hpp>

#include <ekat/logging/ekat_logger.hpp>
#include <ekat/mpi/ekat_comm.hpp>

using namespace mam4;

TEST_CASE("modal_averages", "") {
  ekat::Comm comm;
  ekat::logger::Logger<> logger("modal averages tests",
                                ekat::logger::LogLevel::info, comm);

  /// Use EAMxx default number of levels
  const int nlev = 72;
  /// Initialize prognostics with nonzero mass and number
  /// mixing ratios.
  /// These values are chosen for simplicity; they roughly match
  /// (within an order of magnitude) realistic values.
  Prognostics progs = testing::create_prognostics(nlev);
  Diagnostics diags = testing::create_diagnostics(nlev);

  const Real number_mixing_ratio = 2e7;
  const Real mass_mixing_ratio = 3e-8;
  for (int m = 0; m < 4; ++m) {
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

  /// compute dry particle size for interstitial, cloud, and
  /// total (interstitial + cloud) aerosols.
  /// Here we compute the average sizes separately, then call the kernel
  /// and compare its result to our manually computed values.
  /// We use the same values for interstitial and cloud in this unit test.
  SECTION("dry particle size") {

    /// Separate averages
    Real dry_aero_mean_particle_diam[4];
    Real dry_aero_mean_particle_diam_total[4];
    for (int m = 0; m < 4; ++m) {
      Real dry_vol = 0.0;
      for (int aid = 0; aid < 7; ++aid) {
        const int s = aerosol_index_for_mode(static_cast<ModeIndex>(m),
                                             static_cast<AeroId>(aid));
        if (s >= 0) {
          dry_vol += mass_mixing_ratio / aero_species(s).density;
        }
      }
      const Real mean_vol = dry_vol / number_mixing_ratio;
      dry_aero_mean_particle_diam[m] =
          conversions::mean_particle_diameter_from_volume(
              mean_vol, modes(m).mean_std_dev);
      dry_aero_mean_particle_diam_total[m] =
          conversions::mean_particle_diameter_from_volume(
              2 * mean_vol, modes(m).mean_std_dev);

      logger.info("{} mode has mean particle diameter {}",
                  mode_str(static_cast<ModeIndex>(m)),
                  dry_aero_mean_particle_diam[m]);
    }

    /// Call mam4xx kernel across all modes, levels
    Kokkos::parallel_for(
        "compute_dry_particle_size", nlev, KOKKOS_LAMBDA(const int i) {
          mode_avg_dry_particle_diam(diags, progs, i);
        });

    /// Copy kernel results from device to host and compare
    for (int m = 0; m < 4; ++m) {
      auto h_diam_i =
          Kokkos::create_mirror_view(diags.dry_geometric_mean_diameter_i[m]);
      auto h_diam_c =
          Kokkos::create_mirror_view(diags.dry_geometric_mean_diameter_c[m]);
      auto h_diam_total = Kokkos::create_mirror_view(
          diags.dry_geometric_mean_diameter_total[m]);

      Kokkos::deep_copy(h_diam_i, diags.dry_geometric_mean_diameter_i[m]);
      Kokkos::deep_copy(h_diam_c, diags.dry_geometric_mean_diameter_c[m]);
      Kokkos::deep_copy(h_diam_total,
                        diags.dry_geometric_mean_diameter_total[m]);

      for (int k = 0; k < nlev; ++k) {
        if (!FloatingPoint<Real>::equiv(h_diam_i(k),
                                        dry_aero_mean_particle_diam[m])) {
          logger.debug(
              "h_diam_i({}) = {}, dry_aero_mean_particle_diam[{}] = {}", k,
              h_diam_i(k), m, dry_aero_mean_particle_diam[m]);

          logger.debug("h_diam_c({}) = {}, h_diam_total({}) = {}", k,
                       h_diam_c(k), k, h_diam_total(k));
        }
        if (!FloatingPoint<Real>::equiv(h_diam_c(k),
                                        dry_aero_mean_particle_diam[m])) {
          logger.debug(
              "h_diam_c({}) = {}, dry_aero_mean_particle_diam[{}] = {}", k,
              h_diam_c(k), m, dry_aero_mean_particle_diam[m]);
        }
        if (!FloatingPoint<Real>::equiv(h_diam_total(k),
                                        dry_aero_mean_particle_diam_total[m])) {
          logger.debug("h_diam_total({}) = {}, "
                       "dry_aero_mean_particle_diam_total[{}] = {}",
                       k, h_diam_total(k), m,
                       dry_aero_mean_particle_diam_total[m]);
        }
        REQUIRE(FloatingPoint<Real>::equiv(h_diam_i(k),
                                           dry_aero_mean_particle_diam[m]));
        REQUIRE(FloatingPoint<Real>::equiv(h_diam_c(k),
                                           dry_aero_mean_particle_diam[m]));
        REQUIRE(FloatingPoint<Real>::equiv(
            h_diam_total(k), dry_aero_mean_particle_diam_total[m]));
      }
    }
    logger.info("dry particle size tests complete.");
  } // section (dry particle size)

  SECTION("hygroscopicity") {
    Real hygro[4];
    for (int m = 0; m < 4; ++m) {
      Real dry_vol = 0.0;
      Real hyg = 0.0;
      for (int aid = 0; aid < 7; ++aid) {
        const int s = aerosol_index_for_mode(static_cast<ModeIndex>(m),
                                             static_cast<AeroId>(aid));
        if (s >= 0) {
          dry_vol += mass_mixing_ratio / aero_species(s).density;
          hyg += mass_mixing_ratio * aero_species(s).hygroscopicity /
                 aero_species(s).density;
        }
      }
      hygro[m] = hyg / dry_vol;
      logger.info("{} mode has hygroscopicity {}",
                  mode_str(static_cast<ModeIndex>(m)), hygro[m]);
    }

    Kokkos::parallel_for(
        "compute_hygroscopicity", nlev,
        KOKKOS_LAMBDA(const int i) { mode_hygroscopicity(diags, progs, i); });

    for (int m = 0; m < 4; ++m) {
      auto h_hyg = Kokkos::create_mirror_view(diags.hygroscopicity[m]);
      Kokkos::deep_copy(h_hyg, diags.hygroscopicity[m]);
      for (int k = 0; k < nlev; ++k) {
        if (!FloatingPoint<Real>::equiv(h_hyg(k), hygro[m])) {
          logger.debug("h_hyg({}) = {}, hygro[{}] = {}", k, h_hyg(k), m,
                       hygro[m]);
        }
        REQUIRE(FloatingPoint<Real>::equiv(h_hyg(k), hygro[m]));
      }
    }
    logger.info("hygroscopicity tests complete.");
  } // section (hygroscopicity)

  SECTION("wet particle size") {
    const Real pblh = 0;
    Atmosphere atm = testing::create_atmosphere(nlev, pblh);
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

    const auto w = atm.vapor_mixing_ratio;
    const auto T = atm.temperature;
    const auto P = atm.pressure;
    ColumnView relative_humidity = testing::create_column_view(nlev);
    Kokkos::parallel_for(
        "compute relative humidity", nlev, KOKKOS_LAMBDA(const int k) {
          relative_humidity(k) =
              conversions::relative_humidity_from_vapor_mixing_ratio(w(k), T(k),
                                                                     P(k));
        });

    typedef typename Kokkos::MinMax<Real>::value_type MinMax;
    MinMax rh_mm;
    Kokkos::parallel_reduce(
        nlev,
        KOKKOS_LAMBDA(const int k, MinMax &mm) {
          const auto rhk = relative_humidity(k);
          if (rhk < mm.min_val)
            mm.min_val = rhk;
          if (rhk > mm.max_val)
            mm.max_val = rhk;
        },
        Kokkos::MinMax<Real>(rh_mm));

    logger.info("relative humidity range = [{}, {}]", rh_mm.min_val,
                rh_mm.max_val);

    Kokkos::parallel_for(
        "compute_wet_particle_size", nlev, KOKKOS_LAMBDA(const int i) {
          mode_avg_dry_particle_diam(diags, progs, i);
          mode_hygroscopicity(diags, progs, i);
          mode_avg_wet_particle_diam_water_uptake(diags, atm, i);
        });

    for (int m = 0; m < 4; ++m) {
      auto h_dry_diam =
          Kokkos::create_mirror_view(diags.dry_geometric_mean_diameter_i[m]);
      auto h_wet_diam =
          Kokkos::create_mirror_view(diags.wet_geometric_mean_diameter_i[m]);
      Kokkos::deep_copy(h_dry_diam, diags.dry_geometric_mean_diameter_i[m]);
      Kokkos::deep_copy(h_wet_diam, diags.wet_geometric_mean_diameter_i[m]);

      if (!FloatingPoint<Real>::in_bounds(
              h_dry_diam(0) * 1e3, KohlerPolynomial::dry_radius_min_microns,
              KohlerPolynomial::dry_radius_max_microns)) {

        logger.error("dry particle size out of bounds for mode {}", m);
      }

      for (int k = 0; k < nlev; ++k) {
        logger.debug("m = {} k = {} rdry = {} rwet = {}", m, k, h_dry_diam(k),
                     h_wet_diam(k));

        CHECK(h_wet_diam(k) >= h_dry_diam(k));
      }
    }

  } // section wet particle size
} // test case
