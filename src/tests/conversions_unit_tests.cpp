// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include "testing.hpp"

#include <mam4xx/aero_modes.hpp>
#include <mam4xx/conversions.hpp>

#include <catch2/catch.hpp>

#include <haero/atmosphere.hpp>
#include <haero/constants.hpp>
#include <haero/floating_point.hpp>
#include <haero/haero.hpp>

#include "atmosphere_utils.hpp"
#include "mam4xx/conversions.hpp"

#include <ekat/logging/ekat_logger.hpp>
#include <ekat/mpi/ekat_comm.hpp>

#include <cmath>
#include <sstream>

using namespace mam4;
using namespace mam4::conversions;

TEST_CASE("conversions", "") {
  ekat::Comm comm;
  ekat::logger::Logger<> logger("conversions unit tests",
                                ekat::logger::LogLevel::info, comm);

  const int nlev = 72;
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

  const auto T = atm.temperature;
  const auto P = atm.pressure;
  const auto w = atm.vapor_mixing_ratio;

  auto h_w = Kokkos::create_mirror_view(w);
  auto h_T = Kokkos::create_mirror_view(T);
  auto h_P = Kokkos::create_mirror_view(P);
  auto h_z = Kokkos::create_mirror_view(atm.height);
  auto h_hdp = Kokkos::create_mirror_view(atm.hydrostatic_dp);

  Kokkos::deep_copy(h_w, w);
  Kokkos::deep_copy(h_T, T);
  Kokkos::deep_copy(h_P, P);
  Kokkos::deep_copy(h_z, atm.height);
  Kokkos::deep_copy(h_hdp, atm.hydrostatic_dp);

  auto const unit_pressure = h_P(0);
  auto const unit_temp = h_T(0);

  SECTION("atm. init") {
    logger.info("SECTION atm. init");

    // write atm. column data to log for comparison with e3sm
    for (int k = 0; k < nlev; ++k) {
      REQUIRE(h_w(k) >= 0);
      logger.info("levek {}: T = {} P = {} z = {} dp = {} w = {}", k, h_T(k),
                  h_P(k), h_z(k), h_hdp(k), h_w(k));
    }
  }

  SECTION("air density") {
    // total air density -> dry air density
    logger.info("SECTION air density");
    auto const rho = density_of_ideal_gas(unit_temp, unit_pressure);
    auto const dry_air_density =
        dry_air_density_from_total_air_density(rho, qv0);
    auto const vapor_density = vapor_from_total_mass_density(rho, qv0);
    logger.info("unit_temp = {}, unit_pressure = {}, rho = {}, qv0 = {}",
                unit_temp, unit_pressure, rho, qv0);

    auto const check_dry_air_density = rho * (1 - qv0);
    auto const check_vapor_density = rho * qv0;

    logger.info("dry_air_density = {}, check = {}", dry_air_density,
                check_dry_air_density);
    logger.info("vapor_density = {}, check = {}", vapor_density,
                check_vapor_density);

    REQUIRE(FloatingPoint<Real>::equiv(dry_air_density, check_dry_air_density));
    REQUIRE(FloatingPoint<Real>::equiv(vapor_density, check_vapor_density));
  }

  SECTION("mixing ratios") {
    // mass mixing ratio (mmr) <-> number_conc
    logger.info("SECTION mixing ratios");
    auto const rho = density_of_ideal_gas(unit_temp, unit_pressure);
    auto const mmr = 1e-8;
    auto const num_conc =
        number_conc_from_mmr(mmr, haero::Constants::molec_weight_nacl, rho);
    auto const mmr0 = mmr_from_number_conc(
        num_conc, haero::Constants::molec_weight_nacl, rho);

    logger.info("unit_temp = {}, unit_pressure = {}, rho = {}", unit_temp,
                unit_pressure, rho);
    logger.info("molec_weight_nacl= {}", haero::Constants::molec_weight_nacl);
    logger.info("mixing ratio = {}, num_conc = {}, mmr0 = {}", mmr, num_conc,
                mmr0);
    REQUIRE(FloatingPoint<Real>::equiv(mmr, mmr0));

    // mass mixing ratio (mmr) <-> molar mixing ratio (vmr)
    auto const vmr = vmr_from_mmr(mmr0, haero::Constants::molec_weight_nacl);
    auto const mmr1 = mmr_from_vmr(vmr, haero::Constants::molec_weight_nacl);
    logger.info("mmr0 = {}, vmr = {}, mmr1 = {}", mmr0, vmr, mmr1);
    REQUIRE(FloatingPoint<Real>::equiv(mmr1, mmr0));
  }

  SECTION("temperature") {
    // temperature <-> virtual temperature
    logger.info("SECTION temperature");
    int c = 257;
    auto tol = c * std::numeric_limits<Real>::epsilon(); //~5.7e-14
    auto const temp = temperature_from_virtual_temperature(Tv0, qv0);
    auto const vtemp = virtual_temperature_from_temperature(temp, qv0);
    auto const P0 = 100000; // Pa
    auto const rho_dry = density_of_ideal_gas(temp, P0);
    auto const rho_wet = density_of_ideal_gas(vtemp, P0);
    logger.info("tol = {}", tol);
    logger.info("qv0 init = {}, Tv0 init = {}, calc temp = {}, calc vtemp = {}",
                qv0, Tv0, temp, vtemp);
    logger.info("rho_dry = {}, rho_wet = {}", rho_dry, rho_wet);
    REQUIRE(rho_dry > rho_wet);
    REQUIRE(FloatingPoint<Real>::equiv(Tv0, vtemp, tol));
  }

  SECTION("specfic humidity") {
    // vapor mixing ratio <-> specific humidity
    logger.info("SECTION specific humidity");
    auto const vmr = vapor_mixing_ratio_from_specific_humidity(qv0);
    auto const sh = specific_humidity_from_vapor_mixing_ratio(vmr);
    logger.info("qv0 initial = {}, calc vapor mixing ratio = {}, calc specific "
                "humidity = {}",
                qv0, vmr, sh);
    REQUIRE(FloatingPoint<Real>::equiv(qv0, sh));
  }

  SECTION("vapor saturation pressure") {
    // vapor saturation pressure
    logger.info("SECTION vapor saturation pressure");
    auto const magnus_ew = vapor_saturation_pressure_magnus_ew(unit_temp);
    auto const magnus =
        vapor_saturation_pressure_magnus(unit_temp, unit_pressure);
    auto const hardy = vapor_saturation_pressure_hardy(unit_temp);
    logger.info("unit_temp = {}, unit_pressure = {}", unit_temp, unit_pressure);

    auto const temp_celsius = unit_temp - Constants::freezing_pt_h2o;
    auto const check_magnus_ew =
        6.1094 * exp((17.625 * temp_celsius) / (234.04 + temp_celsius));
    logger.info("magnus ew = {}, check magnus ew = {}", magnus_ew,
                check_magnus_ew * 100);

    auto const check_magnus =
        1.00071 * exp(0.000000045 * unit_pressure) * check_magnus_ew;
    logger.info("magnus = {}, check magnus = {}", magnus, check_magnus * 100);
    logger.info("hardy = {}", hardy);

    REQUIRE(FloatingPoint<Real>::equiv(magnus_ew, check_magnus_ew * 100));
    REQUIRE(FloatingPoint<Real>::equiv(magnus, check_magnus * 100));
    bool check_hardy = FloatingPoint<Real>::equiv(hardy, 0.36647176529292896) ||
                       FloatingPoint<Real>::equiv(hardy, 0.36647177);
    REQUIRE(check_hardy);
  }

  SECTION("saturation mixing ratio") {
    // saturation mixing ratio
    logger.info("SECTION saturation mixing ratio");
    auto const hardy = saturation_mixing_ratio_hardy(unit_temp, unit_pressure);
    logger.info("unit_temp = {}, unit_pressure = {}", unit_temp, unit_pressure);

    auto const eps_h2o =
        Constants::molec_weight_h2o / Constants::molec_weight_dry_air;
    auto const check_hardy =
        (eps_h2o * vapor_saturation_pressure_hardy(unit_temp)) /
        (unit_pressure - vapor_saturation_pressure_hardy(unit_temp));

    logger.info("hardy = {}, check hardy = {}", hardy, check_hardy);
    REQUIRE(FloatingPoint<Real>::equiv(hardy, check_hardy));
  }

  SECTION("humidity and vapor mixing ratio") {
    // relative humidity <-> vapor mixing ratio <-> specific humidity
    logger.info("SECTION humidity and vapor mixing ratio");
    auto const rel_humidity0 =
        relative_humidity_from_specific_humidity(qv0, unit_temp, unit_pressure);
    auto const vmr = vapor_mixing_ratio_from_relative_humidity(
        rel_humidity0, unit_pressure, unit_temp);
    auto const rel_humidity1 = relative_humidity_from_vapor_mixing_ratio(
        vmr, unit_temp, unit_pressure);
    logger.info("qv0 = {}, unit_temp = {}, unit_pressure = {}", qv0, unit_temp,
                unit_pressure);
    logger.info("calc rel humidity from sh = {}, calc vmr = {}, calc rel "
                "humidity from vmr = {}",
                rel_humidity0, vmr, rel_humidity1);
    REQUIRE(FloatingPoint<Real>::equiv(rel_humidity0, rel_humidity1));
  }

  SECTION("mean particle") {
    // mean particle diameter <-> mean particle volume
    logger.info("SECTION mean particle");
    auto const accum_diam = mam4_accum_nom_diameter_m;
    auto const volume =
        mean_particle_volume_from_diameter(accum_diam, mam4_accum_mead_std_dev);
    auto const diameter =
        mean_particle_diameter_from_volume(volume, mam4_accum_mead_std_dev);
    logger.info("mam4_accum_diam = {}, mam4_accum_std_dev = {}, calc volume = "
                "{}, calc diameter = {}",
                accum_diam, mam4_accum_mead_std_dev, volume, diameter);
    REQUIRE(FloatingPoint<Real>::equiv(diameter, accum_diam));
  }

  SECTION("relative humidity") {
    logger.info("SECTION relative humidity");
    ColumnView specific_humidity = testing::create_column_view(nlev);
    ColumnView relative_humidity_w = testing::create_column_view(nlev);
    ColumnView relative_humidity_q = testing::create_column_view(nlev);

    // compute relative humidity with respect to specific humidity and to
    // mixing ratio
    Kokkos::parallel_for(
        nlev, KOKKOS_LAMBDA(const int k) {
          const auto q = specific_humidity_from_vapor_mixing_ratio(w(k));
          specific_humidity(k) = q;
          relative_humidity_q(k) =
              relative_humidity_from_specific_humidity(q, T(k), P(k));
          relative_humidity_w(k) =
              relative_humidity_from_vapor_mixing_ratio(w(k), T(k), P(k));
        });

    typedef typename Kokkos::MinMax<Real>::value_type MinMax;
    MinMax rh_mm;
    Kokkos::parallel_reduce(
        nlev,
        KOKKOS_LAMBDA(const int k, MinMax &mm) {
          const auto rhk = relative_humidity_q(k);
          if (rhk < mm.min_val)
            mm.min_val = rhk;
          if (rhk > mm.max_val)
            mm.max_val = rhk;
        },
        Kokkos::MinMax<Real>(rh_mm));

    logger.info("relative humidity range = [{}, {}]", rh_mm.min_val,
                rh_mm.max_val);

    auto h_q = Kokkos::create_mirror_view(specific_humidity);
    auto h_rh_q = Kokkos::create_mirror_view(relative_humidity_q);
    auto h_rh_w = Kokkos::create_mirror_view(relative_humidity_w);

    Kokkos::deep_copy(h_q, specific_humidity);
    Kokkos::deep_copy(h_rh_q, relative_humidity_q);
    Kokkos::deep_copy(h_rh_w, relative_humidity_w);

    for (int k = 0; k < nlev; ++k) {
      logger.debug(
          "level {}: T = {} P = {} w = {} q = {} relative humidity = {}", k,
          h_T(k), h_P(k), h_w(k), h_q(k), h_rh_q(k));
      if (!haero::FloatingPoint<Real>::in_bounds(h_rh_q(k), 0, 1)) {
        logger.error("\tlevel {}: w = {} wsat = {}", k,
                     vapor_mixing_ratio_from_specific_humidity(h_q(k)),
                     saturation_mixing_ratio_hardy(h_T(k), h_P(k)));
      }
      // check that relative humidities are in bounds
      CHECK(haero::FloatingPoint<Real>::in_bounds(h_rh_q(k), 0, 1));
      CHECK(haero::FloatingPoint<Real>::in_bounds(h_rh_w(k), 0, 1));

      // both relative humidities should match
      const Real tol = 2 * std::numeric_limits<float>::epsilon();
      if (!haero::FloatingPoint<Real>::rel(h_rh_q(k), h_rh_w(k), tol)) {
        logger.error("rel diff found at level {}: rh_q = {} rh_w = {} rel_diff "
                     "= {} tol = {}",
                     k, h_rh_q(k), h_rh_w(k),
                     abs(h_rh_q(k) - h_rh_w(k)) / h_rh_q(k), tol);
      }
      REQUIRE(haero::FloatingPoint<Real>::rel(h_rh_q(k), h_rh_w(k), tol));
    }
  }
}
