#include "mam4xx/conversions.hpp"
#include "mam4xx/aero_modes.hpp"

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

  const auto T = atm.temperature;
  const auto P = atm.pressure;
  const auto w = atm.vapor_mixing_ratio;

  auto h_w = Kokkos::create_mirror_view(w);
  auto h_T = Kokkos::create_mirror_view(T);
  auto h_P = Kokkos::create_mirror_view(P);
  auto h_z = Kokkos::create_mirror_view(atm.height);
  auto h_hdp = Kokkos::create_mirror_view(atm.hydrostatic_dp);

  SECTION("atm. init") {
    Kokkos::deep_copy(h_w, w);
    Kokkos::deep_copy(h_T, T);
    Kokkos::deep_copy(h_P, P);
    Kokkos::deep_copy(h_z, atm.height);
    Kokkos::deep_copy(h_hdp, atm.hydrostatic_dp);

    // write atm. column data to log for comparison with e3sm
    for (int k = 0; k < nlev; ++k) {
      REQUIRE(h_w(k) >= 0);
      logger.info("levek {}: T = {} P = {} z = {} dp = {} w = {}", k, h_T(k),
                  h_P(k), h_z(k), h_hdp(k), h_w(k));
    }
  }

    // mass mixing ratio (mmr) <-> number_conc
    //auto const mmr = mmr_from_number_conc(num_conc, Constants::molec_weight_h2o, )

    // mass mixing ratio (mmr) <-> molar mixing ratio (vmr)

  SECTION("temperature") {
    // temperature <-> virtual temperature
    int c = 257;
    auto tol = c * std::numeric_limits<Real>::epsilon(); //~5.7e-14
    auto const temp = temperature_from_virtual_temperature(Tv0, qv0);
    auto const vtemp = virtual_temperature_from_temperature(temp, qv0);
    logger.info("tol = {}", tol);
    logger.info("qv0 init = {}, Tv0 init = {}, calc temp = {}, calc vtemp = {}", qv0, Tv0, temp, vtemp);
    REQUIRE(FloatingPoint<Real>::equiv(Tv0, vtemp, tol));  
  }  
    // total mass density -> dry mass density
  SECTION("density") {
    auto const dmd = dry_from_total_mass_density(mam4_density_dst, qv0); 
    logger.info("dmd = {}", dmd);
  }
  SECTION("ratio") {
    // vapor mixing ratio <-> specific humidity
    auto const vmr = vapor_mixing_ratio_from_specific_humidity(qv0);
    auto const sh = specific_humidity_from_vapor_mixing_ratio(vmr);
    logger.info("qv0 initial = {}, calc vapor mixing ratio = {}, calc specific humidity = {}", qv0, vmr, sh);
    REQUIRE(FloatingPoint<Real>::equiv(qv0, sh));  
  }
   // vapor saturation pressure

   // saturation mixing ratio

   // relative humidity <-> vapor mixing ratio
  SECTION("mean particle") { 
   // mean particle diameter <-> mean particle volume
    PackType accum_diam(0);
    accum_diam = mam4_accum_nom_diameter_m;
    auto const volume = mean_particle_volume_from_diameter(accum_diam, mam4_accum_mead_std_dev);
    auto const diameter = mean_particle_diameter_from_volume(volume, mam4_accum_mead_std_dev);
    logger.info("mam4_accum_diam = {}, mam4_accum_std_dev = {}, calc volume = {}, calc diameter = {}", accum_diam, mam4_accum_mead_std_dev, volume, diameter);
    REQUIRE(FloatingPoint<PackType>::equiv(diameter, accum_diam));  

  }

  SECTION("relative humidity") {
    ColumnView specific_humidity("specific_humidity", nlev);
    ColumnView relative_humidity_w("relative_humidity_w", nlev);
    ColumnView relative_humidity_q("relative_humidity_q", nlev);

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
