#include "mam4xx/conversions.hpp"

#include <catch2/catch.hpp>

#include <haero/atmosphere.hpp>
#include <haero/constants.hpp>
#include <haero/floating_point.hpp>
#include <haero/haero.hpp>

#include "atmosphere_init.hpp"
#include "mam4xx/conversions.hpp"

#include <ekat/ekat_pack_math.hpp>
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
  const int npacks = PackInfo::num_packs(nlev);
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
    for (int k = 0; k < npacks; ++k) {
      REQUIRE((h_w(k) >= 0).all());
      logger.info("levek {}: T = {} P = {} z = {} dp = {} w = {}", k, h_T(k),
                  h_P(k), h_z(k), h_hdp(k), h_w(k));
    }
  }

  SECTION("relative humidity") {
    ColumnView specific_humidity("specific_humidity", npacks);
    ColumnView relative_humidity_w("relative_humidity_w", npacks);
    ColumnView relative_humidity_q("relative_humidity_q", npacks);

    // compute relative humidity with respect to specific humidity and to
    // mixing ratio
    Kokkos::parallel_for(
        npacks, KOKKOS_LAMBDA(const int k) {
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
        npacks,
        KOKKOS_LAMBDA(const int k, MinMax &mm) {
          const auto rhk = relative_humidity_q(k);
          for (int i = 0; i < haero::HAERO_PACK_SIZE; ++i) {
            if (rhk[i] < mm.min_val)
              mm.min_val = rhk[i];
            if (rhk[i] > mm.max_val)
              mm.max_val = rhk[i];
          }
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

    for (int k = 0; k < npacks; ++k) {
      logger.debug(
          "level {}: T = {} P = {} w = {} q = {} relative humidity = {}", k,
          h_T(k), h_P(k), h_w(k), h_q(k), h_rh_q(k));
      if (!haero::FloatingPoint<PackType>::in_bounds(h_rh_q(k), 0, 1)) {
        logger.debug("\tlevel {}: w = {} wsat = {}", k,
                     vapor_mixing_ratio_from_specific_humidity(h_q(k)),
                     saturation_mixing_ratio_hardy(h_T(k), h_P(k)));
      }
      // check that relative humidities are in bounds
      CHECK(haero::FloatingPoint<PackType>::in_bounds(h_rh_q(k), 0, 1));
      CHECK(haero::FloatingPoint<PackType>::in_bounds(h_rh_w(k), 0, 1));

      // both relative humidities should match
      if (!haero::FloatingPoint<PackType>::equiv(
              h_rh_q(k), h_rh_w(k), std::numeric_limits<float>::epsilon())) {
        logger.debug(
            "rel diff found at level {}: rh_q = {} rh_w = {} rel_diff = {}", k,
            h_rh_q(k), h_rh_w(k), abs(h_rh_q(k) - h_rh_w(k)) / h_rh_q(k));
      }
      REQUIRE(haero::FloatingPoint<PackType>::equiv(
          h_rh_q(k), h_rh_w(k), std::numeric_limits<float>::epsilon()));
    }
  }
}
