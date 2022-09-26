#include <cmath>
#include <iostream>

#include "kohler.hpp"
#include "catch2/catch.hpp"
#include "haero/haero.hpp"
#include "haero/math.hpp"
#include "haero/constants.hpp"
#include "haero/floating_point.hpp"
#include "ekat/ekat_pack_math.hpp"
#include "ekat/logging/ekat_logger.hpp"
#include "ekat/mpi/ekat_comm.hpp"

using namespace mam4;

TEST_CASE("kohler_physics_functions", "") {
  ekat::Comm comm;
  ekat::logger::Logger<> logger("kohler functions", ekat::logger::LogLevel::debug, comm);

  const Real mam4_surften = haero::Constants::surface_tension_h2o_air_273k;
  const Real mam4_kelvin_a = kelvin_coefficient<Real>();

  // minimum temperature for liquid water to -25 C
  const Real min_temp = 248.16;
  // maximum temperature is 75 C (hottest temperature ever recorded is 56.7 C)
  const Real max_temp = 348.16;

  Real max_rel_diff_sigma = 0;
  Real max_rel_diff_kelvin_a = 0;
  for (int i=0; i<=100; ++i) {
    const Real T = min_temp + i;
    const Real sigma = surface_tension_water_air(T);
    const Real k_a = kelvin_coefficient(T);
    const Real rel_diff_sigma = std::abs(sigma - mam4_surften)/mam4_surften;
    const Real rel_diff_kelvin_a = std::abs(k_a - mam4_kelvin_a)/mam4_kelvin_a;
    if (rel_diff_sigma > max_rel_diff_sigma) {
      max_rel_diff_sigma = rel_diff_sigma;
    }
    if (rel_diff_kelvin_a > max_rel_diff_kelvin_a) {
      max_rel_diff_kelvin_a = rel_diff_kelvin_a;
    }
  }

  logger.info("Accounting for temperature changes causes <= {} relative difference in surface tension.",
  max_rel_diff_sigma);
  logger.info("Accounting for temperature changes causes <= {} relative difference in Kelvin droplet coefficient.",
  max_rel_diff_kelvin_a);
  std::cout << max_rel_diff_kelvin_a << "\n";

  logger.debug("surface tension: abs(default value - mam4) = {}",
    std::abs(surface_tension_water_air<Real>() - mam4_surften)/mam4_surften);

  REQUIRE(surface_tension_water_air<Real>() == Approx(mam4_surften).epsilon(8.5e-5));

}

TEST_CASE("kohler_polynomial_properties", "") {

}

TEST_CASE("kohler_verificiation", "") {
  // number of tests for each of 3 parameters, total of N**3 tests
  static constexpr int N = 20;


}
