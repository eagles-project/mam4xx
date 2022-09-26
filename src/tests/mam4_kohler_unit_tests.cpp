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
  const int nn = 100;
  const Real dT = (max_temp - min_temp)/nn;
  for (int i=0; i<=nn; ++i) {
    const Real T = min_temp + i*dT;
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

  REQUIRE(kelvin_coefficient<Real>()*1e6 == Approx(0.00120746723156361711).epsilon(7e-3));
  REQUIRE(surface_tension_water_air<Real>() == Approx(mam4_surften).epsilon(8.5e-5));

}

TEST_CASE("kohler_verificiation", "") {

  ekat::Comm comm;
  ekat::logger::Logger<> logger("kohler verification", ekat::logger::LogLevel::debug, comm);

  // number of tests for each of 3 parameters, total of N**3 tests
  static constexpr int N = 20;
  KohlerVerification verification(N);

  SECTION("polynomial_properties") {
    DeviceType::view_1d<PackType> k_of_zero("kohler_poly_zero_input", haero::cube(N));
    DeviceType::view_1d<PackType> k_of_rdry("kohler_poly_rdry_input", haero::cube(N));
    DeviceType::view_1d<PackType> k_of_25rdry("kohler_poly_25rdry_input", haero::cube(N));

    const auto rh = verification.relative_humidity;
    const auto hyg = verification.hygroscopicity;
    const auto rdry  = verification.dry_radius;
    Kokkos::parallel_for("KohlerVerification::test_properties",
      haero::cube(N),
      KOKKOS_LAMBDA (const int i) {
        const auto kpoly = KohlerPolynomial<PackType>(rh(i), hyg(i), rdry(i));
        k_of_zero(i) = kpoly(PackType(0));
        k_of_rdry(i) = kpoly(rdry(i));
        k_of_25rdry(i) = kpoly(25*rdry(i));
      });
    auto h_k0 = Kokkos::create_mirror_view(k_of_zero);
    auto h_krdry = Kokkos::create_mirror_view(k_of_rdry);
    auto h_k25 = Kokkos::create_mirror_view(k_of_25rdry);
    auto h_rh = Kokkos::create_mirror_view(rh);
    auto h_hyg = Kokkos::create_mirror_view(hyg);
    auto h_rdry = Kokkos::create_mirror_view(rdry);
    Kokkos::deep_copy(h_k0, k_of_zero);
    Kokkos::deep_copy(h_krdry, k_of_rdry);
    Kokkos::deep_copy(h_k25, k_of_25rdry);
    Kokkos::deep_copy(h_rh, rh);
    Kokkos::deep_copy(h_hyg, hyg);
    Kokkos::deep_copy(h_rdry, rdry);

    const Real mam4_kelvin_a = kelvin_coefficient<Real>() * 1e6;

    for (int i=0; i<PackInfo::num_packs(verification.n_trials); ++i) {

      REQUIRE(FloatingPoint<PackType>::equiv(
        h_k0(i),  mam4_kelvin_a * cube(h_rdry(i))));

      REQUIRE( (h_krdry(i) > 0).all() );

      REQUIRE( (h_k25(i) < 0).all());
    }
  }

//   SECTION("polynomial_roots") {
//     DeviceType::view_1d<PackType> newton_sol("kohler_newton_sol", haero::cube(N));
//     DeviceType::view_1d<PackType> newton_err("kohler_newton_err", haero::cube(N));
//     DeviceType::view_1d<PackType> newton_iterations("kohler_newton_iterations");
//     DeviceType::view_1d<PackType> bisection_sol("kohler_bisection_sol", haero::cube(N));
//     DeviceType::view_1d<PackType> bisection_err("kohler_bisection_err", haero::cube(N));
//     DeviceType::view_1d<PackType> bisection_iterations("kohler_bisection_iterations");
//     DeviceType::view_1d<PackType> bracket_sol("kohler_bracket_sol", haero::cube(N));
//     DeviceType::view_1d<PackType> bracket_err("kohler_bracket_err", haero::cube(N));
//     DeviceType::view_1d<PackType> bracket_iterations("kohler_bracket_iterations");
//
//   }

}
