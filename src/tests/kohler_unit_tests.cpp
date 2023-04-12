// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include "kohler_verification.hpp"
#include <mam4_test_config.hpp>
#include <mam4xx/kohler.hpp>

#include <catch2/catch.hpp>
#include <ekat/logging/ekat_logger.hpp>
#include <ekat/mpi/ekat_comm.hpp>
#include <haero/constants.hpp>
#include <haero/floating_point.hpp>
#include <haero/haero.hpp>
#include <haero/math.hpp>

#include <cmath>
#include <iostream>

using namespace mam4;

struct KohlerSolveTestFtor {
  typedef DeviceType::view_1d<Real> RealView;
  typedef DeviceType::view_1d<int> IntView;

  RealView newton_sol;
  RealView newton_err;
  IntView newton_iterations;
  RealView bisection_sol;
  RealView bisection_err;
  IntView bisection_iterations;
  RealView bracket_sol;
  RealView bracket_err;
  IntView bracket_iterations;
  RealView relative_humidity;
  RealView hygroscopicity;
  RealView dry_radius;
  RealView true_sol;
  Real tol;

  KohlerSolveTestFtor(RealView n_sol, RealView n_err, IntView n_iter,
                      RealView b_sol, RealView b_err, IntView b_iter,
                      RealView br_sol, RealView br_err, IntView br_iter,
                      const RealView rh, const RealView hyg,
                      const RealView rdry, const RealView sol, const Real ctol)
      : newton_sol(n_sol), newton_err(n_err), newton_iterations(n_iter),
        bisection_sol(b_sol), bisection_err(b_err),
        bisection_iterations(b_iter), bracket_sol(br_sol), bracket_err(br_err),
        bracket_iterations(br_iter), relative_humidity(rh), hygroscopicity(hyg),
        dry_radius(rdry), true_sol(sol), tol(ctol) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i) const {
    KohlerSolver<haero::math::NewtonSolver<KohlerPolynomial>> newton_solver(
        relative_humidity(i), hygroscopicity(i), dry_radius(i), tol);
    KohlerSolver<haero::math::BisectionSolver<KohlerPolynomial>>
        bisection_solver(relative_humidity(i), hygroscopicity(i), dry_radius(i),
                         tol);
    KohlerSolver<haero::math::BracketedNewtonSolver<KohlerPolynomial>>
        bracket_solver(relative_humidity(i), hygroscopicity(i), dry_radius(i),
                       tol);

    newton_sol(i) = newton_solver.solve();
    bisection_sol(i) = bisection_solver.solve();
    bracket_sol(i) = bracket_solver.solve();

    newton_err(i) = abs(newton_sol(i) - true_sol(i));
    bisection_err(i) = abs(bisection_sol(i) - true_sol(i));
    bracket_err(i) = abs(bracket_sol(i) - true_sol(i));

    newton_iterations(i) = newton_solver.n_iter;
    bisection_iterations(i) = bisection_solver.n_iter;
    bracket_iterations(i) = bracket_solver.n_iter;
  }
};

TEST_CASE("kohler_physics_functions", "") {
  ekat::Comm comm;
  ekat::logger::Logger<> logger("kohler functions",
                                ekat::logger::LogLevel::debug, comm);

  const Real mam4_surften = haero::Constants::surface_tension_h2o_air_273k;
  const Real mam4_kelvin_a = kelvin_coefficient();

  // minimum temperature for liquid water to -25 C
  const Real min_temp = 248.16;
  // maximum temperature is 75 C (hottest temperature ever recorded is 56.7 C)
  const Real max_temp = 348.16;

  Real max_rel_diff_sigma = 0;
  Real max_rel_diff_kelvin_a = 0;
  const int nn = 100;
  const Real dT = (max_temp - min_temp) / nn;
  for (int i = 0; i <= nn; ++i) {
    const Real T = min_temp + i * dT;
    const Real sigma = surface_tension_water_air(T);
    const Real k_a = kelvin_coefficient(T);
    const Real rel_diff_sigma = std::abs(sigma - mam4_surften) / mam4_surften;
    const Real rel_diff_kelvin_a =
        std::abs(k_a - mam4_kelvin_a) / mam4_kelvin_a;
    if (rel_diff_sigma > max_rel_diff_sigma) {
      max_rel_diff_sigma = rel_diff_sigma;
    }
    if (rel_diff_kelvin_a > max_rel_diff_kelvin_a) {
      max_rel_diff_kelvin_a = rel_diff_kelvin_a;
    }
  }

  logger.info("Accounting for temperature changes causes <= {} relative "
              "difference in surface tension.",
              max_rel_diff_sigma);
  logger.info("Accounting for temperature changes causes <= {} relative "
              "difference in Kelvin droplet coefficient.",
              max_rel_diff_kelvin_a);

  REQUIRE(kelvin_coefficient() * 1e6 ==
          Approx(0.00120746723156361711).epsilon(7e-3));
  REQUIRE(surface_tension_water_air() == Approx(mam4_surften).epsilon(8.5e-5));
}

TEST_CASE("kohler_verificiation", "") {

  ekat::Comm comm;
  ekat::logger::Logger<> logger("kohler verification",
                                ekat::logger::LogLevel::debug, comm);

  // number of tests for each of 3 parameters, total of N**3 tests
  static constexpr int N = 20;
  static constexpr int N3 = N * N * N;
  KohlerVerification verification(N);

  SECTION("polynomial_properties") {
    DeviceType::view_1d<Real> k_of_zero("kohler_poly_zero_input", N3);
    DeviceType::view_1d<Real> k_of_rdry("kohler_poly_rdry_input", N3);
    DeviceType::view_1d<Real> k_of_25rdry("kohler_poly_25rdry_input", N3);

    const auto rh = verification.relative_humidity;
    const auto hyg = verification.hygroscopicity;
    const auto rdry = verification.dry_radius;
    Kokkos::parallel_for(
        "KohlerVerification::test_properties", N3, KOKKOS_LAMBDA(const int i) {
          const Real mam4_default_temperature = Constants::triple_pt_h2o;
          const auto kpoly = KohlerPolynomial(rh(i), hyg(i), rdry(i),
                                              mam4_default_temperature);
          k_of_zero(i) = kpoly(0);
          k_of_rdry(i) = kpoly(rdry(i));
          k_of_25rdry(i) = kpoly(25 * rdry(i));
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

    const Real mam4_kelvin_a = kelvin_coefficient() * 1e6;

    for (int i = 0; i < N3; ++i) {
      REQUIRE(
          FloatingPoint<Real>::equiv(h_k0(i), mam4_kelvin_a * cube(h_rdry(i))));
      REQUIRE(h_krdry(i) > 0);
      REQUIRE(h_k25(i) < 0);
    }
  }

  SECTION("polynomial_roots") {
    const Real conv_tol = 1e-10;

    DeviceType::view_1d<Real> newton_sol("kohler_newton_sol", N3);
    DeviceType::view_1d<Real> newton_err("kohler_newton_err", N3);
    DeviceType::view_1d<int> newton_iterations("kohler_newton_iterations", N3);
    DeviceType::view_1d<Real> bisection_sol("kohler_bisection_sol", N3);
    DeviceType::view_1d<Real> bisection_err("kohler_bisection_err", N3);
    DeviceType::view_1d<int> bisection_iterations("kohler_bisection_iterations",
                                                  N3);
    DeviceType::view_1d<Real> bracket_sol("kohler_bracket_sol", N3);
    DeviceType::view_1d<Real> bracket_err("kohler_bracket_err", N3);
    DeviceType::view_1d<int> bracket_iterations("kohler_bracket_iterations",
                                                N3);

    Kokkos::parallel_for(
        "KohlerVerification::roots", N3,
        KohlerSolveTestFtor(
            newton_sol, newton_err, newton_iterations, bisection_sol,
            bisection_err, bisection_iterations, bracket_sol, bracket_err,
            bracket_iterations, verification.relative_humidity,
            verification.hygroscopicity, verification.dry_radius,
            verification.true_sol, conv_tol));

    Real newton_max_err;
    Real bisection_max_err;
    Real bracket_max_err;
    Kokkos::parallel_reduce(
        N3,
        KOKKOS_LAMBDA(const int i, Real &err) {
          const Real me = newton_err(i);
          err = (me > err ? me : err);
        },
        Kokkos::Max<Real>(newton_max_err));

    Kokkos::parallel_reduce(
        N3,
        KOKKOS_LAMBDA(const int i, Real &err) {
          const Real me = bisection_err(i);
          err = (me > err ? me : err);
        },
        Kokkos::Max<Real>(bisection_max_err));

    Kokkos::parallel_reduce(
        N3,
        KOKKOS_LAMBDA(const int i, Real &err) {
          const Real me = bracket_err(i);
          err = (me > err ? me : err);
        },
        Kokkos::Max<Real>(bracket_max_err));

    int newton_max_iter;
    int bisection_max_iter;
    int bracket_max_iter;
    Kokkos::parallel_reduce(
        N3,
        KOKKOS_LAMBDA(const int i, int &it) {
          it = (newton_iterations(i) > it ? newton_iterations(i) : it);
        },
        Kokkos::Max<int>(newton_max_iter));
    Kokkos::parallel_reduce(
        N3,
        KOKKOS_LAMBDA(const int i, int &it) {
          it = (bisection_iterations(i) > it ? bisection_iterations(i) : it);
        },
        Kokkos::Max<int>(bisection_max_iter));
    Kokkos::parallel_reduce(
        N3,
        KOKKOS_LAMBDA(const int i, int &it) {
          it = (bracket_iterations(i) > it ? bracket_iterations(i) : it);
        },
        Kokkos::Max<int>(bracket_max_iter));

    std::cout << "To generate the verification data with Mathematica, run this "
                 "program:\n\n";

    std::cout << verification.mathematica_verification_program();

    std::cout << "\n\nTo generate the verification data with Matlab, run this "
                 "program:\n\n";

    std::cout << verification.matlab_verification_program();

    logger.info("Newton solve: max err = {}, max_iter = {}", newton_max_err,
                newton_max_iter);
    logger.info("bisection solve: max err = {}, max_iter = {}",
                bisection_max_err, bisection_max_iter);
    logger.info("bracket solve: max err = {}, max_iter = {}", bracket_max_err,
                bracket_max_iter);

    REQUIRE(newton_max_err < 1.5 * conv_tol);
    REQUIRE(bisection_max_err < 5 * conv_tol);
    REQUIRE(bracket_max_err < 1.5 * conv_tol);
  }
}
