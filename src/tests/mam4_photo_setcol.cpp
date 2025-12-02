// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include "testing.hpp"
#include <catch2/catch.hpp>
#include <ekat_logger.hpp>
#include <mam4xx/mam4.hpp>

template <typename T> struct PrecisionTolerance;

template <> struct PrecisionTolerance<float> {
  static constexpr float tol = 1e-5f; // Single precision
};

template <> struct PrecisionTolerance<double> {
  static constexpr double tol = 1e-12; // Double precision
};

using namespace mam4;
using namespace haero;

TEST_CASE("setcol_serial_vs_parallel", "mo_photo") {

  using View2D = DeviceType::view_2d<Real>;
  using View2DHost = typename HostType::view_2d<Real>;
  // Initialize random number generator using std::default_random_engine
  std::default_random_engine generator(12345); // Seed for reproducibility
  std::uniform_real_distribution<double> unif_dist(0.0, 1.0); // Uniform
  ekat::Comm comm;
  ekat::logger::Logger<> logger("nucleate_ice unit tests",
                                ekat::logger::LogLevel::debug, comm);
  std::ostringstream ss;

  // initialization
  constexpr Real tol = PrecisionTolerance<Real>::tol;
  ;
  constexpr int ncol = 3;
  constexpr int pver = mam4::nlev;

  View2DHost col_delta_host("col_delta_host", ncol, pver + 1);
  View2DHost col_dens_host_ref("col_dens_host_ref", ncol, pver);

  for (int i = 0; i < ncol; ++i) {
    for (int j = 0; j < pver + 1; ++j) {
      col_delta_host(i, j) = unif_dist(generator); // Generate random number
    }
  }

  // serial implementation
  for (size_t i = 0; i < ncol; ++i) {
    col_dens_host_ref(i, 0) = col_delta_host(i, 0) + 0.5 * col_delta_host(i, 1);
  }

  for (int kk = 1; kk < pver; ++kk) {
    int km1 = kk - 1;
    for (size_t i = 0; i < ncol; ++i) {
      col_dens_host_ref(i, kk) =
          col_dens_host_ref(i, km1) +
          0.5 * (col_delta_host(i, km1 + 1) + col_delta_host(i, kk + 1));
    }
  }

  View2D col_delta("col_delta", ncol, pver + 1);
  Kokkos::deep_copy(col_delta, col_delta_host);

  View2D col_dens("col_dens", ncol, pver);
  auto team_policy = ThreadTeamPolicy(ncol, Kokkos::AUTO);

  Kokkos::parallel_for(
      team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
        const int icol = team.league_rank(); // column index
        auto col_dens_icol = ekat::subview(col_dens, icol);
        auto col_delta_icol = ekat::subview(col_delta, icol);
        mo_photo::setcol(team, col_delta_icol, col_dens_icol);
      });
  auto col_dens_host = Kokkos::create_mirror_view(col_dens);
  Kokkos::deep_copy(col_dens_host, col_dens);

  for (int i = 0; i < ncol; ++i) {
    for (int j = 0; j < pver; ++j) {
      const Real diff = std::abs(col_dens_host(i, j) - col_dens_host_ref(i, j));
      if (diff >= tol) {
        std::cout << diff << " " << col_dens_host(i, j) << " "
                  << col_dens_host_ref(i, j) << "\n";
        logger.debug(ss.str());
      }
      REQUIRE(diff <= tol);
    }
  }
}
