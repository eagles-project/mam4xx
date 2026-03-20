// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include "testing.hpp"
#include <mam4xx/mam4.hpp>

#include <catch2/catch.hpp>

using namespace mam4;

TEST_CASE("minmod", "mam4_spitfire_transport") {
  Real aa = 1.0;
  Real bb = 2.0;

  Real mm = spitfire::minmod(aa, bb);
  REQUIRE(mm == aa);

  mm = spitfire::minmod(bb, aa);
  REQUIRE(mm == aa);
}

TEST_CASE("median", "mam4_spitfire_transport") {
  Real aa = 0.0;
  Real bb = 1.0;
  Real cc = 2.0;

  Real med = spitfire::median(aa, bb, cc);
  REQUIRE(med == bb);
}

TEST_CASE("get_flux", "mam4_spitfire_transport") {

  auto team_policy = ThreadTeamPolicy(1u, Kokkos::AUTO);

  const Real deltat = 10.0;
  ColumnView xw = testing::create_column_view(nlev);
  ColumnView phi = testing::create_column_view(nlev);
  ColumnView vel = testing::create_column_view(nlev);
  ColumnView flux = testing::create_column_view(nlev);

  spitfire::get_flux(team_policy, xw, phi, vel, deltat, flux);
}
