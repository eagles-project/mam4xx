// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include "testing.hpp"
#include <mam4xx/mam4.hpp>

#include <catch2/catch.hpp>

using mam4::Real;

#ifdef MAM4XX_ENABLE_GPU
constexpr int team_size = mam4::nlev;
#else
constexpr int team_size = 1;
#endif

TEST_CASE("minmod", "mam4_spitfire_transport") {
  Real aa = 1.0;
  Real bb = 2.0;

  Real mm = mam4::spitfire::minmod(aa, bb);
  REQUIRE(mm == aa);

  mm = mam4::spitfire::minmod(bb, aa);
  REQUIRE(mm == aa);
}

TEST_CASE("median", "mam4_spitfire_transport") {
  Real aa = 0.0;
  Real bb = 1.0;
  Real cc = 2.0;

  Real med = mam4::spitfire::median(aa, bb, cc);
  REQUIRE(med == bb);
}

TEST_CASE("get_flux", "mam4_spitfire_transport") {

  auto team_policy = mam4::ThreadTeamPolicy(1u, team_size);

  const Real deltat = 10.0;
  mam4::ColumnView xw = mam4::testing::create_column_view(mam4::nlev);
  mam4::ColumnView phi = mam4::testing::create_column_view(mam4::nlev);
  mam4::ColumnView vel = mam4::testing::create_column_view(mam4::nlev);
  mam4::ColumnView flux = mam4::testing::create_column_view(mam4::nlev);

  mam4::spitfire::get_flux(team_policy, xw, phi, vel, deltat, flux);
}
