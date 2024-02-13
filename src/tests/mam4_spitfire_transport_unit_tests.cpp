// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include "mam4xx/drydep.hpp"
#include "mam4xx/spitfire_transport.hpp"
#include "testing.hpp"
#include <catch2/catch.hpp>
#include <mam4xx/mam4.hpp>

using namespace haero;
using namespace mam4;

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

  auto team_policy = ThreadTeamPolicy(1u, Kokkos::AUTO);

  const Real deltat = 10.0;
  ColumnView xw = haero::testing::create_column_view(mam4::nlev);
  ColumnView phi = haero::testing::create_column_view(mam4::nlev);
  ColumnView vel = haero::testing::create_column_view(mam4::nlev);
  ColumnView flux = haero::testing::create_column_view(mam4::nlev);

  spitfire::get_flux(team_policy, xw, phi, vel, deltat, flux);
}
