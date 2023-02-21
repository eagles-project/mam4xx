// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <catch2/catch.hpp>
#include <haero/math.hpp>
#include <mam4xx/mam4.hpp>

using Real = haero::Real;

TEST_CASE("test_min_max_bound", "utils") {

  Real low = 0.0;
  Real high = 1.0;
  Real num = 0.5;

  /// if num is between low and high return num
  Real ret = mam4::utils::min_max_bound(low, high, num);
  REQUIRE(ret == num);

  /// if num is greater than high return high
  num = 1.5;
  ret = mam4::utils::min_max_bound(low, high, num);
  REQUIRE(ret == high);

  // if num is less than low return low
  num = -0.5;
  ret = mam4::utils::min_max_bound(low, high, num);
  REQUIRE(ret == low);
}
