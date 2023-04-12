// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <ekat/ekat_type_traits.hpp>
#include <ekat/logging/ekat_logger.hpp>
#include <ekat/mpi/ekat_comm.hpp>
#include <mam4xx/mam4.hpp>

#include <catch2/catch.hpp>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>

using namespace haero;
using namespace mam4;

TEST_CASE("get_air_viscosity", "mam4_hetfrz") {

  const Real threshold_error = std::numeric_limits<float>::epsilon();

  // This is a simple function so test with values computed offline
  const Real tc[11] = {-10., -8., -6., -4., -2., 0., 2., 4., 6., 8., 10.};
  const Real visc_test[11] = {
      1.6678000000000002e-05, 1.6780320000000001e-05, 1.6881680000000000e-05,
      1.6982080000000000e-05, 1.7081520000000000e-05, 1.7180000000000002e-05,
      1.7277520000000000e-05, 1.7374080000000002e-05, 1.7469680000000002e-05,
      1.7564319999999999e-05, 1.7657999999999999e-05};

  for (int i = 0; i < 11; ++i) {
    REQUIRE(haero::abs(visc_test[i] - mam4::hetfrz::get_air_viscosity(tc[i])) <
            threshold_error);
  }
}

TEST_CASE("get_latent_heat_vapor", "mam4_hetfrz") {

  const Real threshold_error = std::numeric_limits<float>::epsilon();

  // This is a simple function so test with values computed offline
  const Real tc[11] = {-10., -8., -6., -4., -2., 0., 2., 4., 6., 8., 10.};

  const Real lh_test[11] = {
      2524652.1612,    2519836.6075904, 2515045.5635071998, 2510276.0801088,
      2505525.2085536, 2500790.,        2496067.5056064003, 2491354.7765312,
      2486648.8639328, 2481946.8189696, 2477245.6928000003,
  };

  for (int i = 0; i < 11; ++i) {
    REQUIRE(haero::abs(lh_test[i] - mam4::hetfrz::get_latent_heat_vapor(
                                        tc[i])) < threshold_error);
  }
}