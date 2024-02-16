// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <iostream>
#include <mam4xx/drydep.hpp>
#include <skywalker.hpp>
#include <validation.hpp>

using namespace skywalker;
using namespace mam4;

void air_dynamic_viscosity(Ensemble *ensemble) {

  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    EKAT_REQUIRE_MSG(input.has("temp"), "Required name: temp");

    auto temp = input.get("temp");

    Real air_dynamic_viscosity = 0;
    Kokkos::parallel_reduce(
        1,
        KOKKOS_LAMBDA(const int, Real &vis) {
          vis = drydep::air_dynamic_viscosity(temp);
        },
        air_dynamic_viscosity);

    output.set("air_dynamic_viscosity", air_dynamic_viscosity);
  });
}
