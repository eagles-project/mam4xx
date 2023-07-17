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

void air_kinematic_viscosity(Ensemble *ensemble) {

  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    EKAT_REQUIRE_MSG(input.has("temp"), "Required name: temp");
    EKAT_REQUIRE_MSG(input.has("pres"), "Required name: pres");

    auto temp = input.get("temp");
    auto pres = input.get("pres");

    auto air_kinematic_viscosity = drydep::air_kinematic_viscosity(temp, pres);

    output.set("air_kinematic_viscosity", air_kinematic_viscosity);
  });
}