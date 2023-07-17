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

void schmidt_number(Ensemble *ensemble) {

  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    if (!input.has("temp")) {
      std::cerr << "Required name: "
                << "temp" << std::endl;
      exit(1);
    }

    if (!input.has("pres")) {
      std::cerr << "Required name: "
                << "pres" << std::endl;
      exit(1);
    }

    if (!input.has("radius")) {
      std::cerr << "Required name: "
                << "radius" << std::endl;
      exit(1);
    }

    if (!input.has("vsc_dyn_atm")) {
      std::cerr << "Required name: "
                << "vsc_dyn_atm" << std::endl;
      exit(1);
    }

    if (!input.has("vsc_knm_atm")) {
      std::cerr << "Required name: "
                << "vsc_knm_atm" << std::endl;
      exit(1);
    }

    auto temp = input.get("temp");
    auto pres = input.get("pres");
    auto radius = input.get("radius");
    auto vsc_dyn_atm = input.get("vsc_dyn_atm");
    auto vsc_knm_atm = input.get("vsc_knm_atm");

    auto schmidt_number = drydep::gravit_settling_velocity(
        temp, pres, radius, vsc_dyn_atm, vsc_knm_atm);

    output.set("schmidt_number", schmidt_number);
  });
}