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

void slip_correction_factor(Ensemble *ensemble) {

  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    if (!input.has("dyn_visc")) {
      std::cerr << "Required name: "
                << "dyn_visc" << std::endl;
      exit(1);
    }

    if (!input.has("pres")) {
      std::cerr << "Required name: "
                << "pres" << std::endl;
      exit(1);
    }

    if (!input.has("temp")) {
      std::cerr << "Required name: "
                << "temp" << std::endl;
      exit(1);
    }

    if (!input.has("particle_radius")) {
      std::cerr << "Required name: "
                << "particle_radius" << std::endl;
      exit(1);
    }

    auto dyn_visc = input.get("dyn_visc");
    auto pres = input.get("pres");
    auto temp = input.get("temp");
    auto particle_radius = input.get("particle_radius");

    auto slip_correction_factor = drydep::slip_correction_factor(dyn_visc, pres, temp, particle_radius);

    output.set("slip_correction_factor", slip_correction_factor);
  });
}