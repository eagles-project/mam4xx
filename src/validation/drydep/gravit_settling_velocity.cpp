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

void gravit_settling_velocity(Ensemble *ensemble) {

  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    if (!input.has("particle_radius")) {
      std::cerr << "Required name: "
                << "particle_radius" << std::endl;
      exit(1);
    }

    if (!input.has("particle_density")) {
      std::cerr << "Required name: "
                << "particle_density" << std::endl;
      exit(1);
    }

    if (!input.has("slip_correction")) {
      std::cerr << "Required name: "
                << "slip_correction" << std::endl;
      exit(1);
    }

    if (!input.has("dynamic_viscosity")) {
      std::cerr << "Required name: "
                << "dynamic_viscosity" << std::endl;
      exit(1);
    }

    if (!input.has("particle_sig")) {
      std::cerr << "Required name: "
                << "particle_radius" << std::endl;
      exit(1);
    }

    auto particle_radius = input.get("particle_radius");
    auto particle_density = input.get("particle_density");
    auto slip_correction = input.get("slip_correction");
    auto dynamic_viscosity = input.get("dynamic_viscosity");
    auto particle_sig = input.get("particle_sig");

    auto gsv = drydep::gravit_settling_velocity(
        particle_radius, particle_density, slip_correction, dynamic_viscosity,
        particle_sig);

    output.set("gravit_settling_velocity", gsv);
  });
}