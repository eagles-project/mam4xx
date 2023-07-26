// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <iostream>
#include <mam4xx/hetfrz.hpp>
#include <skywalker.hpp>
#include <validation.hpp>

using namespace skywalker;
using namespace mam4;

void collkernel(Ensemble *ensemble) {

  ensemble->process([=](const Input &input, Output &output) {
    if (!input.has("temperature")) {
      std::cerr << "Required name: "
                << "temperature" << std::endl;
      exit(1);
    }

    if (!input.has("pressure")) {
      std::cerr << "Required name: "
                << "pressure" << std::endl;
      exit(1);
    }

    if (!input.has("eswtr")) {
      std::cerr << "Required name: "
                << "eswtr" << std::endl;
      exit(1);
    }

    if (!input.has("r3ls")) {
      std::cerr << "Required name: "
                << "r3ls" << std::endl;
      exit(1);
    }

    if (!input.has("r_bc")) {
      std::cerr << "Required name: "
                << "r_bc" << std::endl;
      exit(1);
    }

    if (!input.has("r_dust_a1")) {
      std::cerr << "Required name: "
                << "r_dust_a1" << std::endl;
      exit(1);
    }

    if (!input.has("r_dust_a3")) {
      std::cerr << "Required name: "
                << "r_dust_a3" << std::endl;
      exit(1);
    }

    // Fetch input values.
    skywalker::Real temperature = input.get("temperature");
    skywalker::Real pressure = input.get("pressure");
    skywalker::Real eswtr = input.get("eswtr");
    skywalker::Real r3ls = input.get("r3ls");
    skywalker::Real r_bc = input.get("r_bc");
    skywalker::Real r_dust_a1 = input.get("r_dust_a1");
    skywalker::Real r_dust_a3 = input.get("r_dust_a3");

    // Output variables
    skywalker::Real Kcoll_bc = 0.0;
    skywalker::Real Kcoll_dust_a1 = 0.0;
    skywalker::Real Kcoll_dust_a3 = 0.0;

    hetfrz::collkernel(temperature, pressure, eswtr, r3ls, r_bc, r_dust_a1,
                       r_dust_a3, Kcoll_bc, Kcoll_dust_a1, Kcoll_dust_a3);

    // Store collection kernels as output
    output.set("Kcoll_bc", Kcoll_bc);
    output.set("Kcoll_dust_a1", Kcoll_dust_a1);
    output.set("Kcoll_dust_a3", Kcoll_dust_a3);
  });
}
