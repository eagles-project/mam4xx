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

void get_reynolds_number(Ensemble *ensemble) {
  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    if (!input.has("viscos_air")) {
      std::cerr << "Required name: "
                << "viscos_air" << std::endl;
      exit(1);
    }

    if (!input.has("r3lx")) {
      std::cerr << "Required name: "
                << "r3lx" << std::endl;
      exit(1);
    }

    if (!input.has("rho_air")) {
      std::cerr << "Required name: "
                << "rho_air" << std::endl;
      exit(1);
    }

    // Fetch input values.
    skywalker::Real viscos_air = input.get("viscos_air");
    skywalker::Real r3lx = input.get("r3lx");
    skywalker::Real rho_air = input.get("rho_air");

    // Compute reynolds number
    skywalker::Real re = hetfrz::get_reynolds_num(r3lx, rho_air, viscos_air);

    // Store re as an output variable
    output.set("re", re);
  });
}
