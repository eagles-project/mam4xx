// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <iostream>
#include <mam4xx/hetfrz.hpp>
#include <validation.hpp>

using namespace skywalker;

void get_latent_heat_vapor(Ensemble *ensemble) {

  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    if (!input.has("temperature")) {
      std::cerr << "Required name: "
                << "temperature" << std::endl;
      exit(1);
    }

    // Fetch input values.
    skywalker::Real temperature = input.get("temperature");

    // Compute reynolds number
    skywalker::Real latvap = mam4::hetfrz::get_latent_heat_vapor(temperature);

    // Store re as an output variable
    output.set("latvap", latvap);
  });
}
