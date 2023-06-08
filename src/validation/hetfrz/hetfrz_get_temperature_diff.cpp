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

void get_temperature_diff(Ensemble *ensemble) {
  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    if (!input.has("latvap")) {
      std::cerr << "Required name: "
                << "latvap" << std::endl;
      exit(1);
    }

    if (!input.has("Ktherm_air")) {
      std::cerr << "Required name: "
                << "Kthemr_air" << std::endl;
      exit(1);
    }

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

    // Fetch input values.
    skywalker::Real latvap = input.get("latvap");
    skywalker::Real Ktherm_air = input.get("Ktherm_air");
    skywalker::Real temperature = input.get("temperature");
    skywalker::Real eswtr = input.get("eswtr");
    skywalker::Real pressure = input.get("pressure");

    // Compute temperature difference
    skywalker::Real T_diff = hetfrz::get_temperature_diff(
        temperature, pressure, eswtr, latvap, Ktherm_air);

    // Store T_diff as an output variable
    output.set("T_diff", T_diff);
  });
}