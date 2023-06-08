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

void get_aer_radius(Ensemble *ensemble) {

  ensemble->process([=](const Input &input, Output &output) {
    if (!input.has("specdens")) {
      std::cerr << "Required name: "
                << "specdens" << std::endl;
      exit(1);
    }

    if (!input.has("aermc")) {
      std::cerr << "Required name: "
                << "aermc" << std::endl;
      exit(1);
    }

    if (!input.has("aernum")) {
      std::cerr << "Required name: "
                << "aernum" << std::endl;
      exit(1);
    }

    // Fetch input values.
    skywalker::Real specdens = input.get("specdens");
    skywalker::Real aermc = input.get("aermc");
    skywalker::Real aernum = input.get("aernum");

    skywalker::Real radius = hetfrz::get_aer_radius(specdens, aermc, aernum);

    // Store radius
    output.set("radius", radius);
  });
}
