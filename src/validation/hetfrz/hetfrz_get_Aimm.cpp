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

void get_Aimm(Ensemble *ensemble) {
  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    if (!input.has("vwice")) {
      std::cerr << "Required name: "
                << "vwice" << std::endl;
      exit(1);
    }

    if (!input.has("rgimm")) {
      std::cerr << "Required name: "
                << "rigmm" << std::endl;
      exit(1);
    }

    if (!input.has("temperature")) {
      std::cerr << "Required name: "
                << "temperature" << std::endl;
      exit(1);
    }

    if (!input.has("dg0imm")) {
      std::cerr << "Required name: "
                << "dg0imm" << std::endl;
      exit(1);
    }

    // Fetch input values.
    skywalker::Real vwice = input.get("vwice");
    skywalker::Real rgimm = input.get("rgimm");
    skywalker::Real temperature = input.get("temperature");
    skywalker::Real dg0imm = input.get("dg0imm");

    // Compute Aimm
    skywalker::Real Aimm = hetfrz::get_Aimm(vwice, rgimm, temperature, dg0imm);

    // Store Aimm as an output variable
    output.set("Aimm", Aimm);
  });
}