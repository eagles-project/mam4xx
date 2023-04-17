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

void get_dg0imm(Ensemble *ensemble) {

  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    if (!input.has("sigma_iw")) {
      std::cerr << "Required name: "
                << "sigma_iw" << std::endl;
      exit(1);
    }

    if (!input.has("rgimm")) {
      std::cerr << "Required name: "
                << "rgimm" << std::endl;
      exit(1);
    }

    // Fetch input values.
    skywalker::Real sigma_iw = input.get("sigma_iw");
    skywalker::Real rgimm = input.get("rgimm");

    // Compute dg0imm
    skywalker::Real dg0imm = hetfrz::get_dg0imm(sigma_iw, rgimm);

    // Store dg0imm
    output.set("dg0imm", dg0imm);
  });
}