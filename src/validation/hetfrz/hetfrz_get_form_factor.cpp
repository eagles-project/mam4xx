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

void get_form_factor(Ensemble *ensemble) {

  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    if (!input.has("alpha")) {
      std::cerr << "Required name: "
                << "alpha" << std::endl;
      exit(1);
    }

    // Fetch input values.
    skywalker::Real alpha = input.get("alpha");

    // Compute dg0imm
    skywalker::Real f_form = hetfrz::get_form_factor(alpha);

    // Store dg0imm
    output.set("f_form", f_form);
  });
}