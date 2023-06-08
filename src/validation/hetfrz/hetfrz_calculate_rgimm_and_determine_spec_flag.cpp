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

void calculate_rgimm_and_determine_spec_flag(Ensemble *ensemble) {
  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    if (!input.has("vwice")) {
      std::cerr << "Required name: "
                << "vwice" << std::endl;
      exit(1);
    }

    if (!input.has("sigma_iw")) {
      std::cerr << "Required name: "
                << "sigma_iw" << std::endl;
      exit(1);
    }

    if (!input.has("temperature")) {
      std::cerr << "Required name: "
                << "temperatures" << std::endl;
      exit(1);
    }

    if (!input.has("aw")) {
      std::cerr << "Required name: "
                << "aw" << std::endl;
      exit(1);
    }

    if (!input.has("supersatice")) {
      std::cerr << "Required name: "
                << "supersatice" << std::endl;
      exit(1);
    }

    if (!input.has("rgimm")) {
      std::cerr << "Required name: "
                << "rgimm" << std::endl;
      exit(1);
    }

    auto vwice = input.get("vwice");
    auto sigma_iw = input.get("sigma_iw");
    auto temperature = input.get("temperature");
    auto aw = input.get("aw");
    auto supersatice = input.get("supersatice");
    auto rgimm = input.get("rgimm");

    bool do_spec_flag = true;
    hetfrz::calculate_rgimm_and_determine_spec_flag(
        vwice, sigma_iw, temperature, aw, supersatice, rgimm, do_spec_flag);

    skywalker::Real do_spec_flag_real = int(do_spec_flag);

    output.set("rgimm", rgimm);
    output.set("do_spec_flag", do_spec_flag_real);
  });
}