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

void calculate_mass_mean_radius(Ensemble *ensemble) {

  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    if (!input.has("bcmac")) {
      std::cerr << "Required name: "
                << "bcmac" << std::endl;
      exit(1);
    }

    if (!input.has("bcmpc")) {
      std::cerr << "Required name: "
                << "bcmpc" << std::endl;
      exit(1);
    }

    if (!input.has("dmac")) {
      std::cerr << "Required name: "
                << "dmac" << std::endl;
      exit(1);
    }

    if (!input.has("dmc")) {
      std::cerr << "Required name: "
                << "dmc" << std::endl;
      exit(1);
    }

    if (!input.has_array("total_interstitial_aer_num")) {
      std::cerr << "Required name: "
                << "total_interstitial_aer_num" << std::endl;
      exit(1);
    }

    if (!input.has_array("hetraer")) {
      std::cerr << "Required name: "
                << "hetraer" << std::endl;
      exit(1);
    }

    auto bcmac = input.get("bcmac");
    auto bcmpc = input.get("bcmpc");
    auto dmac = input.get("dmac");
    auto dmc = input.get("dmc");
    auto total_interstitial_aer_num =
        input.get_array("total_interstitial_aer_num");
    auto hetraer = input.get_array("hetraer");

    hetfrz::calculate_mass_mean_radius(bcmac, bcmpc, dmac, dmc,
                                       total_interstitial_aer_num.data(),
                                       hetraer.data());

    output.set("hetraer", hetraer);
  });
}