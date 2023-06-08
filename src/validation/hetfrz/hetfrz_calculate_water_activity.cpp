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

void calculate_water_activity(Ensemble *ensemble) {
  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    if (!input.has_array("total_interstitial_aer_num")) {
      std::cerr << "Required name: "
                << "total_interstitial_aer_num" << std::endl;
      exit(1);
    }

    if (!input.has_array("awcam")) {
      std::cerr << "Required name: "
                << "awcam" << std::endl;
      exit(1);
    }

    if (!input.has_array("awfacm")) {
      std::cerr << "Required name: "
                << "awfacm" << std::endl;
      exit(1);
    }

    if (!input.has_array("aw")) {
      std::cerr << "Required name: "
                << "aw" << std::endl;
      exit(1);
    }

    if (!input.has("r3lx")) {
      std::cerr << "Required name: "
                << "r3lx" << std::endl;
      exit(1);
    }

    auto total_interstitial_aer_num =
        input.get_array("total_interstitial_aer_num");
    auto awcam = input.get_array("awcam");
    auto awfacm = input.get_array("awfacm");
    auto aw = input.get_array("aw");
    auto r3lx = input.get("r3lx");

    // Compute aw
    hetfrz::calculate_water_activity(total_interstitial_aer_num.data(),
                                     awcam.data(), awfacm.data(), r3lx,
                                     aw.data());

    output.set("aw", aw);
  });
}