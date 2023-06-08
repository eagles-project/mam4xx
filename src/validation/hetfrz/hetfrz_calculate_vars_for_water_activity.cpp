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

void calculate_vars_for_water_activity(Ensemble *ensemble) {

  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    if (!input.has("bcmac")) {
      std::cerr << "Required name: "
                << "bcmac" << std::endl;
      exit(1);
    }

    if (!input.has("bcmc")) {
      std::cerr << "Required name: "
                << "bcmc" << std::endl;
      exit(1);
    }

    if (!input.has("mommac")) {
      std::cerr << "Required name: "
                << "mommac" << std::endl;
      exit(1);
    }

    if (!input.has("mommc")) {
      std::cerr << "Required name: "
                << "mommc" << std::endl;
      exit(1);
    }

    if (!input.has("num_accum")) {
      std::cerr << "Required name: "
                << "num_accum" << std::endl;
      exit(1);
    }

    if (!input.has("num_coarse")) {
      std::cerr << "Required name: "
                << "num_coarse" << std::endl;
      exit(1);
    }

    if (!input.has("pommac")) {
      std::cerr << "Required name: "
                << "pommac" << std::endl;
      exit(1);
    }

    if (!input.has("pommc")) {
      std::cerr << "Required name: "
                << "pommc" << std::endl;
      exit(1);
    }

    if (!input.has("so4mac")) {
      std::cerr << "Required name: "
                << "so4mac" << std::endl;
      exit(1);
    }

    if (!input.has("so4mc")) {
      std::cerr << "Required name: "
                << "so4mc" << std::endl;
      exit(1);
    }

    if (!input.has("soamac")) {
      std::cerr << "Required name: "
                << "soamac" << std::endl;
      exit(1);
    }

    if (!input.has("soamc")) {
      std::cerr << "Required name: "
                << "soamc" << std::endl;
      exit(1);
    }

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

    auto bcmac = input.get("bcmac");
    auto bcmc = input.get("bcmc");
    auto mommac = input.get("mommac");
    auto mommc = input.get("mommc");
    auto num_accum = input.get("num_accum");
    auto num_coarse = input.get("num_coarse");
    auto pommac = input.get("pommac");
    auto pommc = input.get("pommc");
    auto so4mac = input.get("so4mac");
    auto so4mc = input.get("so4mc");
    auto soamac = input.get("soamac");
    auto soamc = input.get("soamc");
    auto awcam = input.get_array("awcam");
    auto awfacm = input.get_array("awfacm");
    auto total_interstitial_aer_num =
        input.get_array("total_interstitial_aer_num");

    hetfrz::calculate_vars_for_water_activity(
        so4mac, soamac, bcmac, mommac, pommac, num_accum, so4mc, mommc, bcmc,
        pommc, soamc, num_coarse, total_interstitial_aer_num.data(),
        awcam.data(), awfacm.data());

    output.set("awcam", awcam);
    output.set("awfacm", awfacm);
  });
}