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

void calculate_interstitial_aer_num(Ensemble *ensemble) {

  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    if (!input.has("bcmac")) {
      std::cerr << "Required name: "
                << "bcmac" << std::endl;
      exit(1);
    }

    if (!input.has("dmac")) {
      std::cerr << "Required name: "
                << "dmac" << std::endl;
      exit(1);
    }

    if (!input.has("bcmpc")) {
      std::cerr << "Required name: "
                << "bcmpc" << std::endl;
      exit(1);
    }

    if (!input.has("dmc")) {
      std::cerr << "Required name: "
                << "dmc" << std::endl;
      exit(1);
    }

    if (!input.has("ssmc")) {
      std::cerr << "Required name: "
                << "ssmc" << std::endl;
      exit(1);
    }

    if (!input.has("mommc")) {
      std::cerr << "Required name: "
                << "mommc" << std::endl;
      exit(1);
    }

    if (!input.has("bcmc")) {
      std::cerr << "Required name: "
                << "bcmc" << std::endl;
      exit(1);
    }

    if (!input.has("pommc")) {
      std::cerr << "Required name: "
                << "pommc" << std::endl;
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

    auto bcmac = input.get("bcmac");
    auto dmac = input.get("dmac");
    auto bcmpc = input.get("bcmpc");
    auto dmc = input.get("dmc");
    auto ssmc = input.get("ssmc");
    auto mommc = input.get("mommc");
    auto bcmc = input.get("bcmc");
    auto pommc = input.get("pommc");
    auto soamc = input.get("soamc");
    auto ncoarse = input.get("ncoarse");

    auto total_interstitial_aer_num =
        input.get_array("total_interstitial_aer_num");

    hetfrz::calculate_interstitial_aer_num(bcmac, dmac, bcmpc, dmc, ssmc, mommc,
                                           bcmc, pommc, soamc, ncoarse,
                                           total_interstitial_aer_num.data());

    output.set("total_interstitial_aer_num", total_interstitial_aer_num);
  });
}