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

void calculate_coated_fraction(Ensemble *ensemble) {

  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    if (!input.has("so4mac")) {
      std::cerr << "Required name: "
                << "so4mac" << std::endl;
      exit(1);
    }

    if (!input.has("pommac")) {
      std::cerr << "Required name: "
                << "pommac" << std::endl;
      exit(1);
    }

    if (!input.has("mommac")) {
      std::cerr << "Required name: "
                << "mommac" << std::endl;
      exit(1);
    }

    if (!input.has("soamac")) {
      std::cerr << "Required name: "
                << "soamac" << std::endl;
      exit(1);
    }

    if (!input.has("dmac")) {
      std::cerr << "Required name: "
                << "dmac" << std::endl;
      exit(1);
    }

    if (!input.has("bcmac")) {
      std::cerr << "Required name: "
                << "bcmac" << std::endl;
      exit(1);
    }

    if (!input.has("mommpc")) {
      std::cerr << "Required name: "
                << "mommpc" << std::endl;
      exit(1);
    }

    if (!input.has("pommpc")) {
      std::cerr << "Required name: "
                << "pommpc" << std::endl;
      exit(1);
    }

    if (!input.has("bcmpc")) {
      std::cerr << "Required name: "
                << "bcmpc" << std::endl;
      exit(1);
    }

    if (!input.has("so4mc")) {
      std::cerr << "Required name: "
                << "so4mc" << std::endl;
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

    if (!input.has("mommc")) {
      std::cerr << "Required name: "
                << "mommc" << std::endl;
      exit(1);
    }

    if (!input.has("dmc")) {
      std::cerr << "Required name: "
                << "dmc" << std::endl;
      exit(1);
    }

    if (!input.has_array("total_cloudborne_aer_num")) {
      std::cerr << "Required name: "
                << "total_cloudborne_aer_num" << std::endl;
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

    if (!input.has("air_density")) {
      std::cerr << "Required name: "
                << "air_density" << std::endl;
      exit(1);
    }

    auto so4mac = input.get("so4mac");
    auto pommac = input.get("pommac");
    auto mommac = input.get("mommac");
    auto soamac = input.get("soamac");
    auto dmac = input.get("dmac");
    auto bcmac = input.get("bcmac");
    auto mommpc = input.get("mommpc");
    auto pommpc = input.get("pommpc");
    auto bcmpc = input.get("bcmpc");
    auto so4mc = input.get("so4mc");
    auto pommc = input.get("pommc");
    auto soamc = input.get("soamc");
    auto mommc = input.get("mommc");
    auto dmc = input.get("dmc");
    auto air_density = input.get("air_density");

    auto total_cloudborne_aer_num = input.get_array("total_cloudborne_aer_num");
    auto total_interstitial_aer_num =
        input.get_array("total_interstitial_aer_num");
    auto hetraer = input.get_array("hetraer");
    auto total_aer_num = input.get_array("total_aer_num");
    auto coated_aer_num = input.get_array("coated_aer_num");
    auto uncoated_aer_num = input.get_array("uncoated_aer_num");
    auto dstcoat = input.get_array("dstcoat");
    haero::Real na500 = 0.0;
    haero::Real tot_na500 = 0.0;

    hetfrz::calculate_coated_fraction(
        air_density, so4mac, pommac, mommac, soamac, dmac, bcmac, mommpc,
        pommpc, bcmpc, so4mc, pommc, soamc, mommc, dmc,
        total_interstitial_aer_num.data(), total_cloudborne_aer_num.data(),
        hetraer.data(), total_aer_num.data(), coated_aer_num.data(),
        uncoated_aer_num.data(), dstcoat.data(), na500, tot_na500);

    output.set("na500", na500);
    output.set("tot_na500", tot_na500);
    output.set("coated_aer_num", coated_aer_num);
    output.set("dstcoat", dstcoat);
    output.set("total_aer_num", total_aer_num);
    output.set("uncoated_aer_num", uncoated_aer_num);
  });
}