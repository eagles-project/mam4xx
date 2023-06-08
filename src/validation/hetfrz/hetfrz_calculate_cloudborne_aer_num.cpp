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

void calculate_cloudborne_aer_num(Ensemble *ensemble) {

  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    if (!input.has("dmac_cb")) {
      std::cerr << "Required name: "
                << "dmac_cb" << std::endl;
      exit(1);
    }

    if (!input.has("ssmac_cb")) {
      std::cerr << "Required name: "
                << "ssmac_cb" << std::endl;
      exit(1);
    }

    if (!input.has("so4mac_cb")) {
      std::cerr << "Required name: "
                << "so4mac_cb" << std::endl;
      exit(1);
    }

    if (!input.has("bcmac_cb")) {
      std::cerr << "Required name: "
                << "bcmac_cb" << std::endl;
      exit(1);
    }

    if (!input.has("pommac_cb")) {
      std::cerr << "Required name: "
                << "pommac_cb" << std::endl;
      exit(1);
    }

    if (!input.has("soamac_cb")) {
      std::cerr << "Required name: "
                << "soamac_cb" << std::endl;
      exit(1);
    }

    if (!input.has("mommac_cb")) {
      std::cerr << "Required name: "
                << "mommac_cb" << std::endl;
      exit(1);
    }

    if (!input.has("num_accum_cb")) {
      std::cerr << "Required name: "
                << "num_accum_cb" << std::endl;
      exit(1);
    }

    if (!input.has("dmc_cb")) {
      std::cerr << "Required name: "
                << "dmc_cb" << std::endl;
      exit(1);
    }

    if (!input.has("ssmc_cb")) {
      std::cerr << "Required name: "
                << "ssmc_cb" << std::endl;
      exit(1);
    }

    if (!input.has("mommc_cb")) {
      std::cerr << "Required name: "
                << "mommc_cb" << std::endl;
      exit(1);
    }

    if (!input.has("bcmc_cb")) {
      std::cerr << "Required name: "
                << "bcmc_cb" << std::endl;
      exit(1);
    }

    if (!input.has("pommc_cb")) {
      std::cerr << "Required name: "
                << "pommc_cb" << std::endl;
      exit(1);
    }

    if (!input.has("soamc_cb")) {
      std::cerr << "Required name: "
                << "soamc_cb" << std::endl;
      exit(1);
    }

    if (!input.has("num_coarse_cb")) {
      std::cerr << "Required name: "
                << "num_coarse_cb" << std::endl;
      exit(1);
    }

    if (!input.has_array("total_cloudborne_aer_num")) {
      std::cerr << "Required name: "
                << "total_cloudborne_aer_num" << std::endl;
      exit(1);
    }
    auto dmac_cb = input.get("dmac_cb");
    auto ssmac_cb = input.get("ssmac_cb");
    auto so4mac_cb = input.get("so4mac_cb");
    auto bcmac_cb = input.get("bcmac_cb");
    auto pommac_cb = input.get("pommac_cb");
    auto soamac_cb = input.get("soamac_cb");
    auto mommac_cb = input.get("mommac_cb");
    auto num_accum_cb = input.get("num_accum_cb");
    auto dmc_cb = input.get("dmc_cb");
    auto ssmc_cb = input.get("ssmc_cb");
    auto mommc_cb = input.get("mommc_cb");
    auto bcmc_cb = input.get("bcmc_cb");
    auto pommc_cb = input.get("pommc_cb");
    auto soamc_cb = input.get("soamc_cb");
    auto num_coarse_cb = input.get("num_coarse_cb");
    auto total_cloudborne_aer_num = input.get_array("total_cloudborne_aer_num");
    hetfrz::calculate_cloudborne_aer_num(
        dmac_cb, ssmac_cb, so4mac_cb, bcmac_cb, pommac_cb, soamac_cb, mommac_cb,
        num_accum_cb, dmc_cb, ssmc_cb, mommc_cb, bcmc_cb, pommc_cb, soamc_cb,
        num_coarse_cb, total_cloudborne_aer_num.data());

    output.set("total_cloudborne_aer_num", total_cloudborne_aer_num);
  });
}