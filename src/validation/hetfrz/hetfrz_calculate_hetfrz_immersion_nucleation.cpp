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

void calculate_hetfrz_immersion_nucleation(Ensemble *ensemble) {
  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    if (!input.has("deltat")) {
      std::cerr << "Required name: "
                << "detlat" << std::endl;
      exit(1);
    }

    if (!input.has("temperature")) {
      std::cerr << "Required name: "
                << "temperature" << std::endl;
      exit(1);
    }

    if (!input.has_array("uncoated_aer_num")) {
      std::cerr << "Required name: "
                << "uncoated_aer_num" << std::endl;
      exit(1);
    }

    if (!input.has_array("total_interstitial_aer_num")) {
      std::cerr << "Required name: "
                << "total_interstitial_aer_num" << std::endl;
      exit(1);
    }

    if (!input.has_array("total_cloudborne_aer_num")) {
      std::cerr << "Required name: "
                << "total_cloudborne_aer_num" << std::endl;
      exit(1);
    }

    if (!input.has_array("dim_theta")) {
      std::cerr << "Required name: "
                << "dim_theta" << std::endl;
      exit(1);
    }

    if (!input.has_array("pdf_imm_theta")) {
      std::cerr << "Required name: "
                << "pdf_imm_theta" << std::endl;
      exit(1);
    }

    if (!input.has("sigma_iw")) {
      std::cerr << "Required name: "
                << "sigma_iw" << std::endl;
      exit(1);
    }

    if (!input.has("eswtr")) {
      std::cerr << "Required name: "
                << "eswtr" << std::endl;
      exit(1);
    }

    if (!input.has("vwice")) {
      std::cerr << "Required name: "
                << "vwice" << std::endl;
      exit(1);
    }

    if (!input.has("r_bc")) {
      std::cerr << "Required name: "
                << "r_bc" << std::endl;
      exit(1);
    }

    if (!input.has("r_dust_a1")) {
      std::cerr << "Required name: "
                << "r_dust_a1" << std::endl;
      exit(1);
    }

    if (!input.has("r_dust_a3")) {
      std::cerr << "Required name: "
                << "r_dust_a3" << std::endl;
      exit(1);
    }

    if (!input.has("rgimm_bc")) {
      std::cerr << "Required name: "
                << "rgimm_bc" << std::endl;
      exit(1);
    }

    if (!input.has("rgimm_dust_a1")) {
      std::cerr << "Required name: "
                << "rgimm_dust_a1" << std::endl;
      exit(1);
    }

    if (!input.has("rgimm_dust_a3")) {
      std::cerr << "Required name: "
                << "rgimm_dust_a3" << std::endl;
      exit(1);
    }

    if (!input.has("do_bc")) {
      std::cerr << "Required name: "
                << "do_bc" << std::endl;
      exit(1);
    }

    if (!input.has("do_dst1")) {
      std::cerr << "Required name: "
                << "do_dst1" << std::endl;
      exit(1);
    }

    if (!input.has("do_dst3")) {
      std::cerr << "Required name: "
                << "do_dst3" << std::endl;
      exit(1);
    }

    // Fetch input values.
    auto deltat = input.get("deltat");
    auto temperature = input.get("temperature");
    auto uncoated_aer_num = input.get_array("uncoated_aer_num");
    auto total_interstitial_aer_num =
        input.get_array("total_interstitial_aer_num");
    auto total_cloudborne_aer_num = input.get_array("total_cloudborne_aer_num");
    auto dim_theta = input.get_array("dim_theta");
    auto pdf_imm_theta = input.get_array("pdf_imm_theta");
    auto sigma_iw = input.get("sigma_iw");
    auto eswtr = input.get("eswtr");
    auto vwice = input.get("vwice");
    auto r_bc = input.get("r_bc");
    auto r_dust_a1 = input.get("r_dust_a1");
    auto r_dust_a3 = input.get("r_dust_a3");
    auto rgimm_bc = input.get("rgimm_bc");
    auto rgimm_dust_a1 = input.get("rgimm_dust_a1");
    auto rgimm_dust_a3 = input.get("rgimm_dust_a3");
    auto do_bc = input.get("do_bc");
    auto do_dst1 = input.get("do_dst1");
    auto do_dst3 = input.get("do_dst3");

    bool do_bc_b = (do_bc != 0.0);
    bool do_dst1_b = (do_dst1 != 0.0);
    bool do_dst3_b = (do_dst3 != 0.0);

    haero::Real frzbcimm;
    haero::Real frzduimm;

    hetfrz::calculate_hetfrz_immersion_nucleation(
        deltat, temperature, uncoated_aer_num.data(),
        total_interstitial_aer_num.data(), total_cloudborne_aer_num.data(),
        sigma_iw, eswtr, vwice, dim_theta.data(), pdf_imm_theta.data(),
        rgimm_bc, rgimm_dust_a1, rgimm_dust_a3, r_bc, r_dust_a1, r_dust_a3,
        do_bc_b, do_dst1_b, do_dst3_b, frzbcimm, frzduimm);

    output.set("frzbcimm", frzbcimm);
    output.set("frzduimm", frzduimm);
  });
}