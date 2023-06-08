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

void calculate_hetfrz_contact_nucleation(Ensemble *ensemble) {

  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    if (!input.has("deltat")) {
      std::cerr << "Required name: "
                << "deltat" << std::endl;
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

    if (!input.has("icnlx")) {
      std::cerr << "Required name: "
                << "icnlx" << std::endl;
      exit(1);
    }

    if (!input.has("sigma_iv")) {
      std::cerr << "Required name: "
                << "sigma_iv" << std::endl;
      exit(1);
    }

    if (!input.has("eswtr")) {
      std::cerr << "Required name: "
                << "eswtr" << std::endl;
      exit(1);
    }

    if (!input.has("rgimm")) {
      std::cerr << "Required name: "
                << "rgimm" << std::endl;
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

    if (!input.has("Kcoll_bc")) {
      std::cerr << "Required name: "
                << "Kcoll_bc" << std::endl;
      exit(1);
    }

    if (!input.has("Kcoll_dust_a1")) {
      std::cerr << "Required name: "
                << "Kcoll_dust_a1" << std::endl;
      exit(1);
    }

    if (!input.has("Kcoll_dust_a3")) {
      std::cerr << "Required name: "
                << "Kcoll_dust_a3" << std::endl;
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
    auto icnlx = input.get("icnlx");
    auto sigma_iv = input.get("sigma_iv");
    auto eswtr = input.get("eswtr");
    auto rgimm = input.get("rgimm");
    auto r_bc = input.get("r_bc");
    auto r_dust_a1 = input.get("r_dust_a1");
    auto r_dust_a3 = input.get("r_dust_a3");
    auto Kcoll_bc = input.get("Kcoll_bc");
    auto Kcoll_dust_a1 = input.get("Kcoll_dust_a1");
    auto Kcoll_dust_a3 = input.get("Kcoll_dust_a3");
    auto do_bc = input.get("do_bc");
    auto do_dst1 = input.get("do_dst1");
    auto do_dst3 = input.get("do_dst3");

    bool do_bc_b = (do_bc != 0.0);
    bool do_dst1_b = (do_dst1 != 0.0);
    bool do_dst3_b = (do_dst3 != 0.0);

    haero::Real frzbccnt;
    haero::Real frzducnt;
    // Compute frzbccnt and frzducnt
    hetfrz::calculate_hetfrz_contact_nucleation(
        deltat, temperature, uncoated_aer_num.data(), icnlx, sigma_iv, eswtr,
        rgimm, r_bc, r_dust_a1, r_dust_a3, Kcoll_bc, Kcoll_dust_a1,
        Kcoll_dust_a3, do_bc_b, do_dst1_b, do_dst3_b, frzbccnt, frzducnt);

    output.set("frzbccnt", frzbccnt);
    output.set("frzducnt", frzducnt);
  });
}