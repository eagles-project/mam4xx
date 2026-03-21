// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/hetfrz.hpp>
#include <validation.hpp>

#include <iostream>

using namespace skywalker;

void calculate_vars_for_pdf_imm(Ensemble *ensemble) {
  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
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

    auto dim_theta = input.get_array("dim_theta");
    auto pdf_imm_theta = input.get_array("pdf_imm_theta");

    // Compute dim_theta and pdf_imm_theta
    mam4::hetfrz::calculate_vars_for_pdf_imm(dim_theta.data(),
                                             pdf_imm_theta.data());

    output.set("dim_theta", dim_theta);
    output.set("pdf_imm_theta", pdf_imm_theta);
  });
}
