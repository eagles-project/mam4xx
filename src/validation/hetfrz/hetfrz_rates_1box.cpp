// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/aging.hpp>
#include <skywalker.hpp>
#include <validation.hpp>

using namespace skywalker;
using namespace mam4;

void hetfrz_rates_1box(Ensemble *ensemble) {

  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    if (!input.has("dt")) {
      std::cerr << "Required name: "
                << "dt" << std::endl;
      exit(1);
    }

    if (!input.has_array("ncnst")) {
      std::cerr << "Required name: "
                << "ncnst" << std::endl;
      exit(1);
    }

    if (!input.has_array("pi")) {
      std::cerr << "Required name: "
                << "pi" << std::endl;
      exit(1);
    }

    if (!input.has_array("rhoh2o")) {
      std::cerr << "Required name: "
                << "rhoh2o" << std::endl;
      exit(1);
    }

    if (!input.has_array("deltatin")) {
      std::cerr << "Required name: "
                << "deltatin" << std::endl;
      exit(1);
    }

    if (!input.has_array("rair")) {
      std::cerr << "Required name: "
                << "rair" << std::endl;
      exit(1);
    }

    if (!input.has_array("mincld")) {
      std::cerr << "Required name: "
                << "mincld" << std::endl;
      exit(1);
    }

    if (!input.has_array("temperature")) {
      std::cerr << "Required name: "
                << "temperature" << std::endl;
      exit(1);
    }

    if (!input.has_array("pmid")) {
      std::cerr << "Required name: "
                << "pmid" << std::endl;
      exit(1);
    }

    if (!input.has_array("rho")) {
      std::cerr << "Required name: "
                << "rho" << std::endl;
      exit(1);
    }

    if (!input.has_array("ast")) {
      std::cerr << "Required name: "
                << "ast" << std::endl;
      exit(1);
    }

    if (!input.has_array("qc")) {
      std::cerr << "Required name: "
                << "qc" << std::endl;
      exit(1);
    }

    if (!input.has_array("nc")) {
      std::cerr << "Required name: "
                << "nc" << std::endl;
      exit(1);
    }

    if (!input.has_array("state_q_bc_accum")) {
      std::cerr << "Required name: "
                << "state_q_bc_accum" << std::endl;
      exit(1);
    }

    if (!input.has_array("state_q_pom_accum")) {
      std::cerr << "Required name: "
                << "state_q_pom_accum" << std::endl;
      exit(1);
    }

    if (!input.has_array("state_q_soa_accum")) {
      std::cerr << "Required name: "
                << "state_q_soa_accum" << std::endl;
      exit(1);
    }

    if (!input.has_array("state_q_dust_accum")) {
      std::cerr << "Required name: "
                << "state_q_dust_accum" << std::endl;
      exit(1);
    }

    if (!input.has_array("state_q_nacl_accum")) {
      std::cerr << "Required name: "
                << "state_q_nacl_accum" << std::endl;
      exit(1);
    }

    if (!input.has_array("state_q_mom_accum")) {
      std::cerr << "Required name: "
                << "state_q_mom_accum" << std::endl;
      exit(1);
    }

    if (!input.has_array("state_q_num_accum")) {
      std::cerr << "Required name: "
                << "state_q_num_accum" << std::endl;
      exit(1);
    }

    if (!input.has_array("state_q_dust_coarse")) {
      std::cerr << "Required name: "
                << "state_q_dust_coarse" << std::endl;
      exit(1);
    }

    if (!input.has_array("state_q_nacl_coarse")) {
      std::cerr << "Required name: "
                << "state_q_nacl_coarse" << std::endl;
      exit(1);
    }

    if (!input.has_array("state_q_so4_coarse")) {
      std::cerr << "Required name: "
                << "state_q_so4_coarse" << std::endl;
      exit(1);
    }

    if (!input.has_array("state_q_bc_coarse")) {
      std::cerr << "Required name: "
                << "state_q_bc_coarse" << std::endl;
      exit(1);
    }

    if (!input.has_array("state_q_pom_coarse")) {
      std::cerr << "Required name: "
                << "state_q_pom_coarse" << std::endl;
      exit(1);
    }

    if (!input.has_array("state_q_soa_coarse")) {
      std::cerr << "Required name: "
                << "state_q_soa_coarse" << std::endl;
      exit(1);
    }

    if (!input.has_array("state_q_mom_coarse")) {
      std::cerr << "Required name: "
                << "state_q_mom_coarse" << std::endl;
      exit(1);
    }

    if (!input.has_array("state_q_num_coarse")) {
      std::cerr << "Required name: "
                << "state_q_num_coarse" << std::endl;
      exit(1);
    }

    if (!input.has_array("state_q_bc_pcarbon")) {
      std::cerr << "Required name: "
                << "state_q_bc_pcarbon" << std::endl;
      exit(1);
    }

    if (!input.has_array("state_q_pom_pcarbon")) {
      std::cerr << "Required name: "
                << "state_q_pom_pcarbon" << std::endl;
      exit(1);
    }

    if (!input.has_array("state_q_mom_pcarbon")) {
      std::cerr << "Required name: "
                << "state_q_mom_pcarbon" << std::endl;
      exit(1);
    }

    if (!input.has_array("state_q_num_pcarbon")) {
      std::cerr << "Required name: "
                << "state_q_num_pcarbon" << std::endl;
      exit(1);
    }

    if (!input.has_array("aer_cb")) {
      std::cerr << "Required name: "
                << "aer_cb" << std::endl;
      exit(1);
    }

    if (!input.has_array("factnum")) {
      std::cerr << "Required name: "
                << "factnum" << std::endl;
      exit(1);
    }

    auto dt = input.get("dt");
    auto ncnst = input.get_array("ncnst");
    auto pi = input.get_array("pi");
    auto rhoh2o = input.get_array("rhoh2o");
    auto deltatin = input.get_array("deltatin");
    auto rair = input.get_array("rair");
    auto mincld = input.get_array("mincld");
    auto temperature = input.get_array("temperature");
    auto pmid = input.get_array("pmid");
    auto rho = input.get_array("rho");
    auto ast = input.get_array("ast");
    auto qc = input.get_array("qc");
    auto nc = input.get_array("nc");
    auto state_q_bc_accum = input.get_array("state_q_bc_accum");
    auto state_q_pom_accum = input.get_array("state_q_pom_accum");
    auto state_q_soa_accum = input.get_array("state_q_soa_accum");
    auto state_q_dust_accum = input.get_array("state_q_dust_accum");
    auto state_q_nacl_accum = input.get_array("state_q_nacl_accum");
    auto state_q_mom_accum = input.get_array("state_q_mom_accum");
    auto state_q_num_accum = input.get_array("state_q_num_accum");
    auto state_q_dust_coarse = input.get_array("state_q_dust_coarse");
    auto state_q_nacl_coarse = input.get_array("state_q_nacl_coarse");
    auto state_q_so4_coarse = input.get_array("state_q_so4_coarse");
    auto state_q_bc_coarse = input.get_array("state_q_bc_coarse");
    auto state_q_pom_coarse = input.get_array("state_q_pom_coarse");
    auto state_q_soa_coarse = input.get_array("state_q_soa_coarse");
    auto state_q_mom_coarse = input.get_array("state_q_mom_coarse");
    auto state_q_num_coarse = input.get_array("state_q_num_coarse");
    auto state_q_bc_pcarbon = input.get_array("state_q_bc_pcarbon");
    auto state_q_pom_pcarbon = input.get_array("state_q_pom_pcarbon");
    auto state_q_mom_pcarbon = input.get_array("state_q_mom_pcarbon");
    auto state_q_num_pcarbon = input.get_array("state_q_num_pcarbon");
    auto aer_cb = input.get_array("aer_cb");
    auto factnum = input.get_array("factnum");

  });
}