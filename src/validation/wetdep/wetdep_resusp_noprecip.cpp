// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <iostream>
#include <mam4xx/mam4.hpp>
#include <mam4xx/wet_dep.hpp>
#include <skywalker.hpp>
#include <validation.hpp>
#include <vector>

using namespace haero;
using namespace skywalker;

void test_wetdep_resusp_noprecip_process(const Input &input, Output &output) {
  // Ensemble parameters
  // Declare array of strings for input names
  std::string input_variables[] = {"dt",
                                   "is_st_cu",
                                   "mam_prevap_resusp_optcc",
                                   "precabx_base_old",
                                   "precabx_old",
                                   "scavabx_old",
                                   "precnumx_base_old"};

  // Iterate over input_variables and error if not in input
  for (std::string name : input_variables) {
    if (!input.has(name.c_str())) {
      std::cerr << "Required name for variable: " << name << std::endl;
      exit(1);
    }
  }

  // Parse input
  // These first two values are unused
  assert(0 == input.get("dt"));
  const int is_st_cu = input.get("is_st_cu");
  const int mam_prevap_resusp_optcc = input.get("mam_prevap_resusp_optcc");
  const Real precabx_base_old = input.get("precabx_base_old");
  const Real precabx_old = input.get("precabx_old");
  const Real scavabx_old = input.get("scavabx_old");
  const Real precnumx_base_old = input.get("precnumx_base_old");

  ColumnView return_vals = mam4::validation::create_column_view(4);
  Kokkos::parallel_for(
      "wetdep::wetdep_resusp_noprecip", 1, KOKKOS_LAMBDA(const int) {
        Real precabx_base_new = 0, precabx_new = 0, scavabx_new = 0,
             resusp_x = 0;
        mam4::wetdep::wetdep_resusp_noprecip(
            is_st_cu, mam_prevap_resusp_optcc, precabx_old, precabx_base_old,
            scavabx_old, precnumx_base_old, precabx_new, precabx_base_new,
            scavabx_new, resusp_x);

        return_vals[0] = precabx_base_new;
        return_vals[1] = precabx_new;
        return_vals[2] = scavabx_new;
        return_vals[3] = resusp_x;
      });

  // Create mirror views for output arrays
  auto vals_host = Kokkos::create_mirror_view(return_vals);

  // Copy values back to the host
  Kokkos::deep_copy(vals_host, return_vals);

  // Set the output values
  output.set("precabx_base_new", vals_host[1]);
  output.set("precabx_new", vals_host[2]);
  output.set("scavabx_new", vals_host[3]);
  output.set("resusp_x", vals_host[4]);
}

void test_wetdep_resusp_noprecip(std::unique_ptr<Ensemble> &ensemble) {
  ensemble->process([&](const Input &input, Output &output) {
    test_wetdep_resusp_noprecip_process(input, output);
  });
}
