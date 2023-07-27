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

void test_wetdep_scavenging_process(const Input &input, Output &output) {
  // Ensemble parameters
  // Declare array of strings for input names
  std::string input_variables[] = {
      "dt",       "is_st_cu", "is_strat_cloudborne", "deltat",    "fracp",
      "precabx",  "cldv_ik",  "scavcoef_ik",         "sol_factb", "sol_facti",
      "tracer_1", "tracer_2"};

  // Iterate over input_variables and error if not in input
  for (std::string name : input_variables) {
    if (!input.has(name.c_str())) {
      std::cerr << "Required name for variable: " << name << std::endl;
      exit(1);
    }
  }

  // Parse input
  // These first two values are unused
  EKAT_ASSERT(0 == input.get("dt"));
  const int is_st_cu = input.get("is_st_cu");
  const int is_strat_cloudborne = input.get("is_strat_cloudborne");
  const Real deltat = input.get("deltat");
  const Real fracp = input.get("fracp");
  const Real precabx = input.get("precabx");
  const Real cldv_ik = input.get("cldv_ik");
  const Real scavcoef_ik = input.get("scavcoef_ik");
  const Real sol_factb = input.get("sol_factb");
  const Real sol_facti = input.get("sol_facti");
  const Real tracer_1 = input.get("tracer_1");
  const Real tracer_2 = input.get("tracer_2");

  ColumnView return_vals = mam4::validation::create_column_view(2);
  Kokkos::parallel_for(
      "wetdep::wetdep_scavenging", 1, KOKKOS_LAMBDA(const int) {
        Real src = 0, fin = 0;
        mam4::wetdep::wetdep_scavenging(
            is_st_cu, is_strat_cloudborne, deltat, fracp, precabx, cldv_ik,
            scavcoef_ik, sol_factb, sol_facti, tracer_1, tracer_2, src, fin);

        return_vals[0] = src;
        return_vals[1] = fin;
      });

  // Create mirror views for output arrays
  auto vals_host = Kokkos::create_mirror_view(return_vals);

  // Copy values back to the host
  Kokkos::deep_copy(vals_host, return_vals);

  // Set the output values
  output.set("src", vals_host[1]);
  output.set("fin", vals_host[2]);
}

void test_wetdep_scavenging(std::unique_ptr<Ensemble> &ensemble) {
  ensemble->process([&](const Input &input, Output &output) {
    test_wetdep_scavenging_process(input, output);
  });
}
