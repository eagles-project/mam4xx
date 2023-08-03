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

void test_compute_evap_frac_process(const Input &input, Output &output) {
  // Ensemble parameters
  // Declare array of strings for input names
  std::string input_variables[] = {"dt", "mam_prevap_resusp_optcc", "pdel_ik",
                                   "evap_ik", "precabx"};

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
  const int mam_prevap_resusp_optcc = input.get("mam_prevap_resusp_optcc");
  const Real pdel_ik = input.get("pdel_ik");
  const Real evap_ik = input.get("evap_ik");
  const Real precabx = input.get("precabx");

  ColumnView return_vals = mam4::validation::create_column_view(1);
  Kokkos::parallel_for(
      "wetdep::compute_evap_frac", 1, KOKKOS_LAMBDA(const int) {
        Real fracevx = 0;
        mam4::wetdep::compute_evap_frac(mam_prevap_resusp_optcc, pdel_ik,
                                        evap_ik, precabx, fracevx);

        return_vals[0] = fracevx;
      });

  // Create mirror views for output arrays
  auto vals_host = Kokkos::create_mirror_view(return_vals);

  // Copy values back to the host
  Kokkos::deep_copy(vals_host, return_vals);

  // Set the output values
  output.set("fracevx", vals_host[0]);
}

void test_compute_evap_frac(std::unique_ptr<Ensemble> &ensemble) {
  ensemble->process([&](const Input &input, Output &output) {
    test_compute_evap_frac_process(input, output);
  });
}
