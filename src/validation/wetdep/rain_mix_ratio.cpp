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

void test_rain_mix_ratio_process(const Input &input, Output &output) {
  // Ensemble parameters
  // Declare array of strings for input names
  std::string input_variables[] = {"dt", "ncol", "temperature", "pmid",
                                   "sumppr"};

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
  EKAT_ASSERT(4 == input.get("ncol"));
  const Real temperature = input.get("temperature");
  const Real pmid = input.get("pmid");
  const Real sumppr = input.get("sumppr");

  ColumnView return_vals = mam4::validation::create_column_view(1);
  Kokkos::parallel_for(
      "wetdep::rain_mix_ratio", 1, KOKKOS_LAMBDA(const int) {
        return_vals[0] =
            mam4::wetdep::rain_mix_ratio(temperature, pmid, sumppr);
      });

  // Create mirror views for output arrays
  auto vals_host = Kokkos::create_mirror_view(return_vals);

  // Copy values back to the host
  Kokkos::deep_copy(vals_host, return_vals);

  // Set the output values
  output.set("rain", vals_host[0]);
}

void test_rain_mix_ratio(std::unique_ptr<Ensemble> &ensemble) {
  ensemble->process([&](const Input &input, Output &output) {
    test_rain_mix_ratio_process(input, output);
  });
}
