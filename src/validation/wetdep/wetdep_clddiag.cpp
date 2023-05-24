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

void test_wetdep_clddiag_process(const Input &input, Output &output) {
  // pver is constant?
  // making ncol constant seems easiest - a single value in an array in the yaml
  const int pver = 72;
  const int ncol = 4;
  // Ensemble parameters
  // Declare array of strings for input names
  std::string input_variables[] = {"dt"};

  std::string input_arrays[] = {"temperature", "pmid", "pdel",  "cmfdqr",
                                "evapc",       "cldt", "cldcu", "cldst",
                                "evapr",       "prain"};

  // Iterate over input_variables and error if not in input
  for (std::string name : input_variables) {
    if (!input.has(name.c_str())) {
      std::cerr << "Required name for variable: " << name << std::endl;
      exit(1);
    }
  }
  // Iterate over input_arrays and error if not in input
  for (std::string name : input_arrays) {
    if (!input.has_array(name.c_str())) {
      std::cerr << "Required name for array: " << name << std::endl;
      exit(1);
    }
  }

  // Parse input
  auto dt = input.get("dt");
  auto temperature = input.get_array("temperature");
  auto pmid = input.get_array("pmid");
  auto pdel = input.get_array("pdel");
  auto cmfdqr = input.get_array("cmfdqr");
  auto evapc = input.get_array("evapc");
  auto cldt = input.get_array("cldt");
  auto cldcu = input.get_array("cldcu");
  auto cldst = input.get_array("cldst");
  auto evapr = input.get_array("evapr");
  auto prain = input.get_array("prain");

  // Assert arrays are the correct size
  assert(temperature.size() == pver);
  assert(pmid.size() == pver);
  assert(pdel.size() == pver);
  assert(cmfdqr.size() == pver);
  assert(evapc.size() == pver);
  assert(cldt.size() == pver);
  assert(cldcu.size() == pver);
  assert(cldst.size() == pver);
  assert(evapr.size() == pver);
  assert(prain.size() == pver);

  // Create Real arrays for inputs
  // std::vectors can't be copied directly to device memory by Kokkos
  // Maybe this should be a unique_ptr..
  // Since pver is actually hard coded, maybe this isn't necessary
  auto temperature_arr = new Real[temperature.size()];
  auto pdel_arr = new Real[pdel.size()];
  auto cmfdqr_arr = new Real[cmfdqr.size()];
  auto evapc_arr = new Real[evapc.size()];
  auto cldt_arr = new Real[cldt.size()];
  auto cldcu_arr = new Real[cldcu.size()];
  auto cldst_arr = new Real[cldst.size()];
  auto evapr_arr = new Real[evapr.size()];
  auto prain_arr = new Real[prain.size()];

  // Use std::copy to copy input arrays to Real arrays
  std::copy(temperature.begin(), temperature.end(), temperature_arr);
  std::copy(pdel.begin(), pdel.end(), pdel_arr);
  std::copy(cmfdqr.begin(), cmfdqr.end(), cmfdqr_arr);
  std::copy(evapc.begin(), evapc.end(), evapc_arr);
  std::copy(cldt.begin(), cldt.end(), cldt_arr);
  std::copy(cldcu.begin(), cldcu.end(), cldcu_arr);
  std::copy(cldst.begin(), cldst.end(), cldst_arr);
  std::copy(evapr.begin(), evapr.end(), evapr_arr);
  std::copy(prain.begin(), prain.end(), prain_arr);

  // Prepare device views for output arrays
  ColumnView cldv_dev = mam4::validation::create_column_view(pver);
  ColumnView cldvcu_dev = mam4::validation::create_column_view(pver);
  ColumnView cldvst_dev = mam4::validation::create_column_view(pver);
  ColumnView rain_dev = mam4::validation::create_column_view(pver);

  Kokkos::parallel_for(
      "wetdep::clddiag", 1, KOKKOS_LAMBDA(const int) {
        // On device, create Real arrays for outputs
        Real cldv[pver];
        Real cldvcu[pver];
        Real cldvst[pver];
        Real rain[pver];

        // TODO - actually run function
        mam4::wetdep::clddiag();

        // Copy values back to host
        for (size_t i = 0; i < pver; ++i) {
          cldv_dev(i) = cldv[i];
          cldvcu_dev(i) = cldvcu[i];
          cldvst_dev(i) = cldvst[i];
          rain_dev(i) = rain[i];
        }
      });

  // Create mirror views for output arrays
  auto cldv_host = Kokkos::create_mirror_view(cldv_dev);
  auto cldvcu_host = Kokkos::create_mirror_view(cldvcu_dev);
  auto cldvst_host = Kokkos::create_mirror_view(cldvst_dev);
  auto rain_host = Kokkos::create_mirror_view(rain_dev);

  // Copy values back to the host
  Kokkos::deep_copy(cldv_host, cldv_dev);
  Kokkos::deep_copy(cldvcu_host, cldvcu_dev);
  Kokkos::deep_copy(cldvst_host, cldvst_dev);
  Kokkos::deep_copy(rain_host, rain_dev);

  // Copy into a temporary real array before putting into std::vector
  auto cldv_arr = new Real[pver];
  auto cldvcu_arr = new Real[pver];
  auto cldvst_arr = new Real[pver];
  auto rain_arr = new Real[pver];

  for (size_t i = 0; i < pver; ++i) {
    cldv_arr[i] = cldv_host(i);
    cldvcu_arr[i] = cldvcu_host(i);
    cldvst_arr[i] = cldvst_host(i);
    rain_arr[i] = rain_host(i);
  }

  // Create Vectors for output arrays and copy in place
  std::vector<Real> cldv(cldv_arr, cldv_arr + pver);
  std::vector<Real> cldvcu(cldvcu_arr, cldvcu_arr + pver);
  std::vector<Real> cldvst(cldvst_arr, cldvst_arr + pver);
  std::vector<Real> rain(rain_arr, rain_arr + pver);

  // Set the output values
  output.set("cldv", cldv);
  output.set("cldvcu", cldvcu);
  output.set("cldvst", cldvst);
  output.set("rain", rain);

  // Clean up Real arrays on the stack
  delete[] pdel_arr;
  delete[] cmfdqr_arr;
  delete[] evapc_arr;
  delete[] cldt_arr;
  delete[] cldcu_arr;
  delete[] cldst_arr;
  delete[] evapr_arr;
  delete[] prain_arr;
  delete[] cldv_arr;
  delete[] cldvcu_arr;
  delete[] cldvst_arr;
  delete[] rain_arr;
}

void test_wetdep_clddiag(std::unique_ptr<Ensemble> &ensemble) {
  ensemble->process([&](const Input &input, Output &output) {
    test_wetdep_clddiag_process(input, output);
  });
}
