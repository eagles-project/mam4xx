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
  // nlev is constant and the size of our arrays
  const int nlev = 72;
  Real pblh = 1000;
  Atmosphere atm = mam4::testing::create_atmosphere(nlev, pblh);
  // Ensemble parameters
  // Declare array of strings for input names
  std::string input_variables[] = {"dt"};

  std::string input_arrays[] = {"ncol",   "temperature", "pmid", "pdel",
                                "cmfdqr", "evapc",       "cldt", "cldcu",
                                "cldst",  "evapr",       "prain"};

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
  // These first two values are unused
  // auto dt = input.get("dt");
  // auto ncol = input.get_array("ncol");
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
  EKAT_ASSERT(temperature.size() == nlev);
  EKAT_ASSERT(pmid.size() == nlev);
  EKAT_ASSERT(pdel.size() == nlev);
  EKAT_ASSERT(cmfdqr.size() == nlev);
  EKAT_ASSERT(evapc.size() == nlev);
  EKAT_ASSERT(cldt.size() == nlev);
  EKAT_ASSERT(cldcu.size() == nlev);
  EKAT_ASSERT(cldst.size() == nlev);
  EKAT_ASSERT(evapr.size() == nlev);
  EKAT_ASSERT(prain.size() == nlev);

  // Create Real arrays for inputs
  // std::vectors can't be copied directly to device memory by Kokkos
  // Maybe this should be a unique_ptr..
  // Since nlev is actually hard coded, maybe this isn't necessary
  Real temperature_arr[nlev];
  Real pmid_arr[nlev];
  Real pdel_arr[nlev];
  Real cmfdqr_arr[nlev];
  Real evapc_arr[nlev];
  Real cldt_arr[nlev];
  Real cldcu_arr[nlev];
  Real cldst_arr[nlev];
  Real evapr_arr[nlev];
  Real prain_arr[nlev];

  // Use std::copy to copy input arrays to Real arrays
  std::copy(temperature.begin(), temperature.end(), temperature_arr);
  std::copy(pmid.begin(), pmid.end(), pmid_arr);
  std::copy(pdel.begin(), pdel.end(), pdel_arr);
  std::copy(cmfdqr.begin(), cmfdqr.end(), cmfdqr_arr);
  std::copy(evapc.begin(), evapc.end(), evapc_arr);
  std::copy(cldt.begin(), cldt.end(), cldt_arr);
  std::copy(cldcu.begin(), cldcu.end(), cldcu_arr);
  std::copy(cldst.begin(), cldst.end(), cldst_arr);
  std::copy(evapr.begin(), evapr.end(), evapr_arr);
  std::copy(prain.begin(), prain.end(), prain_arr);

  // Prepare device views for output arrays
  ColumnView cldv_dev = mam4::validation::create_column_view(nlev);
  ColumnView cldvcu_dev = mam4::validation::create_column_view(nlev);
  ColumnView cldvst_dev = mam4::validation::create_column_view(nlev);
  ColumnView rain_dev = mam4::validation::create_column_view(nlev);

  Kokkos::parallel_for(
      "wetdep::clddiag", 1, KOKKOS_LAMBDA(const int) {
        // On device, create Real arrays for outputs
        Real cldv[nlev];
        Real cldvcu[nlev];
        Real cldvst[nlev];
        Real rain[nlev];

        mam4::wetdep::clddiag(nlev, temperature_arr, pmid_arr, pdel_arr,
                              cmfdqr_arr, evapc_arr, cldt_arr, cldcu_arr,
                              cldst_arr, evapr_arr, prain_arr, cldv, cldvcu,
                              cldvst, rain);

        // Copy values back to host
        for (size_t i = 0; i < nlev; ++i) {
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
  Real cldv_arr[nlev];
  Real cldvcu_arr[nlev];
  Real cldvst_arr[nlev];
  Real rain_arr[nlev];

  for (size_t i = 0; i < nlev; ++i) {
    cldv_arr[i] = cldv_host(i);
    cldvcu_arr[i] = cldvcu_host(i);
    cldvst_arr[i] = cldvst_host(i);
    rain_arr[i] = rain_host(i);
  }

  // Create Vectors for output arrays and copy in place
  std::vector<Real> cldv(cldv_arr, cldv_arr + nlev);
  std::vector<Real> cldvcu(cldvcu_arr, cldvcu_arr + nlev);
  std::vector<Real> cldvst(cldvst_arr, cldvst_arr + nlev);
  std::vector<Real> rain(rain_arr, rain_arr + nlev);

  // Set the output values
  output.set("cldv", cldv);
  output.set("cldvcu", cldvcu);
  output.set("cldvst", cldvst);
  output.set("rain", rain);
}

void test_wetdep_clddiag(std::unique_ptr<Ensemble> &ensemble) {
  ensemble->process([&](const Input &input, Output &output) {
    test_wetdep_clddiag_process(input, output);
  });
}
