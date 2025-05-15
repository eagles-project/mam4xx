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
  // pver is constant and the size of our arrays
  const int pver = 72;
  int nlev = 72;
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
  EKAT_ASSERT(temperature.size() == pver);
  EKAT_ASSERT(pmid.size() == pver);
  EKAT_ASSERT(pdel.size() == pver);
  EKAT_ASSERT(cmfdqr.size() == pver);
  EKAT_ASSERT(evapc.size() == pver);
  EKAT_ASSERT(cldt.size() == pver);
  EKAT_ASSERT(cldcu.size() == pver);
  EKAT_ASSERT(cldst.size() == pver);
  EKAT_ASSERT(evapr.size() == pver);
  EKAT_ASSERT(prain.size() == pver);

  using View1DHost = typename HostType::view_1d<Real>;
  using View1D = typename DeviceType::view_1d<Real>;

  // For temperature
  View1DHost temperature_host((Real *)temperature.data(), pver);
  View1D temperature_arr("temperature", pver);
  Kokkos::deep_copy(temperature_arr, temperature_host);

  // For pmid
  View1DHost pmid_host((Real *)pmid.data(), pver);
  View1D pmid_arr("pmid", pver);
  Kokkos::deep_copy(pmid_arr, pmid_host);

  // For pdel
  View1DHost pdel_host((Real *)pdel.data(), pver);
  View1D pdel_arr("pdel", pver);
  Kokkos::deep_copy(pdel_arr, pdel_host);

  // For cmfdqr
  View1DHost cmfdqr_host((Real *)cmfdqr.data(), pver);
  View1D cmfdqr_arr("cmfdqr", pver);
  Kokkos::deep_copy(cmfdqr_arr, cmfdqr_host);

  // For evapc
  View1DHost evapc_host((Real *)evapc.data(), pver);
  View1D evapc_arr("evapc", pver);
  Kokkos::deep_copy(evapc_arr, evapc_host);

  // For cldt
  View1DHost cldt_host((Real *)cldt.data(), pver);
  View1D cldt_arr("cldt", pver);
  Kokkos::deep_copy(cldt_arr, cldt_host);

  // For cldcu
  View1DHost cldcu_host((Real *)cldcu.data(), pver);
  View1D cldcu_arr("cldcu", pver);
  Kokkos::deep_copy(cldcu_arr, cldcu_host);

  // For cldst
  View1DHost cldst_host((Real *)cldst.data(), pver);
  View1D cldst_arr("cldst", pver);
  Kokkos::deep_copy(cldst_arr, cldst_host);

  // For evapr
  View1DHost evapr_host((Real *)evapr.data(), pver);
  View1D evapr_arr("evapr", pver);
  Kokkos::deep_copy(evapr_arr, evapr_host);

  // For prain
  View1DHost prain_host((Real *)prain.data(), pver);
  View1D prain_arr("prain", pver);
  Kokkos::deep_copy(prain_arr, prain_host);

  // Prepare device views for output arrays
  ColumnView cldv_dev = mam4::validation::create_column_view(pver);
  ColumnView cldvcu_dev = mam4::validation::create_column_view(pver);
  ColumnView cldvst_dev = mam4::validation::create_column_view(pver);
  ColumnView rain_dev = mam4::validation::create_column_view(pver);

  // TODO: let's use team policy
  Kokkos::parallel_for(
      "wetdep::clddiag", 1, KOKKOS_LAMBDA(const int) {
        mam4::wetdep::clddiag(pver, temperature_arr, pmid_arr, pdel_arr,
                              cmfdqr_arr, evapc_arr, cldt_arr, cldcu_arr,
                              cldst_arr, evapr_arr, prain_arr, cldv_dev,
                              cldvcu_dev, cldvst_dev, rain_dev);
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
  Real cldv_arr[pver];
  Real cldvcu_arr[pver];
  Real cldvst_arr[pver];
  Real rain_arr[pver];

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
}

void test_wetdep_clddiag(std::unique_ptr<Ensemble> &ensemble) {
  ensemble->process([&](const Input &input, Output &output) {
    test_wetdep_clddiag_process(input, output);
  });
}
