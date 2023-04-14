// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <iostream>
#include <mam4xx/gasaerexch.hpp>
#include <mam4xx/mam4.hpp>
#include <skywalker.hpp>
#include <validation.hpp>
#include <vector>

using mam4::gasaerexch::gas_aer_uptkrates_1box1gas;

using namespace haero;
using namespace skywalker;

void test_gasaerexch_uptkrates_1box1gas_process(const Input &input,
                                                Output &output,
                                                mam4::GasAerExch &gasaerexch) {
  // Ensemble parameters
  if (!input.has("temp")) {
    std::cerr << "Required name: "
              << "temp" << std::endl;
    exit(1);
  }
  if (!input.has_array("dgncur_awet")) {
    std::cerr << "Required name: "
              << "dgncur_awet" << std::endl;
    exit(1);
  }
  if (!input.has_array("lnsg")) {
    std::cerr << "Required name: "
              << "lnsg" << std::endl;
    exit(1);
  }
  if (!input.has_array("aernum")) {
    std::cerr << "Required name: "
              << "aernum" << std::endl;
    exit(1);
  }
  const bool has_mw_gas = input.has("mw_gas");
  const bool has_pmid = input.has("pmid");
  const bool has_beta = input.has("beta");
  const bool has_nghq = input.has("nghq");
  const bool has_vol_molar_gas = input.has("vol_molar_gas");
  const bool has_condense_to_mode = input.has_array("condense_to_mode");
  const bool has_solution = input.has_array("uptkaer");

  //-------------------------------------------------------
  // Process input, do calculations, and prepare output
  //-------------------------------------------------------
  const int n_mode = 4;
  int nghq = 2;
  const Real accom = 0.65000000000000002;
  Real beta_inp = 1.5000000000000000;
  const Real pi = 3.1415926535897931;
  const Real r_universal = 8314.4675910000005;
  Real mw_gas = 98.078400000000002;
  const Real mw_air = 28.966000000000001;
  Real pmid = 100000.00000000000;
  const Real pstd = 101325.00000000000;
  Real vol_molar_gas = 42.880000000000003;
  const Real vol_molar_air = 20.100000000000001;

  bool l_condense_to_mode[n_mode] = {true, true, true, true};
  // Parse input
  Real dgncur_awet[n_mode];
  Real lnsg[n_mode];
  {
    const std::vector<Real> array = input.get_array("dgncur_awet");
    for (size_t i = 0; i < array.size() && i < n_mode; ++i)
      dgncur_awet[i] = array[i];
  }
  {
    const std::vector<Real> array = input.get_array("lnsg");
    for (size_t i = 0; i < array.size() && i < n_mode; ++i)
      lnsg[i] = array[i];
  }
  const std::vector<Real> aernum = input.get_array("aernum");
  const Real temp = input.get("temp");
  if (has_mw_gas)
    mw_gas = input.get("mw_gas");
  if (has_pmid)
    pmid = input.get("pmid");
  if (has_beta)
    beta_inp = input.get("beta");
  if (has_nghq)
    nghq = static_cast<int>(input.get("nghq"));
  if (has_vol_molar_gas)
    vol_molar_gas = input.get("vol_molar_gas");
  if (has_condense_to_mode) {
    const std::vector<Real> values = input.get_array("condense_to_mode");
    for (int i = 0; i < n_mode; ++i)
      l_condense_to_mode[i] = (values[i] <= 0) ? false : true;
  }
  std::vector<Real> test_uptkaer;
  if (has_solution) {
    test_uptkaer = input.get_array("uptkaer");
  }
  ColumnView uptkaer_dev = mam4::validation::create_column_view(n_mode);
  Kokkos::parallel_for(
      "gasaerexch::gas_aer_uptkrates_1box1gas", 1, KOKKOS_LAMBDA(const int) {
        Real uptkaer[n_mode];
        mam4::gasaerexch::gas_aer_uptkrates_1box1gas(
            l_condense_to_mode, temp, pmid, pstd, mw_gas, mw_air, vol_molar_gas,
            vol_molar_air, accom, r_universal, pi, beta_inp, nghq, dgncur_awet,
            lnsg, uptkaer);
        for (size_t i = 0; i < n_mode; ++i)
          uptkaer_dev(i) = uptkaer[i];
      });
  Kokkos::Array<Real, n_mode> uptkaer;
  {
    auto host_view = Kokkos::create_mirror_view(uptkaer_dev);
    Kokkos::deep_copy(host_view, uptkaer_dev);
    for (size_t i = 0; i < n_mode; ++i)
      uptkaer[i] = host_view[i];
  }
  // Write the computed nucleation rate.
  {
    std::vector<Real> values(n_mode);
    for (size_t i = 0; i < values.size() && i < n_mode; ++i)
      values[i] = uptkaer[i];
    output.set("uptkaer", values);
  }
}

void test_gasaerexch_uptkrates_1box1gas(std::unique_ptr<Ensemble> &ensemble) {
  mam4::AeroConfig mam4_config;
  mam4::GasAerExch gasaerexch;
  mam4::GasAerExch::Config config;
  gasaerexch.init(mam4_config, config);

  ensemble->process([&](const Input &input, Output &output) {
    test_gasaerexch_uptkrates_1box1gas_process(input, output, gasaerexch);
  });
}
