// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/mam4.hpp>

#include <mam4xx/aero_config.hpp>
#include <skywalker.hpp>
#include <validation.hpp>

using namespace skywalker;
using namespace mam4;
using namespace haero;
void srf_emissions_smoketest(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    // Ensemble parameters
    // Declare array of strings for input names
    // std::string input_arrays[] = {
    //     "tfld",      "h2ovmr",  "vmr",       "pmid",
    //     "ncol",      "lchnk",   "pcols",     "pver",
    //     "gas_pcnst", "nfs",     "boltz_cgs", "num_tracer_cnst",
    //     "has_n2",    "m_ndx",   "n2_ndx",    "has_o2",
    //     "o2_ndx",    "has_h2o", "h2o_ndx",   "cnst_offline_yaml"};

    // // Iterate over input_arrays and error if not in input
    // for (std::string name : input_arrays) {
    //   if (!input.has_array(name.c_str())) {
    //     std::cerr << "Required name for array: " << name << std::endl;
    //     exit(1);
    //   }
    // }

    // const Real tfld = input.get_array("tfld")[0];
    // const Real h2ovmr = input.get_array("h2ovmr")[0];
    // // NOTE: vmr turns out to be unused, but still in the fxn signature for now
    // auto vmr = input.get_array("vmr");
    // const Real pmid = input.get_array("pmid")[0];
    // const int nfs = input.get_array("nfs")[0];
    // auto cnst_offline = input.get_array("cnst_offline_yaml");

    // const mam4::mo_srf_emissions::Config srf_emissions_config_;

    // Real invariants[nfs];

    // mam4::mo_srf_emissions::srf_emissions_single_level(invariants, tfld, h2ovmr, vmr.data(),
    //                                      pmid, cnst_offline.data(),
    //                                      srf_emissions_config_);

    // std::vector<Real> inv_out;
    // for (int i = 0; i < nfs; ++i) {
    //   inv_out.push_back(invariants[i]);
    // }

    // output.set("invariants", inv_out);
  });
}
