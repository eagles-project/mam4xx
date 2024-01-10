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
void setinv_test_single_level(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    // Ensemble parameters
    // Declare array of strings for input names
    std::string input_arrays[] = {
        "tfld",      "h2ovmr",  "vmr",       "pmid",
        "ncol",      "lchnk",   "pcols",     "pver",
        "gas_pcnst", "nfs",     "boltz_cgs", "num_tracer_cnst",
        "has_n2",    "m_ndx",   "n2_ndx",    "has_o2",
        "o2_ndx",    "has_h2o", "h2o_ndx",   "cnst_offline_yaml"};

    // Iterate over input_arrays and error if not in input
    for (std::string name : input_arrays) {
      if (!input.has_array(name.c_str())) {
        std::cerr << "Required name for array: " << name << std::endl;
        exit(1);
      }
    }

    const Real tfld = input.get_array("tfld")[0];
    const Real h2ovmr = input.get_array("h2ovmr")[0];
    // NOTE: vmr turns out to be unused, but still in the fxn signature for now
    auto vmr = input.get_array("vmr");
    const Real pmid = input.get_array("pmid")[0];
    const int nfs = input.get_array("nfs")[0];
    const Real boltz_cgs = input.get_array("boltz_cgs")[0];
    const int has_n2 = input.get_array("has_n2")[0];
    const int m_ndx = input.get_array("m_ndx")[0];
    const int n2_ndx = input.get_array("n2_ndx")[0];
    const int has_o2 = input.get_array("has_o2")[0];
    const int o2_ndx = input.get_array("o2_ndx")[0];
    const int has_h2o = input.get_array("has_h2o")[0];
    const int h2o_ndx = input.get_array("h2o_ndx")[0];
    auto cnst_offline = input.get_array("cnst_offline_yaml");

    const mam4::mo_setinv::Config setinv_config_;

    Real invariants[nfs];

    mam4::mo_setinv::setinv_single_level(invariants, tfld, h2ovmr, vmr.data(),
                                         pmid, cnst_offline.data(),
                                         setinv_config_);

    std::vector<Real> inv_out;
    for (int i = 0; i < nfs; ++i) {
      inv_out.push_back(invariants[i]);
    }

    output.set("invariants", inv_out);
  });
}
