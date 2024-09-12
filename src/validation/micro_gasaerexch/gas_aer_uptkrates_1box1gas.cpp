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
using namespace gasaerexch;

void gas_aer_uptkrates_1box1gas(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    // Extract scalar values from input arrays
    const Real accom = input.get_array("accom")[0];
    const Real gasdiffus = input.get_array("gasdiffus")[0];
    const Real gasfreepath = input.get_array("gasfreepath")[0];
    const Real beta_inp = input.get_array("beta_inp")[0];

    // Extract array values from input arrays
    auto dgncur_awet_vec = input.get_array("dgncur_awet");
    auto lnsg_vec = input.get_array("lnsg");

    Real dgncur_awet[GasAerExch::num_mode];
    Real lnsg[GasAerExch::num_mode];
    Real uptkaer[GasAerExch::num_mode];

    for (int mode = 0; mode < GasAerExch::num_mode; ++mode) {
      dgncur_awet[mode] = dgncur_awet_vec[mode];
      lnsg[mode] = lnsg_vec[mode];
    }

    gas_aer_uptkrates_1box1gas(accom, gasdiffus, gasfreepath, beta_inp,
                               dgncur_awet, lnsg, uptkaer);

    std::vector<Real> uptkaer_vec(GasAerExch::num_mode);
    for (int mode = 0; mode < GasAerExch::num_mode; ++mode) {
      uptkaer_vec[mode] = uptkaer[mode];
    }

    // Set output values
    output.set("uptkrate", uptkaer_vec);
  });
}
