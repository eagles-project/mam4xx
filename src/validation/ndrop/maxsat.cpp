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

void maxsat(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    const Real zeta = input.get_array("zeta")[0];
    const int nmode = input.get_array("nmode")[0];

    Real eta[nmode];
    Real smc[nmode];
    std::vector<Real> eta_vec(nmode);
    std::vector<Real> smc_vec(nmode);
    eta_vec = input.get_array("eta");
    smc_vec = input.get_array("smc");
    for (int n = 0; n < nmode; ++n) {
      eta[n] = eta_vec[n];
      smc[n] = smc_vec[n];
    }

    Real smax = 0;

    ndrop::maxsat(zeta, eta, nmode, smc, smax);

    output.set("smax", smax);
  });
}
