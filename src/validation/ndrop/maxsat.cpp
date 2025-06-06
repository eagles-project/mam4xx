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
    int nmode = input.get_array("nmode")[0];

    std::vector<Real> smc(nmode);
    std::vector<Real> eta(nmode);
    eta = input.get_array("eta");
    smc = input.get_array("smc");
    Real smax = 0;
    ndrop::maxsat(zeta, eta.data(), nmode, smc.data(), smax);
    output.set("smax", smax);
  });
}
