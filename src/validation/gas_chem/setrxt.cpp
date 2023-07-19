// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/mam4.hpp>

#include <mam4xx/aero_config.hpp>
#include <mam4xx/gas_chem.hpp>
#include <skywalker.hpp>
#include <validation.hpp>

using namespace skywalker;
using namespace mam4;
using namespace gas_chemistry;

void setrxt(Ensemble *ensemble) {

  ensemble->process([=](const Input &input, Output &output) {
    const auto temp = input.get_array("temp")[0];
    auto rate = input.get_array("rate");
    setrxt(rate.data(), temp);

    output.set("rate", rate);
  });
}
