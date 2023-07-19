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

void adjrxt(Ensemble *ensemble) {

  ensemble->process([=](const Input &input, Output &output) {
    auto inv = input.get_array("inv");
    auto mtot = input.get_array("mtot")[0];
    auto rate = input.get_array("rate");

    adjrxt(rate.data(), inv.data(), mtot);

    output.set("rate", rate);
  });
}
