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

void indprd(Ensemble *ensemble) {

  ensemble->process([=](const Input &input, Output &output) {
    const Real zero = 0;
    const int class_id = int(input.get_array("class_id")[0]);
    std::vector<Real> prod(clscnt4, zero);

    const auto rxt = input.get_array("rxt");
    const auto extfrc = input.get_array("extfrc");

    indprd(class_id, prod.data(), rxt.data(), extfrc.data());

    output.set("prod", prod);
  });
}
