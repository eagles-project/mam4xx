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
using namespace mo_sethet;

void calc_het_rates(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {

    Real satf = input.get_array("satf")[0];
    Real rain = input.get_array("rain")[0];
    Real xhen = input.get_array("xhen")[0];
    Real tmp_hetrates = input.get_array("tmp_hetrates")[0];
    Real work1 = input.get_array("work1")[0];
    Real work2 = input.get_array("work2")[0];

    Real het_rates = 0;
         
    calc_het_rates( satf, rain, xhen, tmp_hetrates, work1, work2, het_rates);

    output.set("het_rates", het_rates);
  });
}