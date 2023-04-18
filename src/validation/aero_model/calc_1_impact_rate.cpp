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

void calc_1_impact_rate(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    const Real zero = 0;
    Real scavratenum = zero;
    Real scavratevol = zero;
    const Real dg0 = input.get_array("dg0")[0];
    const Real sigmag = input.get_array("sigmag")[0];
    const Real rhoaero = input.get_array("rhoaero")[0];
    const Real temp = input.get_array("temp")[0];
    const Real press = input.get_array("press")[0];

    // const Real input.get_array("")[0]

    aero_model::calc_1_impact_rate(dg0,         //  in
                                   sigmag,      //  in
                                   rhoaero,     //  in
                                   temp,        //  in
                                   press,       //  in
                                   scavratenum, // out
                                   scavratevol);

    output.set("scavratenum", std::vector(1, scavratenum));
    output.set("scavratevol", std::vector(1, scavratevol));
  });
}