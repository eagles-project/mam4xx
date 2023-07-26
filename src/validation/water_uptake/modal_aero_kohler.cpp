// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <iostream>
#include <mam4xx/aero_modes.hpp>
#include <mam4xx/mam4.hpp>
#include <mam4xx/water_uptake.hpp>
#include <skywalker.hpp>
#include <validation.hpp>

using namespace skywalker;
using namespace mam4;

void modal_aero_kohler(Ensemble *ensemble) {
  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    EKAT_REQUIRE_MSG(input.has_array("rdry_in"), "Required name: rdry_in");
    EKAT_REQUIRE_MSG(input.has_array("hygro"), "Required name: hygro");
    EKAT_REQUIRE_MSG(input.has_array("rh"), "Required name: rh");

    auto rdry_in = input.get_array("rdry_in");
    auto hygro = input.get_array("hygro");
    auto rh = input.get_array("rh");

    std::vector<Real> rwet_out(1, 0);

    water_uptake::modal_aero_kohler(rdry_in.data()[0], hygro.data()[0],
                                    rh.data()[0], rwet_out.data()[0]);

    output.set("rwet_out", rwet_out);
  });
}