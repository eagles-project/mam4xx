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

void modal_aero_water_uptake_rh_clearair(Ensemble *ensemble) {

  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    EKAT_REQUIRE_MSG(input.has_array("temperature"),
                     "Required name: temperature");
    EKAT_REQUIRE_MSG(input.has_array("pmid"), "Required name: pmid");
    EKAT_REQUIRE_MSG(input.has_array("h2ommr"), "Required name: h2ommr");
    EKAT_REQUIRE_MSG(input.has_array("cldn"), "Required name: cldn");

    auto temperature = input.get_array("temperature");
    auto pmid = input.get_array("pmid");
    auto cldn = input.get_array("cldn");
    auto h2ommr = input.get_array("h2ommr");

    std::vector<Real> rh(1, 0);
    water_uptake::modal_aero_water_uptake_rh_clearair(
        temperature.data()[0], pmid.data()[0], h2ommr.data()[0], cldn.data()[0],
        rh.data()[0]);

    output.set("rh", rh);
  });
}