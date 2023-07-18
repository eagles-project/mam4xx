// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <iostream>
#include <mam4xx/drydep.hpp>
#include <skywalker.hpp>
#include <validation.hpp>

using namespace skywalker;
using namespace mam4;

void radius_for_moment(Ensemble *ensemble) {

  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    EKAT_REQUIRE_MSG(input.has("moment"), "Required name: moment");
    EKAT_REQUIRE_MSG(input.has("sig_part"), "Required name: sig_part");
    EKAT_REQUIRE_MSG(input.has("radius_part"), "Required name: radius_part");
    EKAT_REQUIRE_MSG(input.has("radius_max"), "Required name: radius_max");

    auto moment = input.get("moment");
    auto sig_part = input.get("sig_part");
    auto radius_part = input.get("radius_part");
    auto radius_max = input.get("radius_max");

    auto radius_for_moment =
        drydep::radius_for_moment(moment, sig_part, radius_part, radius_max);

    output.set("radius_for_moment", radius_for_moment);
  });
}