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
using namespace modal_aer_opt;

void calc_diag_spec(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    EKAT_REQUIRE_MSG(input.has_array("specmmr_k"), "Required name: specmmr_k");
    EKAT_REQUIRE_MSG(input.has_array("mass_k"), "Required name: mass_k");
    EKAT_REQUIRE_MSG(input.has_array("vol"), "Required name: vol");
    EKAT_REQUIRE_MSG(input.has_array("specrefr"), "Required name: specrefr ");
    EKAT_REQUIRE_MSG(input.has_array("hygro_aer"), "Required name: hygro_aer");
    EKAT_REQUIRE_MSG(input.has_array("specrefi"), "Required name:specrefi ");

    constexpr Real zero = 0;
    const auto specmmr_k = input.get_array("specmmr_k")[0];
    const auto mass_k = input.get_array("mass_k")[0];
    const auto vol = input.get_array("vol")[0];
    const auto specrefr = input.get_array("specrefr")[0];
    const auto specrefi = input.get_array("specrefi")[0];
    const auto hygro_aer = input.get_array("hygro_aer")[0];

    Real burden_s = zero;
    Real scat_s = zero;
    Real abs_s = zero;
    Real hygro_s = zero;

    calc_diag_spec(specmmr_k, mass_k, vol, specrefr, specrefi, hygro_aer,
                   burden_s, scat_s, abs_s, hygro_s);
    // NOTE: skywalker fails if there are duplicates in outputs.
    output.set("burden_s", std::vector<Real>(1, burden_s));
    output.set("scat_s", std::vector<Real>(1, scat_s));
    output.set("abs_s", std::vector<Real>(1, abs_s));
    output.set("hygro_s", std::vector<Real>(1, hygro_s));
  });
}
