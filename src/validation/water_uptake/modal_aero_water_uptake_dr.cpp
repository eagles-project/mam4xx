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
using namespace haero;

void modal_aero_water_uptake_dr(Ensemble *ensemble) {

  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    EKAT_REQUIRE_MSG(input.has_array("state_q"), "Required name: state_q");
    EKAT_REQUIRE_MSG(input.has_array("temperature"),
                     "Required name: temperature");
    EKAT_REQUIRE_MSG(input.has_array("pmid"), "Required name: pmid");
    EKAT_REQUIRE_MSG(input.has_array("cldn"), "Required name: cldn");
    EKAT_REQUIRE_MSG(input.has_array("dgncur_a"), "Required name: dgncur_a");
    EKAT_REQUIRE_MSG(input.has_array("dgncur_awet"),
                     "Required name: dgncur_awet");
    EKAT_REQUIRE_MSG(input.has_array("qaerwat"), "Required name: qaerwat");

    auto temperature = input.get_array("temperature")[0];
    auto pmid = input.get_array("pmid")[0];
    auto cldn = input.get_array("cldn")[0];
    auto state_q = input.get_array("state_q");
    auto dgncur_a = input.get_array("dgncur_a");
    auto dgncur_awet = input.get_array("dgncur_awet");
    auto qaerwat = input.get_array("qaerwat");

    int nspec_amode[AeroConfig::num_modes()];
    int lspectype_amode[water_uptake::maxd_aspectype][AeroConfig::num_modes()];
    Real specdens_amode[water_uptake::maxd_aspectype];
    Real spechygro[water_uptake::maxd_aspectype];

    water_uptake::get_e3sm_parameters(nspec_amode, lspectype_amode,
                                      specdens_amode, spechygro);

    water_uptake::modal_aero_water_uptake_dr(
        nspec_amode, specdens_amode, spechygro, lspectype_amode, state_q.data(),
        temperature, pmid, cldn, dgncur_a.data(), dgncur_awet.data());

    output.set("dgncur_awet", dgncur_awet);
    output.set("qaerwat", qaerwat);
  });
}