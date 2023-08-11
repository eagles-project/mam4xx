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

void modal_aero_water_uptake_dryaer(Ensemble *ensemble) {
  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    EKAT_REQUIRE_MSG(input.has_array("dgncur_a"), "Required name: dgncur_a");
    EKAT_REQUIRE_MSG(input.has_array("state_q"), "Required name: state_q");

    auto dgncur_a = input.get_array("dgncur_a");
    auto state_q = input.get_array("state_q");

    int nspec_amode[AeroConfig::num_modes()];
    int lspectype_amode[water_uptake::maxd_aspectype][AeroConfig::num_modes()];
    Real specdens_amode[water_uptake::maxd_aspectype];
    Real spechygro[water_uptake::maxd_aspectype];

    water_uptake::get_e3sm_parameters(nspec_amode, lspectype_amode,
                                      specdens_amode, spechygro);

    std::vector<Real> hygro(AeroConfig::num_modes(), 0.0);
    std::vector<Real> naer(AeroConfig::num_modes(), 0.0);
    std::vector<Real> dryrad(AeroConfig::num_modes(), 0.0);
    std::vector<Real> dryvol(AeroConfig::num_modes(), 0.0);
    std::vector<Real> drymass(AeroConfig::num_modes(), 0.0);
    std::vector<Real> rhcrystal(AeroConfig::num_modes(), 0.0);
    std::vector<Real> rhdeliques(AeroConfig::num_modes(), 0.0);
    std::vector<Real> specdens_1(AeroConfig::num_modes(), 0.0);

    water_uptake::modal_aero_water_uptake_dryaer(
        nspec_amode, specdens_amode, spechygro, lspectype_amode, state_q.data(),
        dgncur_a.data(), hygro.data(), naer.data(), dryrad.data(),
        dryvol.data(), drymass.data(), rhcrystal.data(), rhdeliques.data(),
        specdens_1.data());

    output.set("hygro", hygro);
    output.set("naer", naer);
    output.set("dryrad", dryrad);
    output.set("dryvol", dryvol);
    output.set("drymass", drymass);
    output.set("rhcrystal", rhcrystal);
    output.set("rhdeliques", rhdeliques);
    output.set("specdens_1", specdens_1);
  });
}