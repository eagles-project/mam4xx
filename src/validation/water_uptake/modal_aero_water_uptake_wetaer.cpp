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

void modal_aero_water_uptake_wetaer(Ensemble *ensemble) {

  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    EKAT_REQUIRE_MSG(input.has_array("rhcrystal"), "Required name: rhcrystal");
    EKAT_REQUIRE_MSG(input.has_array("rhdeliques"),
                     "Required name: rhdeliques");
    EKAT_REQUIRE_MSG(input.has_array("dgncur_a"), "Required name: dgncur_a");
    EKAT_REQUIRE_MSG(input.has_array("dryrad"), "Required name: dryrad");
    EKAT_REQUIRE_MSG(input.has_array("hygro"), "Required name: hygro");
    EKAT_REQUIRE_MSG(input.has_array("rh"), "Required name: rh");
    EKAT_REQUIRE_MSG(input.has_array("naer"), "Required name: naer");
    EKAT_REQUIRE_MSG(input.has_array("dryvol"), "Required name: dryvol");

    auto rhcrystal = input.get_array("rhcrystal");
    auto rhdeliques = input.get_array("rhdeliques");
    auto dgncur_a = input.get_array("dgncur_a");
    auto dryrad = input.get_array("dryrad");
    auto hygro = input.get_array("hygro");
    auto rh = input.get_array("rh");
    auto naer = input.get_array("naer");
    auto dryvol = input.get_array("dryvol");

    std::vector<Real> wetrad(AeroConfig::num_modes(), 0);
    std::vector<Real> wetvol(AeroConfig::num_modes(), 0);
    std::vector<Real> wtrvol(AeroConfig::num_modes(), 0);
    std::vector<Real> dgncur_awet(AeroConfig::num_modes(), 0);
    std::vector<Real> qaerwat(AeroConfig::num_modes(), 0);

    water_uptake::modal_aero_water_uptake_wetaer(
        rhcrystal.data(), rhdeliques.data(), dgncur_a.data(), dryrad.data(),
        hygro.data(), rh.data()[0], naer.data(), dryvol.data(), wetrad.data(),
        wetvol.data(), wtrvol.data(), dgncur_awet.data(), qaerwat.data());

    output.set("wetrad", wetrad);
    output.set("wetvol", wetvol);
    output.set("wtrvol", wtrvol);
    output.set("dgncur_awet", dgncur_awet);
    output.set("qaerwat", qaerwat);
  });
}