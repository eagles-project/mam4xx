// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/mam4.hpp>
#include <validation.hpp>

using namespace skywalker;

void water_uptake_wetdens(Ensemble *ensemble) {

  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    EKAT_REQUIRE_MSG(input.has_array("wetvol"), "Required name: wetvol");
    EKAT_REQUIRE_MSG(input.has_array("wtrvol"), "Required name: wtrvol");
    EKAT_REQUIRE_MSG(input.has_array("drymass"), "Required name: drymass");
    EKAT_REQUIRE_MSG(input.has_array("specdens_1"),
                     "Required name: specdens_1");

    auto wetvol = input.get_array("wetvol");
    auto wtrvol = input.get_array("wtrvol");
    auto drymass = input.get_array("drymass");
    auto specdens_1 = input.get_array("specdens_1");

    std::vector<Real> wetdens(mam4::AeroConfig::num_modes(), 0);

    mam4::water_uptake::modal_aero_water_uptake_wetdens(
        wetvol.data(), wtrvol.data(), drymass.data(), specdens_1.data(),
        wetdens.data());

    output.set("wetdens", wetdens);
  });
}
