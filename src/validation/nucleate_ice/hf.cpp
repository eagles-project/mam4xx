// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/mam4.hpp>
#include <skywalker.hpp>
#include <validation.hpp>

using namespace skywalker;
using namespace mam4;
using namespace haero;

void hf(Ensemble *ensemble) {

  // We don't need any settings for this particular test.
  // Settings settings = ensemble->settings();

  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    // Fetch ensemble parameters

    const Real temp = input.get_array("Temperature")[0]; // air temperature
    const Real w_vlc = input.get_array("w_vlc")[0];
    const Real RH = input.get_array("RH")[0];
    const Real Na = input.get_array("Na")[0];
    const Real subgrid = input.get_array("subgrid")[0];
    Real Ni = 0;
    nucleate_ice::hf(temp, w_vlc, RH, Na, subgrid, // inputs
                     Ni);

    output.set("Ni", std::vector(1, Ni));
  });
}