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

void nucleate_ice_test(Ensemble *ensemble) {

  // We don't need any settings for this particular test.
  // Settings settings = ensemble->settings();

  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    // Fetch ensemble parameters

    NucleateIce this_nucleate_ice;
    // outputs
    const Real zero = 0;
    Real nuci, onihf, oniimm, onidep, onimey = zero;

    const Real pmid = input.get("pressure");    // air pressure
    const Real temp = input.get("temperature"); // air temperature
    const Real cloud_fraction = input.get("cloud_fraction");

    const Real wbar = input.get("updraft_vel_ice_nucleation");
    const Real relhum = input.get("relative_humidity");
    const Real rhoair = input.get("air_density");
    const Real so4_num = input.get("so4_num");
    const Real dst3_num = input.get("dst3_num");

    const Real subgrid = input.get("subgrid_in");

    this_nucleate_ice.nucleati(wbar, temp, pmid, relhum, cloud_fraction, rhoair,
                               so4_num, dst3_num, subgrid,
                               // outputs
                               nuci, onihf, oniimm, onidep, onimey);

    output.set("nuci", nuci);
    output.set("onihf", onihf);
    output.set("oniimm", oniimm);
    output.set("onidep", onidep);
    output.set("onimey", onimey);
  });
}