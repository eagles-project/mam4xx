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

    const Real pmid = input.get_array("pmid")[0];    // air pressure
    const Real temp = input.get_array("tair")[0]; // air temperature
    const Real cloud_fraction = input.get_array("cldn")[0];
    //updraft_vel_ice_nucleation
    const Real wbar = input.get_array("wbar")[0];
    const Real relhum = input.get_array("relhum")[0];
    const Real rhoair = input.get_array("rhoair")[0];
    const Real so4_num = input.get_array("so4_num")[0];
    const Real dst3_num = input.get_array("dst3_num")[0];

    const Real subgrid = input.get_array("subgrid")[0];

    this_nucleate_ice.nucleati(wbar, temp, pmid, relhum, cloud_fraction, rhoair,
                               so4_num, dst3_num, subgrid,
                               // outputs
                               nuci, onihf, oniimm, onidep, onimey);
    
    // using std::vector to match format from e3sm validation test.
    output.set("nuci", std::vector<Real>(1,nuci));
    output.set("onihf", std::vector<Real>(1,onihf));
    output.set("oniimm", std::vector<Real>(1,oniimm));
    output.set("onidep", std::vector<Real>(1,onidep));
    output.set("onimey", std::vector<Real>(1,onimey));
  });
}