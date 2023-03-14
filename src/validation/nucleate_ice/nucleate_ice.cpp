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

    // get values from input when using either a 1D vector or a scalar
    auto get_value = [=](const std::string var_name) {
      if (input.has_array(var_name)) {
        return input.get_array(var_name)[0]; //
      } else if (input.has(var_name)) {
        return input.get(var_name);
      } else {
        std::cerr << "Required name: " << var_name << std::endl;
        exit(1);
      }
    };

    const Real pmid = get_value("pmid"); // air pressure
    const Real temp = get_value("tair"); // air temperature
    const Real cloud_fraction = get_value("cldn");
    // updraft_vel_ice_nucleation
    const Real wbar = get_value("wbar");
    const Real relhum = get_value("relhum");
    const Real rhoair = get_value("rhoair");
    const Real so4_num = get_value("so4_num");
    const Real dst3_num = get_value("dst3_num");

    const Real subgrid = get_value("subgrid");

    this_nucleate_ice.nucleati(wbar, temp, pmid, relhum, cloud_fraction, rhoair,
                               so4_num, dst3_num, subgrid,
                               // outputs
                               nuci, onihf, oniimm, onidep, onimey);

    // using std::vector to match format from e3sm validation test.
    output.set("nuci", std::vector<Real>(1, nuci));
    output.set("onihf", std::vector<Real>(1, onihf));
    output.set("oniimm", std::vector<Real>(1, oniimm));
    output.set("onidep", std::vector<Real>(1, onidep));
    output.set("onimey", std::vector<Real>(1, onimey));
  });
}