// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <iostream>
#include <mam4xx/hetfrz.hpp>
#include <skywalker.hpp>
#include <validation.hpp>

using namespace skywalker;
using namespace mam4;

void calcualte_collkernel_sub(Ensemble *ensemble) {

  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    if (!input.has("latvap")) {
      std::cerr << "Required name: "
                << "latvap" << std::endl;
      exit(1);
    }

    if (!input.has("temperature")) {
      std::cerr << "Required name: "
                << "temperature" << std::endl;
      exit(1);
    }

    if (!input.has("Ktherm_air")) {
      std::cerr << "Required name: "
                << "Ktherm_air" << std::endl;
      exit(1);
    }

    if (!input.has("viscos_air")) {
      std::cerr << "Required name: "
                << "viscos_air" << std::endl;
      exit(1);
    }

    if (!input.has("Pr")) {
      std::cerr << "Required name: "
                << "Pr" << std::endl;
      exit(1);
    }

    if (!input.has("pressure")) {
      std::cerr << "Required name: "
                << "pressure" << std::endl;
      exit(1);
    }

    if (!input.has("rho_air")) {
      std::cerr << "Required name: "
                << "rho_air" << std::endl;
      exit(1);
    }

    if (!input.has("Ktherm")) {
      std::cerr << "Required name: "
                << "Ktherm" << std::endl;
      exit(1);
    }

    if (!input.has("r3lx")) {
      std::cerr << "Required name: "
                << "r3lx" << std::endl;
      exit(1);
    }

    if (!input.has("lam")) {
      std::cerr << "Required name: "
                << "lamb" << std::endl;
      exit(1);
    }

    if (!input.has("r_a")) {
      std::cerr << "Required name: "
                << "r_a" << std::endl;
      exit(1);
    }

    if (!input.has("Re")) {
      std::cerr << "Required name: "
                << "Re" << std::endl;
      exit(1);
    }

    if (!input.has("Tdiff")) {
      std::cerr << "Required name: "
                << "Tdiff" << std::endl;
      exit(1);
    }

    // Fetch input values.
    skywalker::Real latvap = input.get("latvap");
    skywalker::Real Ktherm_air = input.get("Ktherm_air");
    skywalker::Real viscos_air = input.get("viscos_air");
    skywalker::Real Pr = input.get("Pr");
    skywalker::Real temperature = input.get("temperature");
    skywalker::Real pressure = input.get("pressure");
    skywalker::Real rho_air = input.get("rho_air");
    skywalker::Real Ktherm = input.get("Ktherm");
    skywalker::Real r3lx = input.get("r3lx");
    skywalker::Real lambda = input.get("lam");
    skywalker::Real r_a = input.get("r_a");
    skywalker::Real Re = input.get("Re");
    skywalker::Real Tdiff = input.get("Tdiff");

    skywalker::Real K_total;

    hetfrz::calculate_collkernel_sub(temperature, pressure, rho_air, r3lx, r_a,
                                     lambda, latvap, viscos_air, Re, Ktherm_air,
                                     Ktherm, Pr, Tdiff, K_total);

    output.set("K_total", K_total);
  });
}