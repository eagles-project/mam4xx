// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/gas_chem.hpp>

#include <iostream>
#include <skywalker.hpp>
#include <validation.hpp>

// This driver computes the binary or ternary gas_chemistry for the given
// input.

void usage() {
  std::cerr << "gas_chem_driver: a Skywalker driver for validating the "
               "MAM4 gas_chemistry parameterizations."
            << std::endl;
  std::cerr << "gas_chem_driver: usage:" << std::endl;
  std::cerr << "gas_chem_driver <input.yaml>" << std::endl;
  exit(0);
}

using namespace skywalker;
using namespace mam4;

// Parameterizations used by the gas_chemistry process.
void indprd(Ensemble *ensemble);
void linmat(Ensemble *ensemble);
void nlnmat(Ensemble *ensemble);
void imp_prod_loss(Ensemble *ensemble);
void newton_raphson_iter(Ensemble *ensemble);
void imp_sol(Ensemble *ensemble);
void adjrxt(Ensemble *ensemble);
void setrxt(Ensemble *ensemble);

int main(int argc, char **argv) {
  if (argc == 1) {
    usage();
  }
  validation::initialize(argc, argv);
  std::string input_file = argv[1];
  std::string output_file = validation::output_name(input_file);
  std::cout << argv[0] << ": reading " << input_file << std::endl;

  // Load the ensemble. Any error encountered is fatal.
  Ensemble *ensemble = skywalker::load_ensemble(input_file, "mam4xx");

  // the settings.
  Settings settings = ensemble->settings();
  if (!settings.has("function")) {
    std::cerr << "No function specified in mam4xx.settings!" << std::endl;
    exit(1);
  }

  // Dispatch to the requested function.
  auto func_name = settings.get("function");
  try {
    if (func_name == "indprd") {
      indprd(ensemble);
    } else if (func_name == "linmat") {
      linmat(ensemble);
    } else if (func_name == "nlnmat") {
      nlnmat(ensemble);
    } else if (func_name == "imp_prod_loss") {
      imp_prod_loss(ensemble);
    } else if (func_name == "newton_raphson_iter") {
      newton_raphson_iter(ensemble);
    } else if (func_name == "imp_sol") {
      imp_sol(ensemble);
    } else if (func_name == "adjrxt") {
      adjrxt(ensemble);
    } else if (func_name == "setrxt") {
      setrxt(ensemble);
    }

  } catch (std::exception &e) {
    std::cerr << argv[0] << ": Error: " << e.what() << std::endl;
  }

  // Write out a Python module.
  std::cout << argv[0] << ": writing " << output_file << std::endl;
  ensemble->write(output_file);

  // Clean up.
  delete ensemble;
  validation::finalize();
}
