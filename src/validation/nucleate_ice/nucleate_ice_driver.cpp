// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/nucleate_ice.hpp>

#include <iostream>
#include <skywalker.hpp>
#include <validation.hpp>

// This driver computes the binary or ternary nucleate_ice for the given
// input.

void usage() {
  std::cerr << "nucleate_ice_driver: a Skywalker driver for validating the "
               "MAM4 nucleate_ice parameterizations."
            << std::endl;
  std::cerr << "nucleate_ice_driver: usage:" << std::endl;
  std::cerr << "nucleate_ice_driver <input.yaml>" << std::endl;
  exit(0);
}

using namespace skywalker;
using namespace mam4;

// Parameterizations used by the nucleate_ice process.
void compute_tendencies(Ensemble *ensemble);
void nucleate_ice_test(Ensemble *ensemble);
void hf(Ensemble *ensemble);
void hetero(Ensemble *ensemble);

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
    if (func_name == "nucleate_ice_cam_calc") {
      compute_tendencies(ensemble);
    } else if (func_name == "nucleati") {
      nucleate_ice_test(ensemble);
    } else if (func_name == "hf") {
      hf(ensemble);
    } else if (func_name == "hetero") {
      hetero(ensemble);
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
