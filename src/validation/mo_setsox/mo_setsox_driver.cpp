// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/mo_setsox.hpp>

#include <iostream>
#include <skywalker.hpp>
#include <validation.hpp>

void usage() {
  std::cerr << "mo_setsox_driver: a Skywalker driver for validating the "
               "MAM4 mo_setsox parameterizations."
            << std::endl;
  std::cerr << "mo_setsox_driver: usage:" << std::endl;
  std::cerr << "mo_setsox_driver <input.yaml>" << std::endl;
  exit(0);
}

using namespace skywalker;
using namespace mam4;

// Parameterizations used by the mo_setsox() process.
void setsox_test(Ensemble *ensemble);
void setsox_test_nlev(Ensemble *ensemble);
void calc_ph_values(Ensemble *ensemble);
void calc_sox_aqueous(Ensemble *ensemble);
void calc_ynetpos(Ensemble *ensemble);

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
    if (func_name == "setsox_test") {
      setsox_test(ensemble);
    } else if (func_name == "setsox_test_nlev") {
      setsox_test_nlev(ensemble);
    } else if (func_name == "calc_ph_values") {
      calc_ph_values(ensemble);
    } else if (func_name == "calc_sox_aqueous") {
      calc_sox_aqueous(ensemble);
    } else if (func_name == "calc_ynetpos") {
      calc_ynetpos(ensemble);
    } else {
      std::cerr << "Error: Function name '" << func_name
                << "' does not have an implemented test!" << std::endl;
      exit(1);
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
