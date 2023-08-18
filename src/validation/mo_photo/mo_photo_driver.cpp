// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/mo_photo.hpp>

#include <iostream>
#include <skywalker.hpp>
#include <validation.hpp>

void usage() {
  std::cerr << "mo_photo_driver: a Skywalker driver for validating the "
               "MAM4 rename parameterizations."
            << std::endl;
  std::cerr << "mo_photo_driver: usage:" << std::endl;
  std::cerr << "mo_photo_driver <input.yaml>" << std::endl;
  exit(0);
}

using namespace skywalker;
using namespace mam4;

// Parameterizations used by the mo_photo() process.
void cloud_mod(Ensemble *ensemble);
void find_index(Ensemble *ensemble);
void interpolate_rsf(Ensemble *ensemble);
void jlong(Ensemble *ensemble);
void calc_sum_wght(Ensemble *ensemble);
void table_photo(Ensemble *ensemble);
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
    if (func_name == "cloud_mod") {
      cloud_mod(ensemble);
    } else if (func_name == "find_index") {
      find_index(ensemble);
    } else if (func_name == "interpolate_rsf") {
      interpolate_rsf(ensemble);
    } else if (func_name == "jlong") {
      jlong(ensemble);
    } else if (func_name == "calc_sum_wght") {
      calc_sum_wght(ensemble);
    } else if (func_name == "table_photo") {
      table_photo(ensemble);
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
