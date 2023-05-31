// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/convproc.hpp>

#include <iostream>
#include <skywalker.hpp>
#include <validation.hpp>

// This driver computes the binary or ternary nucleation rate for the given
// input.

void usage() {
  std::cerr << "convproc_driver: a Skywalker driver for validating the "
               "MAM4 convproc parameterizations."
            << std::endl;
  std::cerr << "convproc_driver: usage:" << std::endl;
  std::cerr << "convproc_driver <input.yaml>" << std::endl;
  exit(0);
}

using namespace skywalker;
using namespace mam4;

// Parameterizations used by the convproc process.
void update_tendency_final(Ensemble *ensemble);
void compute_column_tendency(Ensemble *ensemble);
void ma_resuspend_convproc(Ensemble *ensemble);
void ma_precpevap(Ensemble *ensemble);
void ma_precpprod(Ensemble *ensemble);

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
    if (func_name == "update_tendency_final") {
      update_tendency_final(ensemble);
    } else if (func_name == "compute_tendencies") {
      compute_column_tendency(ensemble);
    } else if (func_name == "ma_resuspend_convproc") {
      ma_resuspend_convproc(ensemble);
    } else if (func_name == "ma_precpevap") {
      ma_precpevap(ensemble);
    } else if (func_name == "ma_precpprod") {
      ma_precpprod(ensemble);
    } else {
      std::cerr << "Error: Test name not recognized:" << func_name << std::endl;
      exit(1);
    }
  } catch (std::exception &e) {
    std::cerr << argv[0] << ": Error: " << e.what() << std::endl;
    throw;
  }

  // Write out a Python module.
  std::cout << argv[0] << ": writing " << output_file << std::endl;
  ensemble->write(output_file);

  // Clean up.
  delete ensemble;
  validation::finalize();
}
