// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/gasaerexch.hpp>

#include <iostream>
#include <skywalker.hpp>
#include <validation.hpp>

using namespace mam4;
using namespace skywalker;

void test_gasaerexch_uptkrates_1box1gas(std::unique_ptr<Ensemble> &ensemble);

void usage(const std::string &prog_name) {
  std::cerr << prog_name << ": usage:" << std::endl;
  std::cerr << prog_name << " <input.yaml>" << std::endl;
  exit(0);
}

int main(int argc, char **argv) {
  if (argc == 1) {
    usage((const char *)argv[0]);
  }
  validation::initialize(argc, argv);
  std::string input_file = argv[1];
  std::string output_file = validation::output_name(input_file);
  std::cout << argv[0] << ": reading " << input_file << std::endl;

  // Load the ensemble. Any error encountered is fatal.
  std::unique_ptr<Ensemble> ensemble(load_ensemble(input_file, "settings"));

  Settings settings = ensemble->settings();
  if (!settings.has("function")) {
    std::cerr << "Required to have 'function' for config file but not found "
              << std::endl;
    exit(1);
  }
  const std::string name = settings.get("function");
  if ((name != "gas_aer_uptkrates_1box1gas")) {
    std::cerr << "Invalid name: " << name << std::endl;
    std::cerr << "Currently the only valid name is: gas_aer_uptkrates_1box1gas"
              << std::endl;
    exit(1);
  }
  // Run the ensemble.
  try {
    if (name == "gas_aer_uptkrates_1box1gas") {
      test_gasaerexch_uptkrates_1box1gas(ensemble);
    }
    // Write out a Python module.
    std::cout << argv[0] << ": writing " << output_file << std::endl;
    ensemble->write(output_file);
  } catch (Exception &e) {
    std::cerr << ": Error: " << e.what() << std::endl;
  }
  validation::finalize();
}
