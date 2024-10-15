// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/aero_model_emissions.hpp>

#include <iostream>
#include <skywalker.hpp>
#include <validation.hpp>

void usage() {
  std::cerr
      << "set_aero_emissions_driver: a Skywalker driver for validating the "
         "MAM4 set_aero_emissions parameterizations."
      << std::endl;
  std::cerr << "set_aero_emissions_driver: usage:" << std::endl;
  std::cerr << "set_aero_emissions_driver <input.yaml>" << std::endl;
  exit(0);
}

using namespace skywalker;
using namespace mam4;

// Parameterizations used by the set_aero_emissions() process.
void calc_om_seasalt(Ensemble *ensemble);
void calculate_seasalt_numflux_in_bins(Ensemble *ensemble);
void marine_organic_emis(Ensemble *ensemble);
void marine_organic_massflx_calc(Ensemble *ensemble);
void marine_organic_numflx_calc(Ensemble *ensemble);
void seasalt_emisflx_calc_massflx(Ensemble *ensemble);
void seasalt_emisflx_calc_numflx(Ensemble *ensemble);
void seasalt_emis(Ensemble *ensemble);
void dust_emis(Ensemble *ensemble);
void aero_model_emissions_test(Ensemble *ensemble);

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
    if (func_name == "calc_om_seasalt") {
      calc_om_seasalt(ensemble);
    } else if (func_name == "calculate_seasalt_numflux_in_bins") {
      calculate_seasalt_numflux_in_bins(ensemble);
    } else if (func_name == "marine_organic_emis") {
      marine_organic_emis(ensemble);
    } else if (func_name == "marine_organic_massflx_calc") {
      marine_organic_massflx_calc(ensemble);
    } else if (func_name == "marine_organic_numflx_calc") {
      marine_organic_numflx_calc(ensemble);
    } else if (func_name == "seasalt_emisflx_calc_massflux") {
      seasalt_emisflx_calc_massflx(ensemble);
    } else if (func_name == "seasalt_emisflx_calc_numflux") {
      seasalt_emisflx_calc_numflx(ensemble);
    } else if (func_name == "seasalt_emis") {
      seasalt_emis(ensemble);
    } else if (func_name == "dust_emis") {
      dust_emis(ensemble);
    } else if (func_name == "aero_model_emissions") {
      aero_model_emissions_test(ensemble);
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
