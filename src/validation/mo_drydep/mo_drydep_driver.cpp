// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/mo_drydep.hpp>

#include <iostream>
#include <skywalker.hpp>
#include <validation.hpp>

void usage() {
  std::cerr << "mo_drydep_driver: a Skywalker driver for validating the "
               "MAM4 mo_drydep parameterizations."
            << std::endl;
  std::cerr << "mo_drydep_driver: usage:" << std::endl;
  std::cerr << "mo_drydep_driver <input.yaml>" << std::endl;
  exit(0);
}

using namespace skywalker;
using namespace mam4;

// Parameterizations used by the mo_drydep() process.
void calculate_aerodynamic_and_quasilaminar_resistance(Ensemble *ensemble);
void calculate_gas_drydep_vlc_and_flux(Ensemble *ensemble);
void calculate_obukhov_length(Ensemble *ensemble);
void calculate_resistance_rclx(Ensemble *ensemble);
void calculate_resistance_rgsx_and_rsmx(Ensemble *ensemble);
void calculate_resistance_rlux(Ensemble *ensemble);
void calculate_ustar_over_water(Ensemble *ensemble);
void calculate_ustar(Ensemble *ensemble);
void calculate_uustar(Ensemble *ensemble);
void drydep_xactive(Ensemble *ensemble);

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
    if (func_name == "calculate_aerodynamic_and_quasilaminar_resistance") {
      calculate_aerodynamic_and_quasilaminar_resistance(ensemble);
    } else if (func_name == "calculate_gas_drydep_vlc_and_flux") {
      calculate_gas_drydep_vlc_and_flux(ensemble);
    } else if (func_name == "calculate_obukhov_length") {
      calculate_obukhov_length(ensemble);
    } else if (func_name == "calculate_resistance_rclx") {
      calculate_resistance_rclx(ensemble);
    } else if (func_name == "calculate_resistance_rgsx_and_rsmx") {
      calculate_resistance_rgsx_and_rsmx(ensemble);
    } else if (func_name == "calculate_rlux") {
      calculate_resistance_rlux(ensemble);
    } else if (func_name == "calculate_ustar_over_water") {
      calculate_ustar_over_water(ensemble);
    } else if (func_name == "calculate_ustar") {
      calculate_ustar(ensemble);
    } else if (func_name == "calculate_uustar") {
      calculate_uustar(ensemble);
    } else if (func_name == "drydep_xactive") {
      drydep_xactive(ensemble);
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
