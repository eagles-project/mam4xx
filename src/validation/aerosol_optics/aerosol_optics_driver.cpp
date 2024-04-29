// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/modal_aer_opt.hpp>

#include <iostream>
#include <skywalker.hpp>
#include <validation.hpp>

void usage() {
  std::cerr << "aerosol_optics_driver: a Skywalker driver for validating the "
               "MAM4 rename parameterizations."
            << std::endl;
  std::cerr << "aerosol_optics_driver: usage:" << std::endl;
  std::cerr << "aerosol_optics_driver <input.yaml>" << std::endl;
  exit(0);
}

using namespace skywalker;
using namespace mam4;

// Parameterizations used by the aerosol_optics() process.
void binterp(Ensemble *ensemble);
void calc_diag_spec(Ensemble *ensemble);
void calc_refin_complex(Ensemble *ensemble);
void calc_volc_ext(Ensemble *ensemble);
void modal_size_parameters(Ensemble *ensemble);
void modal_aero_sw(Ensemble *ensemble);
void modal_aero_lw(Ensemble *ensemble);
void calc_parameterized(Ensemble *ensemble);
void update_aod_spec(Ensemble *ensemble);
void aer_rad_props_lw(Ensemble *ensemble);
void aer_rad_props_sw(Ensemble *ensemble);
void volcanic_cmip_sw(Ensemble *ensemble);
void data_transfer_state_q_qqwc_to_prog(Ensemble *ensemble);

int main(int argc, char **argv) {
  if (argc == 1) {
    usage();
  }
  validation::initialize(argc, argv, validation::default_fpes);
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
    if (func_name == "binterp") {
      binterp(ensemble);
    } else if (func_name == "calc_diag_spec") {
      calc_diag_spec(ensemble);
    } else if (func_name == "calc_refin_complex") {
      calc_refin_complex(ensemble);
    } else if (func_name == "calc_volc_ext") {
      calc_volc_ext(ensemble);
    } else if (func_name == "modal_size_parameters") {
      modal_size_parameters(ensemble);
    } else if (func_name == "modal_aero_sw") {
      modal_aero_sw(ensemble);
    } else if (func_name == "modal_aero_lw") {
      modal_aero_lw(ensemble);
    } else if (func_name == "calc_parameterized") {
      calc_parameterized(ensemble);
    } else if (func_name == "update_aod_spec") {
      update_aod_spec(ensemble);
    } else if (func_name == "aer_rad_props_lw") {
      aer_rad_props_lw(ensemble);
    } else if (func_name == "aer_rad_props_sw") {
      aer_rad_props_sw(ensemble);
    } else if (func_name == "volcanic_cmip_sw") {
      volcanic_cmip_sw(ensemble);
    } else if (func_name == "data_transfer_state_q_qqwc_to_prog") {
      data_transfer_state_q_qqwc_to_prog(ensemble);
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
