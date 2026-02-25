// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/mam4_amicphys.hpp>

#include <iostream>
#include <skywalker.hpp>
#include <validation.hpp>

void usage() {
  std::cerr
      << "amicphys_subareas_driver: a Skywalker driver for validating the "
         "MAM4 amicphys_subareas_driver parameterizations."
      << std::endl;
  std::cerr << "amicphys_subareas_driver: usage:" << std::endl;
  std::cerr << "amicphys_subareas_driver <input.yaml>" << std::endl;
  exit(0);
}

using namespace skywalker;
using namespace mam4;

// Parameterizations used by the amicphys_subareas_driver() process.
void compute_qsub_from_gcm_and_qsub_of_other_subarea(Ensemble *ensemble);
void form_gcm_of_gases_and_aerosols_from_subareas(Ensemble *ensemble);
void get_partition_factors(Ensemble *ensemble);
void set_subarea_gases_and_aerosols(Ensemble *ensemble);
void set_subarea_qmass_for_cldbrn_aerosols(Ensemble *ensemble);
void set_subarea_qmass_for_intrst_aerosols(Ensemble *ensemble);
void set_subarea_qnumb_for_cldbrn_aerosols(Ensemble *ensemble);
void set_subarea_qnumb_for_intrst_aerosols(Ensemble *ensemble);
void set_subarea_rh(Ensemble *ensemble);
void setup_subareas(Ensemble *ensemble);

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
    if (func_name == "compute_qsub_from_gcm_and_qsub_of_other_subarea") {
      compute_qsub_from_gcm_and_qsub_of_other_subarea(ensemble);
    } else if (func_name == "form_gcm_of_gases_and_aerosols_from_subareas") {
      form_gcm_of_gases_and_aerosols_from_subareas(ensemble);
    } else if (func_name == "get_partition_factors") {
      get_partition_factors(ensemble);
    } else if (func_name == "set_subarea_gases_and_aerosols") {
      set_subarea_gases_and_aerosols(ensemble);
    } else if (func_name == "set_subarea_qmass_for_cldbrn_aerosols") {
      set_subarea_qmass_for_cldbrn_aerosols(ensemble);
    } else if (func_name == "set_subarea_qmass_for_intrst_aerosols") {
      set_subarea_qmass_for_intrst_aerosols(ensemble);
    } else if (func_name == "set_subarea_qnumb_for_cldbrn_aerosols") {
      set_subarea_qnumb_for_cldbrn_aerosols(ensemble);
    } else if (func_name == "set_subarea_qnumb_for_intrst_aerosols") {
      set_subarea_qnumb_for_intrst_aerosols(ensemble);
    } else if (func_name == "set_subarea_rh") {
      set_subarea_rh(ensemble);
    } else if (func_name == "setup_subareas") {
      setup_subareas(ensemble);
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
  validation::finalize(ensemble);
}
