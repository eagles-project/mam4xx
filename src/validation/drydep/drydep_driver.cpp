#include <iostream>
#include <mam4xx/hetfrz.hpp>
#include <skywalker.hpp>
#include <validation.hpp>

using namespace skywalker;
using namespace mam4;

// Parameterizations used by the drydep process.
void gravit_settling_velocity(Ensemble *ensemble);
void modal_aero_turb_drydep_velocity(Ensemble *ensemble);
void modal_aero_depvel_part(Ensemble *ensemble);
void aero_model_drydep(Ensemble *ensemble);
void compute_tendencies(Ensemble *ensemble);
void modal_aero_gravit_settling_velocity(Ensemble *ensemble);
void schmidt_number(Ensemble *ensemble);
void slip_correction_factor(Ensemble *ensemble);
void air_kinematic_viscosity(Ensemble *ensemble);
void air_dynamic_viscosity(Ensemble *ensemble);
void radius_for_moment(Ensemble *ensemble);
void calcram(Ensemble *ensemble);

void usage() {
  std::cerr << "aging_driver: a Skywalker driver for validating the "
               "MAM4 drydep parameterizations."
            << std::endl;
  std::cerr << "aging_driver: usage:" << std::endl;
  std::cerr << "aging_driver <input.yaml>" << std::endl;
  exit(1);
}

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
    if (func_name == "gravit_settling_velocity") {
      gravit_settling_velocity(ensemble);
    } else if (func_name == "modal_aero_gravit_settling_velocity") {
      modal_aero_gravit_settling_velocity(ensemble);
    } else if (func_name == "calcram") {
      calcram(ensemble);
    } else if (func_name == "compute_tendencies") {
      compute_tendencies(ensemble);
    } else if (func_name == "aero_model_drydep") {
      aero_model_drydep(ensemble);
    } else if (func_name == "modal_aero_depvel_part") {
      modal_aero_depvel_part(ensemble);
    } else if (func_name == "modal_aero_turb_drydep_velocity") {
      modal_aero_turb_drydep_velocity(ensemble);
    } else if (func_name == "schmidt_number") {
      schmidt_number(ensemble);
    } else if (func_name == "slip_correction_factor") {
      slip_correction_factor(ensemble);
    } else if (func_name == "air_kinematic_viscosity") {
      air_kinematic_viscosity(ensemble);
    } else if (func_name == "air_dynamic_viscosity") {
      air_dynamic_viscosity(ensemble);
    } else if (func_name == "radius_for_moment") {
      radius_for_moment(ensemble);
    }
  } catch (std::exception &e) {
    std::cerr << argv[0] << ": Error: " << e.what() << std::endl;
  }

  // Write out a Python module.
  std::cout << argv[0] << ": writing " << output_file << std::endl;
  ensemble->write(output_file);

  validation::finalize(ensemble);
}
