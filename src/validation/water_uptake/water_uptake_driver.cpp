#include <iostream>
#include <mam4xx/water_uptake.hpp>
#include <skywalker.hpp>
#include <validation.hpp>

using namespace skywalker;
using namespace mam4;

// Parameterizations used by the drydep process.
// void water_uptake_wetdens(Ensemble *ensemble);
void water_uptake_wetdens(Ensemble *ensemble);
void find_real_solution(Ensemble *ensemble);
void makoh_quartic(Ensemble *ensemble);
void modal_aero_kohler(Ensemble *ensemble);
void modal_aero_water_uptake_rh_clearair(Ensemble *ensemble);
void modal_aero_water_uptake_wetaer(Ensemble *ensemble);
void modal_aero_water_uptake_dryaer(Ensemble *ensemble);
void modal_aero_water_uptake_dr(Ensemble *ensemble);
void modal_aero_water_uptake_dr_wetdens(Ensemble *ensemble);

void usage() {
  std::cerr << "aging_driver: a Skywalker driver for validating the "
               "MAM4 water_uptake parameterizations."
            << std::endl;
  std::cerr << "aging_driver: usage:" << std::endl;
  std::cerr << "aging_driver <input.yaml>" << std::endl;
  exit(1);
}

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
    if (func_name == "water_uptake_wetdens") {
      water_uptake_wetdens(ensemble);
    }
    if (func_name == "find_real_solution") {
      find_real_solution(ensemble);
    }
    if (func_name == "makoh_quartic") {
      makoh_quartic(ensemble);
    }
    if (func_name == "modal_aero_kohler") {
      modal_aero_kohler(ensemble);
    }
    if (func_name == "modal_aero_water_uptake_rh_clearair") {
      modal_aero_water_uptake_rh_clearair(ensemble);
    }
    if (func_name == "modal_aero_water_uptake_wetaer") {
      modal_aero_water_uptake_wetaer(ensemble);
    }
    if (func_name == "modal_aero_water_uptake_dryaer") {
      modal_aero_water_uptake_dryaer(ensemble);
    }
    if (func_name == "modal_aero_water_uptake_dr") {
      modal_aero_water_uptake_dr(ensemble);
    }
    if (func_name == "modal_aero_water_uptake_dr_wetdens") {
      modal_aero_water_uptake_dr_wetdens(ensemble);
    }
  } catch (std::exception &e) {
    std::cerr << argv[0] << ": Error: " << e.what() << std::endl;
  }

  // Write out a Python module.

  std::cout << argv[0] << ": writing " << output_file << std::endl;
  ensemble->write(output_file);

  //
  delete ensemble;
  validation::finalize();
};