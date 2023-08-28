// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/wet_dep.hpp>

#include <iostream>
#include <skywalker.hpp>
#include <validation.hpp>

using namespace mam4;
using namespace skywalker;

void test_wetdep_clddiag(std::unique_ptr<Ensemble> &ensemble);
void test_update_scavenging(std::unique_ptr<Ensemble> &ensemble);
void test_wetdep_prevap(std::unique_ptr<Ensemble> &ensemble);
void test_wetdep_resusp_nonlinear(std::unique_ptr<Ensemble> &ensemble);
void test_wetdep_resusp_noprecip(std::unique_ptr<Ensemble> &ensemble);
void test_wetdep_scavenging(std::unique_ptr<Ensemble> &ensemble);
void test_compute_evap_frac(std::unique_ptr<Ensemble> &ensemble);
void test_rain_mix_ratio(std::unique_ptr<Ensemble> &ensemble);
void test_calculate_cloudy_volume(std::unique_ptr<Ensemble> &ensemble);
void test_local_precip_production(std::unique_ptr<Ensemble> &ensemble);
void test_wetdep_resusp(std::unique_ptr<Ensemble> &ensemble);
void test_wetdepa_v2(std::unique_ptr<Ensemble> &ensemble);
void test_compute_tendencies(std::unique_ptr<Ensemble> &ensemble);

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
  std::unique_ptr<Ensemble> ensemble(load_ensemble(input_file, "mam4xx"));

  Settings settings = ensemble->settings();
  if (!settings.has("function")) {
    std::cerr << "Required to have 'function' for config file but not found "
              << std::endl;
    exit(1);
  }
  const std::string name = settings.get("function");
  if (name != "wetdep_clddiag" && name != "update_scavenging" &&
      name != "wetdep_prevap" && name != "wetdep_resusp_nonlinear" &&
      name != "wetdep_resusp_noprecip" && name != "wetdep_scavenging" &&
      name != "compute_evap_frac" && name != "rain_mix_ratio" &&
      name != "calculate_cloudy_volume" && name != "local_precip_production" &&
      name != "wetdep_resusp" && name != "wetdepa_v2" &&
      name != "compute_tendencies") {
    std::cerr << "Invalid name: " << name << std::endl;
    std::cerr << "Currently the only valid name are: "
              << "wetdep_clddiag, update_scavenging, wetdep_prevap, "
                 "wetdep_resusp_nonlinear, wetdep_resusp_noprecip, "
                 "wetdep_scavenging, compute_evap_frac, rain_mix_ratio, "
                 "calculate_cloudy_volume, local_precip_production, "
                 "wetdep_resusp, wetdepa_v2 compute_tendencies."
              << std::endl;
    exit(1);
  }
  // Run the ensemble.
  try {
    if (name == "wetdep_clddiag") {
      test_wetdep_clddiag(ensemble);
    } else if (name == "update_scavenging") {
      test_update_scavenging(ensemble);
    } else if (name == "wetdep_prevap") {
      test_wetdep_prevap(ensemble);
    } else if (name == "wetdep_resusp_nonlinear") {
      test_wetdep_resusp_nonlinear(ensemble);
    } else if (name == "wetdep_resusp_noprecip") {
      test_wetdep_resusp_noprecip(ensemble);
    } else if (name == "wetdep_scavenging") {
      test_wetdep_scavenging(ensemble);
    } else if (name == "compute_evap_frac") {
      test_compute_evap_frac(ensemble);
    } else if (name == "rain_mix_ratio") {
      test_rain_mix_ratio(ensemble);
    } else if (name == "calculate_cloudy_volume") {
      test_calculate_cloudy_volume(ensemble);
    } else if (name == "local_precip_production") {
      test_local_precip_production(ensemble);
    } else if (name == "wetdep_resusp") {
      test_wetdep_resusp(ensemble);
    } else if (name == "wetdepa_v2") {
      test_wetdepa_v2(ensemble);
    } else if (name == "compute_tendencies") {
      test_compute_tendencies(ensemble);
    }
    // Write out a Python module.
    std::cout << argv[0] << ": writing " << output_file << std::endl;
    ensemble->write(output_file);
  } catch (Exception &e) {
    std::cerr << ": Error: " << e.what() << std::endl;
  }
  validation::finalize();
}
