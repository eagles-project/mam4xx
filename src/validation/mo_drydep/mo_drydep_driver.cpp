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
using DryDepData = mam4::seq_drydep::Data;

// Parameterizations used by the mo_drydep() process.
void calculate_aerodynamic_and_quasilaminar_resistance(Ensemble *ensemble);
void calculate_gas_drydep_vlc_and_flux(const DryDepData &data,
                                       Ensemble *ensemble);
void calculate_obukhov_length(Ensemble *ensemble);
void calculate_resistance_rclx(const DryDepData &data, Ensemble *ensemble);
void calculate_resistance_rgsx_and_rsmx(const DryDepData &data,
                                        Ensemble *ensemble);
void calculate_resistance_rlux(const DryDepData &data, Ensemble *ensemble);
void calculate_ustar_over_water(Ensemble *ensemble);
void calculate_ustar(const DryDepData &data, Ensemble *ensemble);
void calculate_uustar(const DryDepData &data, Ensemble *ensemble);
void drydep_xactive(const DryDepData &data, Ensemble *ensemble);

namespace {

// this function creates data views for dry deposition of tracers
mam4::seq_drydep::Data create_drydep_data() {
  using View1D = typename DeviceType::view_1d<Real>;
  using View2D = typename DeviceType::view_2d<Real>;
  using View1DHost = typename HostType::view_1d<Real>;
  using View2DHost = typename HostType::view_2d<Real>;

  using ViewBool1D = typename DeviceType::view_1d<bool>;
  using ViewInt1D = typename DeviceType::view_1d<int>;
  using ViewBool1DHost = typename HostType::view_1d<bool>;
  using ViewInt1DHost = typename HostType::view_1d<int>;

  // allocate the views
  constexpr int n_drydep = 3;
  constexpr int NSeas = mam4::seq_drydep::NSeas;
  constexpr int NLUse = mam4::seq_drydep::NLUse;
  constexpr int gas_pcnst = mam4::gas_chemistry::gas_pcnst;
  seq_drydep::Data data{
      View1D("drat", n_drydep),
      View1D("foxd", n_drydep),
      View2D("rac", NSeas, NLUse),
      View2D("rclo", NSeas, NLUse),
      View2D("rcls", NSeas, NLUse),
      View2D("rgso", NSeas, NLUse),
      View2D("rgss", NSeas, NLUse),
      View2D("ri", NSeas, NLUse),
      View2D("rlu", NSeas, NLUse),
      View2D("z0", NSeas, NLUse),
      ViewBool1D("has_dvel", gas_pcnst),
      ViewInt1D("map_dvel", gas_pcnst),
      -1, // FIXME: so2_ndx
  };

  // populate the views with data taken from mam4 validation test data

  Real drat_a[n_drydep] = {1.3740328303634228, 2.3332297194282963,
                           1.8857345177418792};
  View1DHost drat_h(drat_a, n_drydep);
  Kokkos::deep_copy(data.drat, drat_h);


  Real foxd_a[n_drydep] = {1.0, 1e-36, 1e-36};
  View1DHost foxd_h(foxd_a, n_drydep);
  Kokkos::deep_copy(data.foxd, foxd_h);

  // clang-format off
  Real rac_a[] = { 100.0,  100.0,  100.0,  100.0,  100.0,
                   200.0,  150.0,   10.0,   10.0,   50.0,
                   100.0,  100.0,  100.0,   10.0,   80.0,
                  2000.0, 1500.0, 1000.0, 1000.0, 1200.0,
                  2000.0, 2000.0, 2000.0, 2000.0, 2000.0,
                  2000.0, 1700.0, 1500.0, 1500.0, 1500.0,
                   1e-36,  1e-36,  1e-36,  1e-36,  1e-36,
                   1e-36,  1e-36,  1e-36,  1e-36,  1e-36,
                   300.0,  200.0,  100.0,   50.0,  200.0,
                   150.0,  120.0,   50.0,   10.0,   60.0,
                   200.0,  140.0,  120.0,  50.0,   120.0};
  mam4::validation::convert_1d_real_to_2d_view_device(rac_a, data.rac);


  Real rclo_a[] = {1e+36,   1e+36,  1e+36,  1e+36,  1e+36,
                   1000.0,  400.0, 1000.0, 1000.0, 1000.0,
                   1000.0,  400.0,  400.0, 1000.0,  500.0,
                   1000.0,  400.0,  400.0,  400.0,  500.0,
                   1000.0, 1000.0, 1000.0, 1500.0, 1500.0,
                   1000.0,  600.0,  600.0,  600.0,  700.0,
                    1e+36,  1e+36,  1e+36,  1e+36,  1e+36,
                    1e+36,  1e+36,  1e+36,  1e+36,  1e+36,
                   1000.0,  400.0,  800.0,  800.0,  600.0,
                   1000.0,  400.0,  600.0, 1000.0,  800.0,
                   1000.0,  400.0,  600.0,  800.0,  800.0};
  mam4::validation::convert_1d_real_to_2d_view_device(rclo_a, data.rclo);

  Real rcls_a[] = { 1e+36,  1e+36,  1e+36,  1e+36,  1e+36,
                   2000.0, 9000.0,  1e+36,  1e+36, 4000.0,
                   2000.0, 9000.0, 9000.0,  1e+36, 4000.0,
                   2000.0, 9000.0, 9000.0, 9000.0, 4000.0,
                   2000.0, 2000.0, 3000.0, 200.0,  2000.0,
                   2000.0, 4000.0, 6000.0, 400.0,  3000.0,
                    1e+36,  1e+36,  1e+36,  1e+36,  1e+36,
                    1e+36,  1e+36,  1e+36,  1e+36,  1e+36,
                   2500.0, 9000.0, 9000.0, 9000.0, 4000.0,
                   2000.0, 9000.0, 9000.0,  1e+36, 4000.0,
                   4000.0, 9000.0, 9000.0, 9000.0, 8000.0};
  mam4::validation::convert_1d_real_to_2d_view_device(rcls_a, data.rcls);

  Real rgso_a[] = { 300.0,  300.0,  300.0,  600.0,  300.0,
                    150.0,  150.0,  150.0, 3500.0,  150.0,
                    200.0,  200.0,  200.0, 3500.0,  200.0,
                    200.0,  200.0,  200.0, 3500.0,  200.0,
                    300.0,  300.0,  300.0, 3500.0,  300.0,
                   2000.0, 2000.0, 2000.0, 2000.0, 2000.0,
                    400.0,  400.0,  400.0,  400.0,  400.0,
                   1000.0,  800.0, 1000.0, 3500.0, 1000.0,
                    180.0,  180.0,  180.0, 3500.0,  180.0,
                    200.0,  200.0,  200.0, 3500.0,  200.0};
  mam4::validation::convert_1d_real_to_2d_view_device(rgso_a, data.rgso);

  Real rgss_a[] = { 400.0,  400.0,  400.0,  100.0,  500.0,
                    150.0,  200.0,  150.0,  100.0,  150.0,
                    350.0,  350.0,  350.0,  100.0,  350.0,
                    500.0,  500.0,  500.0,  100.0,  500.0,
                    500.0,  500.0,  500.0,  100.0,  500.0,
                    100.0,  100.0,  200.0,  100.0,  200.0,
                      1.0,    1.0,    1.0,    1.0,    1.0,
                   1000.0, 1000.0, 1000.0, 1000.0, 1000.0,
                      1.0,    1.0,    1.0,  100.0,    1.0,
                    220.0,  300.0,  200.0,  100.0,  250.0,
                    400.0,  400.0,  400.0,   50.0,  400.0};

  mam4::validation::convert_1d_real_to_2d_view_device(rgss_a, data.rgss);

  Real ri_a[] = {1e+36, 1e+36, 1e+36, 1e+36, 1e+36,
                  60.0, 1e+36, 1e+36, 1e+36, 120.0,
                 120.0, 1e+36, 1e+36, 1e+36, 240.0,
                  70.0, 1e+36, 1e+36, 1e+36, 140.0,
                 130.0, 250.0, 250.0, 400.0, 250.0,
                 100.0, 500.0, 500.0, 800.0, 190.0,
                 1e+36, 1e+36, 1e+36, 1e+36, 1e+36,
                 1e+36, 1e+36, 1e+36, 1e+36, 1e+36,
                  80.0, 1e+36, 1e+36, 1e+36, 160.0,
                 100.0, 1e+36, 1e+36, 1e+36, 200.0,
                 150.0, 1e+36, 1e+36, 1e+36, 300.0};
  mam4::validation::convert_1d_real_to_2d_view_device(ri_a, data.ri);

  Real rlu_a[] = { 1e+36,  1e+36,  1e+36,  1e+36,  1e+36,
                  2000.0, 9000.0,  1e+36,  1e+36, 4000.0,
                  2000.0, 9000.0, 9000.0,  1e+36, 4000.0,
                  2000.0, 9000.0, 9000.0,  1e+36, 4000.0,
                  2000.0, 4000.0, 4000.0, 6000.0, 2000.0,
                  2000.0, 8000.0, 8000.0, 9000.0, 3000.0,
                   1e+36,  1e+36,  1e+36,  1e+36,  1e+36,
                   1e+36,  1e+36,  1e+36,  1e+36,  1e+36,
                  2500.0, 9000.0, 9000.0, 9000.0, 4000.0,
                  2000.0, 9000.0, 9000.0, 9000.0, 4000.0,
                  4000.0, 9000.0, 9000.0, 9000.0, 8000.0};
  mam4::validation::convert_1d_real_to_2d_view_device(rlu_a, data.rlu);

  Real z0_a[] = {   1.0,    1.0,    1.0,    1.0,    1.0,
                   0.25,    0.1,  0.005,  0.001,   0.03,
                   0.05,   0.05,   0.05,  0.001,   0.02,
                    1.0,    1.0,    1.0,    1.0,    1.0,
                    1.0,    1.0,    1.0,    1.0,    1.0,
                    1.0,    1.0,    1.0,    1.0,    1.0,
                 0.0006, 0.0006, 0.0006, 0.0006, 0.0006,
                  0.002,  0.002,  0.002,  0.002,  0.002,
                   0.15,    0.1,    0.1,  0.001,   0.01,
                    0.1,   0.08,   0.02,  0.001,   0.03,
                    0.1,   0.08,   0.06,   0.04,   0.06};
  mam4::validation::convert_1d_real_to_2d_view_device(z0_a, data.z0);
  // clang-format on

  // initialize has_dvel and map_dvel views from namelist-like values similar
  // to those used by EAM to determine (0-based) species indices
  /*
  std::string drydep_list[] = {"H2O2", "H2SO4", "SO2"};
  std::string solsym[] = {"O3", "H2O2", "H2SO4", "SO2", "DMS", "SOAG",
                          "so4_a1", "pom_a1", "soa_a1", "bc_a1", "dst_a1",
                          "ncl_a1", "mom_a1", "num_a1", "so4_a2", "soa_a2",
                          "ncl_a2", "mom_a2", "num_a2", "dst_a3", "ncl_a3",
                          "so4_a3", "bc_a3", "pom_a3", "soa_a3", "mom_a3",
                          "num_a3", "pom_a4", "bc_a4", "mom_a4", "num_a4"};
   */

  // has_dvel maps species in solsym to true iff they appear in drydep_list
  bool has_dvel_a[gas_pcnst] = {}; // all false by default
  has_dvel_a[1] = true;            // H2O2
  has_dvel_a[2] = true;            // H2SO4
  has_dvel_a[3] = true;            // SO2
  ViewBool1DHost has_dvel_h(has_dvel_a, gas_pcnst);
  Kokkos::deep_copy(data.has_dvel, has_dvel_h);

  // for (int i = 0; i < gas_pcnst; ++i)
  // {
  //   if (has_dvel_a[i]){
  //     printf("has_dvel_a is true \n");
  //   } else {

  //   }printf("has_dvel_a is false \n");
  // }

  // map_dvel maps indices of species in solsym to those in drydep_list
  int map_dvel_a[gas_pcnst] = {};
  for (int i = 0; i < gas_pcnst; ++i) {
    map_dvel_a[i] = -1; // most indices unmapped
  }
  map_dvel_a[1] = 0; // H2O2
  map_dvel_a[2] = 1; // H2SO4
  map_dvel_a[3] = 2; // SO2
  ViewInt1DHost map_dvel_h(map_dvel_a, gas_pcnst);
  Kokkos::deep_copy(data.map_dvel, map_dvel_h);

  // index of "SO2" within solsym above
  data.so2_ndx = 3;

  return data;
}

} // namespace

int main(int argc, char **argv) {
  if (argc == 1) {
    usage();
  }
  validation::initialize(argc, argv);
  std::string input_file = argv[1];
  std::string output_file = validation::output_name(input_file);
  std::cout << argv[0] << ": reading " << input_file << std::endl;

  {
    auto data = create_drydep_data();

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
        calculate_gas_drydep_vlc_and_flux(data, ensemble);
      } else if (func_name == "calculate_obukhov_length") {
        calculate_obukhov_length(ensemble);
      } else if (func_name == "calculate_resistance_rclx") {
        calculate_resistance_rclx(data, ensemble);
      } else if (func_name == "calculate_resistance_rgsx_and_rsmx") {
        calculate_resistance_rgsx_and_rsmx(data, ensemble);
      } else if (func_name == "calculate_resistance_rlux") {
        calculate_resistance_rlux(data, ensemble);
      } else if (func_name == "calculate_ustar_over_water") {
        calculate_ustar_over_water(ensemble);
      } else if (func_name == "calculate_ustar") {
        calculate_ustar(data, ensemble);
      } else if (func_name == "calculate_uustar") {
        calculate_uustar(data, ensemble);
      } else if (func_name == "drydep_xactive") {
        drydep_xactive(data, ensemble);
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

    delete ensemble;
  }
  validation::finalize();
}
