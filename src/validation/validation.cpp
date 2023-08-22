// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include "validation.hpp"

namespace mam4 {
namespace validation {

void initialize(int argc, char **argv) { Kokkos::initialize(argc, argv); }

void finalize() {
  testing::finalize();
  Kokkos::finalize();
}

std::string output_name(const std::string &input_file) {
  std::string output_file;
  size_t slash = input_file.find_last_of('/');
  size_t dot = input_file.find_last_of('.');
  if ((dot == std::string::npos) and (slash == std::string::npos)) {
    dot = input_file.length();
  }
  if (slash == std::string::npos) {
    slash = 0;
  } else {
    slash += 1;
    dot -= slash;
  }
  return std::string("mam4xx_") + input_file.substr(slash, dot) +
         std::string(".py");
}

void convert_vector_to_mass_mixing_ratios(
    const std::vector<Real> &vector_in,
    Real values[AeroConfig::num_modes()][AeroConfig::num_aerosol_ids()]) {
  int count = 0;
  for (int m = 0; m < AeroConfig::num_modes(); ++m) {
    for (int ispec = 0; ispec < AeroConfig::num_aerosol_ids(); ++ispec) {
      values[m][ispec] = vector_in[count];
      count++;
    }
  }
}

void convert_modal_array_to_vector(const Real values[AeroConfig::num_modes()],
                                   std::vector<Real> &values_vector) {
  for (int i = 0; i < AeroConfig::num_modes(); ++i)
    values_vector[i] = values[i];
}

void convert_vector_to_modal_array(const std::vector<Real> &vector_in,
                                   Real values[AeroConfig::num_modes()]) {
  for (int m = 0; m < AeroConfig::num_modes(); ++m)
    values[m] = vector_in[m];
}

void convert_mass_mixing_ratios_to_vector(
    const Real values[AeroConfig::num_modes()][AeroConfig::num_aerosol_ids()],
    std::vector<Real> &values_vector) {
  int count = 0;
  for (int m = 0; m < AeroConfig::num_modes(); ++m) {
    for (int ispec = 0; ispec < AeroConfig::num_aerosol_ids(); ++ispec) {
      values_vector[count] = values[m][ispec];
      count++;
    }
  }
}

void create_synthetic_rsf_tab(View5D &rsf_tab, const int nw, const int nump,
                              const int numsza, const int numcolo3,
                              const int numalb, Real *synthetic_values) {

  rsf_tab = View5D("rsf_tab", nw, nump, numsza, numcolo3, numalb);

  auto rsf_tab_1 = Kokkos::subview(rsf_tab, Kokkos::ALL(), 1, Kokkos::ALL(),
                                   Kokkos::ALL(), Kokkos::ALL());

  auto rsf_tab_2 = Kokkos::subview(rsf_tab, Kokkos::ALL(), Kokkos::ALL(), 6,
                                   Kokkos::ALL(), Kokkos::ALL());

  auto rsf_tab_3 = Kokkos::subview(rsf_tab, Kokkos::ALL(), Kokkos::ALL(),
                                   Kokkos::ALL(), 7, Kokkos::ALL());

  auto rsf_tab_4 = Kokkos::subview(rsf_tab, Kokkos::ALL(), Kokkos::ALL(),
                                   Kokkos::ALL(), Kokkos::ALL(), 3);

  auto rsf_tab_5 = Kokkos::subview(rsf_tab, 0, Kokkos::ALL(), Kokkos::ALL(),
                                   Kokkos::ALL(), Kokkos::ALL());

  auto rsf_tab_6 = Kokkos::subview(rsf_tab, 9, Kokkos::ALL(), Kokkos::ALL(),
                                   Kokkos::ALL(), Kokkos::ALL());

  Kokkos::deep_copy(rsf_tab, synthetic_values[0]);
  Kokkos::deep_copy(rsf_tab_1, synthetic_values[1]);
  Kokkos::deep_copy(rsf_tab_2, synthetic_values[2]);
  Kokkos::deep_copy(rsf_tab_3, synthetic_values[3]);
  Kokkos::deep_copy(rsf_tab_4, synthetic_values[4]);
  Kokkos::deep_copy(rsf_tab_5, synthetic_values[5]);
  Kokkos::deep_copy(rsf_tab_6, synthetic_values[6]);
}

void convert_1d_vector_to_2d_view_device(const std::vector<Real> &var_std,
                                         const View2D &var_device) {
  auto host = Kokkos::create_mirror_view(var_device);
  int count = 0;
  for (int d2 = 0; d2 < var_device.extent(1); ++d2) {
    for (int d1 = 0; d1 < var_device.extent(0); ++d1) {
      host(d1, d2) = var_std[count];
      count++;
    }
  }
  Kokkos::deep_copy(var_device, host);
}

void convert_2d_view_device_to_1d_vector(const View2D &var_device,
                                         std::vector<Real> &var_std) {
  auto host = Kokkos::create_mirror_view(var_device);
  Kokkos::deep_copy(host, var_device);
  int count = 0;
  for (int d2 = 0; d2 < var_device.extent(1); ++d2) {
    for (int d1 = 0; d1 < var_device.extent(0); ++d1) {
      var_std[count] = host(d1, d2);
      count++;
    }
  }
}

} // namespace validation
} // namespace mam4
