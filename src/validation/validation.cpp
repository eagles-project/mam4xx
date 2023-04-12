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

} // namespace validation
} // namespace mam4
