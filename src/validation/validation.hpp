// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#ifndef HAERO_VALIDATION_HPP
#define HAERO_VALIDATION_HPP
#include <cfenv>
#include <haero/testing.hpp>
#include <mam4xx/aero_config.hpp>
#include <mam4xx/mo_photo.hpp>
#include <string>

namespace mam4 {

namespace testing {

using namespace haero::testing;

// these functions are defined in src/tests/testing.cpp and allow the creation
// of standalone objects that use "managed" ColumnViews
Prognostics create_prognostics(int num_levels);
Diagnostics create_diagnostics(int num_levels);
Tendencies create_tendencies(int num_levels);

} // namespace testing

namespace validation {

using View2D = typename DeviceType::view_2d<Real>;
using View3D = typename DeviceType::view_3d<Real>;

// forward functions from mam4::testing
using namespace mam4::testing;

/// Call this function to initialize a validation driver.
void initialize(int argc, char **argv);

/// initialize with FPEs enabled, provided via argument
constexpr int default_fpes = FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW;
void initialize(int argc, char **argv, const int fpes_);

/// Call this function to finalize a validation driver.
void finalize();

/// Given the name of a Skywalker input YAML file, determine the name of the
/// corresponding output Python module file.
/// @param [in] input_file The name of the Skywalker input YAML file.
std::string output_name(const std::string &input_file);

// Convert 1D std::vector to 2D Real array for aerosol mass/molar mixing ratios
void convert_vector_to_mass_mixing_ratios(
    const std::vector<Real> &vector_in,                                   // in
    Real values[AeroConfig::num_modes()][AeroConfig::num_aerosol_ids()]); // out

// Convert 2D Real array for aerosol mass/molar mixing ratios to 1D std::vector
void convert_mass_mixing_ratios_to_vector(
    const Real values[AeroConfig::num_modes()][AeroConfig::num_aerosol_ids()],
    std::vector<Real> &values_vector);

// Convert 1D Real Num Mode to 1D std::vector.
void convert_modal_array_to_vector(
    const Real values[AeroConfig::num_modes()], // in
    std::vector<Real> &values_vector);          // out

// Convert 1D std::vector to 1D Real array for num mode
void convert_vector_to_modal_array(const std::vector<Real> &vector_in,
                                   Real values[AeroConfig::num_modes()]);

using namespace mo_photo;
void create_synthetic_rsf_tab(View5D &rsf_tab, const int nw, const int nump,
                              const int numsza, const int numcolo3,
                              const int numalb, Real *synthetic_values);

// Convert 1D std::vector to 2D view_device
// copies data from 1D std::vector to a 2D view_host. Then, deep_copy to sync
// 1D std::vector is saved in column-major order(Fortran layout)
void convert_1d_vector_to_2d_view_device(const std::vector<Real> &pmid_db,
                                         const View2D &var_device);

// Convert 1D std::vector to 2D view_device
// copies data from 1D std::vector to a 2D view_host. Then, deep_copy to sync
// 1D std::vector is saved in column-major order(Fortran layout)
// original matrix is transposed to match layout in eamxx
void convert_1d_vector_to_transpose_2d_view_device(
    const std::vector<Real> &var_std, const View2D &var_device);

// Convert 2D view_device to 1D std::vector
// create a mirror view of 2d_view_device. Then, it copies data from mirror view
// to 1D std::vector using column-major order
void convert_2d_view_device_to_1d_vector(const View2D &var_device,
                                         std::vector<Real> &var_std);

// Convert 2D view_device to 1D std::vector
// create a mirror view of 2d_view_device. Then, it copies data from mirror view
// to 1D std::vector using column-major order
// device views is transposed matrix of original matrix
void convert_transpose_2d_view_device_to_1d_vector(const View2D &var_device,
                                                   std::vector<Real> &var_std);
// Convert 1D std::vector to 3D view_device
// copies data from 1D std::vector to a 3D view_host. Then, deep_copy to sync
// data to device
void convert_1d_vector_to_3d_view_device(const std::vector<Real> &pmid_db,
                                         const View3D &var_device);

// Convert 3D view_device to 1D std::vector
// create a mirror view of 3d_view_device. Then, it copies data from mirror view
// to 1D std::vector
void convert_3d_view_device_to_1d_vector(const View3D &var_device,
                                         std::vector<Real> &var_std);

void convert_1d_real_to_2d_view_device(const Real var_std[],
                                       const View2D &var_device);

} // namespace validation
} // namespace mam4

#endif
