// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#ifndef HAERO_VALIDATION_HPP
#define HAERO_VALIDATION_HPP
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

/// Call this function to finalize a validation driver.
void finalize();

/// Given the name of a Skywalker input YAML file, determine the name of the
/// corresponding output Python module file.
/// @param [in] input_file The name of the Skywalker input YAML file.
std::string output_name(const std::string &input_file);

/// The E3SM code has different indexing than mam4xx for the aerosol species in
// the accumulation, aitken, coarse, and primary carbon modes.
//  Therefore for validation proposes, we need to convert the index of the
//  aerosol species from the e3sm convention to that of mam4xx and vice versa.
//  We use these variables to convert inputs from e3sm
static constexpr int e3sm_to_mam4xx_aerosol_idx[4][7] = {
    // these arrays provide the index in the e3sm array that corresponds to the
    // species in the given position. an index of -1 means that species is not
    // present in the mam4xx array.

    // accumulation
    // e3sm   : SO4(1), POM(2), SOA(0), BC(3), DST(5), NaCl(4), MOM(6)
    // mam4xx : SOA(0), SO4(1), POM(2), BC(3), NaCl(4), DST(5), MOM(6)
    {1, 2, 0, 3, 5, 4, 6},

    // aitken
    // e3sm:    SO4(1), SOA(0), NaCl(2), MOM(3)
    // mam4xx : SOA(0), SO4(1), NaCl(2), MOM(3)
    {1, 0, 2, 3, -1, -1, -1},

    // coarse
    // e3sm: DST(5), NaCl(4), SO4(1), BC(3), POM(2), SOA(0), MOM(6)
    // mam4xx : SOA(0), SO4(1), POM(2), BC(3), NaCl(4), DST(5), MOM(6)
    {5, 4, 1, 3, 2, 0, 6},

    // primary carbon mode
    // e3sm : POM, BC, MOM
    // mam4xx : POM(0), BC(1), MOM(2)
    {0, 1, 2, -1, -1, -1, -1}};

// Because we need to compare with output arrays from e3sm we must save outputs
// from mam4xx using same indexing as e3sm. The following variable gives the
// index of the aerosol species using the e3sm indexing w.r.t mam4xx indexing.
//  we use these variables to convert outputs from mam4xx
static constexpr int mam4xx_to_e3sm_aerosol_idx[4][7] = {
    // these arrays provide the index in the mam4xx array that corresponds to
    // the species in the given position. an index of -1 means that species is
    // not present in the e3sm array.

    // accumulation
    // e3sm   : SO4(0), POM(1), SOA(2), BC(3), DST(4), NaCl(5), MOM(6)
    // mam4xx : SOA(2), SO4(0), POM(1), BC(3), NaCl(5), DST(4), MOM(6)
    {2, 0, 1, 3, 5, 4, 6},

    // aitken
    // e3sm:    SO4(0), SOA(1), NaCl(2), MOM(3)
    // mam4xx : SOA(1), SO4(0), NaCl(2), MOM(3)
    {1, 0, 2, 3, -1, -1, -1},

    // coarse
    // e3sm: DST(0), NaCl(1), SO4(2), BC(3), POM(4), SOA(5), MOM(6)
    // mam4xx : SOA(5), SO4(2), POM(4), BC(3), NaCl(1), DST(0), MOM(6)
    {5, 2, 4, 3, 1, 0, 6},

    // primary carbon mode
    // e3sm : POM, BC, MOM
    // mam4xx : POM(0), BC(1), MOM(2)
    {0, 1, 2, -1, -1, -1, -1}};

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
} // namespace validation
} // namespace mam4

#endif
