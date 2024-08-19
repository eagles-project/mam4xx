// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/mam4.hpp>

#include <mam4xx/aero_config.hpp>
#include <skywalker.hpp>
#include <validation.hpp>

using namespace skywalker;
using namespace mam4;
using namespace haero;
void marine_organic_emis(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    // Ensemble parameters
    // Declare array of strings for input names
    std::string input_arrays[] = {
        "lchnk",     "ncol",           "fi",    "ocnfrc", "emis_scale",
        "nsections", "emit_this_mode", "mpoly", "mprot",  "mlip", "cflx"};

    // Iterate over input_arrays and error if not in input
    for (std::string name : input_arrays) {
      if (!input.has_array(name.c_str())) {
        std::cerr << "Required name for array: " << name << std::endl;
        exit(1);
      }
    }

    const int salt_nsection = mam4::aero_model_emissions::salt_nsection;
    const int organic_num_modes = mam4::aero_model_emissions::organic_num_modes;

    const auto fi_ = input.get_array("fi");
    const auto ocean_frac = input.get_array("ocnfrc")[0];
    const auto emis_scalefactor = input.get_array("emis_scale")[0];
    const auto mpoly = input.get_array("mpoly")[0];
    const auto mprot = input.get_array("mprot")[0];
    const auto mlip = input.get_array("mlip")[0];
    const auto emit_this_mode_ = input.get_array("emit_this_mode");
    const auto cflux_ = input.get_array("cflx");

    // this test depends on the initial value of the entries that get calculated
    // thus, we have to pick out the initial values from the fortran data
    Real cflux[salt_nsection] = {0.0};
    cflux[13] = cflux_[22];
    cflux[18] = cflux_[27];

    Real fi[salt_nsection];
    for (int i = 0; i < salt_nsection; ++i) {
      fi[i] = fi_[i];
    }
    bool emit_this_mode[organic_num_modes];
    for (int i = 0; i < organic_num_modes; ++i) {
      emit_this_mode[i] = emit_this_mode_[i];
    }

    mam4::aero_model_emissions::SeasaltEmissionsData data;
    mam4::aero_model_emissions::init_seasalt(data);
    mam4::aero_model_emissions::marine_organic_emissions(
        fi, ocean_frac, emis_scalefactor, mpoly, mprot, mlip, data,
        emit_this_mode, cflux);

    std::vector<Real> cflux_out;

    // NOTE: the only entries that are changed are done in
    // calc_marine_organic_numflux() {cflux[13, 18]} and
    // calc_marine_organic_massflux() {cflux[12, 17, 29]}
    // the indices are (c++ indexing): 12, 13, 17, 18, 29
    // see marine_organic_massflx_calc.cpp and marine_organic_numflx_calc.cpp
    // for more information
    cflux_out.push_back(cflux[12]);
    cflux_out.push_back(cflux[13]);
    cflux_out.push_back(cflux[17]);
    cflux_out.push_back(cflux[18]);
    cflux_out.push_back(cflux[29]);

    output.set("cflx", cflux_out);
  });
}
