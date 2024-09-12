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
        "lchnk",          "ncol",  "fi",    "ocnfrc", "emis_scale", "nsections",
        "emit_this_mode", "mpoly", "mprot", "mlip",   "cflx"};

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
    const auto emit_this_mode_ = input.get_array("emit_this_mode");
    const auto cflux_ = input.get_array("cflx");

    constexpr int pcnst = mam4::pcnst;
    Real cflux[pcnst];
    for (int i = 0; i < pcnst; ++i) {
      cflux[i] = cflux_[i];
    }

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
    data.mpoly = input.get_array("mpoly")[0];
    data.mprot = input.get_array("mprot")[0];
    data.mlip = input.get_array("mlip")[0];

    mam4::aero_model_emissions::marine_organic_emissions(
        fi, ocean_frac, emis_scalefactor, data, emit_this_mode, cflux);

    std::vector<Real> cflux_out;
    for (int i = 0; i < pcnst; ++i) {
      cflux_out.push_back(cflux[i]);
    }

    output.set("cflx", cflux_out);
  });
}
