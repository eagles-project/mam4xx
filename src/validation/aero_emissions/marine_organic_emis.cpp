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
    std::string input_arrays[] = {"lchnk",         "ncol",       "fi",
                                  "ocnfrc",        "emis_scale", "nsections",
                                  "emit_this_mode"};

    // Iterate over input_arrays and error if not in input
    for (std::string name : input_arrays) {
      if (!input.has_array(name.c_str())) {
        std::cerr << "Required name for array: " << name << std::endl;
        exit(1);
      }
    }

    const int salt_nsection = mam4::aero_model_emissions::salt_nsection;

    std::vector<Real> fi{input.get_array("fi")};
    const auto ocean_frac = input.get_array("ocnfrc")[0];
    const auto emis_scalefactor = input.get_array("emis_scale")[0];
    const auto mpoly = input.get_array("mpoly")[0];
    const auto mprot = input.get_array("mprot")[0];
    const auto mlip = input.get_array("mlip")[0];
    Real cflux[salt_nsection] = {0.0};

    Real fi_in[salt_nsection];
    for (int i = 0; i < salt_nsection; ++i) {
      fi_in[i] = fi[i];
    }

    mam4::aero_model_emissions::SeasaltSectionData data;
    mam4::aero_model_emissions::init_seasalt(data);
    mam4::aero_model_emissions::marine_organic_emissions(
        fi_in, ocean_frac, emis_scalefactor, mpoly, mprot, mlip, data.rdry,
        cflux);

    std::vector<Real> cflux_out;
    for (int i = 0; i < salt_nsection; ++i) {
      cflux_out.push_back(cflux[i]);
    }

    output.set("cflx", cflux_out);
  });
}
