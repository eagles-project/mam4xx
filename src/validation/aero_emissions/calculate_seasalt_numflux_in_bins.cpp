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
void calculate_seasalt_numflux_in_bins(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    // Ensemble parameters
    // Declare array of strings for input names
    std::string input_arrays[] = {"ncol", "srf_temp",  "ubot", "vbot",
                                  "zbot", "nsections", "z0"};

    // Iterate over input_arrays and error if not in input
    for (std::string name : input_arrays) {
      if (!input.has_array(name.c_str())) {
        std::cerr << "Required name for array: " << name << std::endl;
        exit(1);
      }
    }

    const auto surface_temp = input.get_array("srf_temp")[0];
    const auto u_bottom = input.get_array("ubot")[0];
    const auto v_bottom = input.get_array("vbot")[0];
    const auto z_bottom = input.get_array("zbot")[0];

    const int salt_nsection = mam4::aero_model_emissions::salt_nsection;

    Real fi[salt_nsection] = {0.0};

    mam4::aero_model_emissions::SeasaltEmissionsData data;
    mam4::aero_model_emissions::init_seasalt(data);

    mam4::aero_model_emissions::calculate_seasalt_numflux_in_bins(
        surface_temp, u_bottom, v_bottom, z_bottom, data.consta, data.constb,
        fi);

    std::vector<Real> fi_out;
    for (int i = 0; i < salt_nsection; ++i) {
      fi_out.push_back(fi[i]);
    }

    output.set("fi", fi_out);
  });
}
