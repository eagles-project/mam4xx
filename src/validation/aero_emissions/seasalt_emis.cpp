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
void seasalt_emis(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    // Ensemble parameters
    // Declare array of strings for input names
    std::string input_arrays[] = {"lchnk",        "ncol",         "fi",
                                  "ocnfrc",       "emis_scale",   "nsections",
                                  "num_flx_flag", "mass_flx_flag"};

    // Iterate over input_arrays and error if not in input
    for (std::string name : input_arrays) {
      if (!input.has_array(name.c_str())) {
        std::cerr << "Required name for array: " << name << std::endl;
        exit(1);
      }
    }

    const int salt_nsection = mam4::aero_model_emissions::salt_nsection;

    const auto fi_ = input.get_array("fi");
    const Real ocean_frac = input.get_array("ocnfrc")[0];
    const Real emis_scalefactor = input.get_array("emis_scale")[0];

    constexpr int pcnst = mam4::pcnst;
    Real cflux[pcnst] = {0.0};

    Real fi[salt_nsection];
    for (int i = 0; i < salt_nsection; ++i) {
      fi[i] = fi_[i];
    }

    mam4::aero_model_emissions::SeasaltEmissionsData data;
    mam4::aero_model_emissions::init_seasalt(data);
    mam4::aero_model_emissions::seasalt_emis(fi, ocean_frac, emis_scalefactor,
                                             data, cflux);

    std::vector<Real> cflux_out;
    // NOTE: the only entries that are changed are (c++ indexing):
    //       20, 25, 29, 22, 27, 35
    // see seasalt_emisflx_calc_numflx.cpp and seasalt_emisflx_calc_massflx.cpp
    // to get a deeper explanation on this
    cflux_out.push_back(cflux[20]);
    cflux_out.push_back(cflux[25]);
    cflux_out.push_back(cflux[29]);
    cflux_out.push_back(cflux[22]);
    cflux_out.push_back(cflux[27]);
    cflux_out.push_back(cflux[35]);

    output.set("cflx", cflux_out);
  });
}
