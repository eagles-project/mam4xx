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
void marine_organic_numflx_calc(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    // Ensemble parameters
    // Declare array of strings for input names
    std::string input_arrays[] = {"ncol",
                                  "fi",
                                  "ocnfrc",
                                  "emis_scale",
                                  "om_ssa",
                                  "emit_this_mode",
                                  "nsections",
                                  "Dg",
                                  "om_num_ind",
                                  "om_num_modes",
                                  "seasalt_indices",
                                  "sst_sz_range_lo",
                                  "sst_sz_range_hi"};

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
    const Real ocean_frac = input.get_array("ocnfrc")[0];
    const Real emis_scalefactor = input.get_array("emis_scale")[0];
    const auto om_seasalt_ = input.get_array("om_ssa");
    const auto emit_this_mode_ = input.get_array("emit_this_mode");

    // TODO: need the input values for this
    Real cflux[salt_nsection];

    Real fi[salt_nsection];
    Real om_seasalt[salt_nsection];
    bool emit_this_mode[organic_num_modes];

    for (int i = 0; i < salt_nsection; ++i) {
      fi[i] = fi_[i];
      om_seasalt[i] = om_seasalt_[i];
    }

    for (int i = 0; i < organic_num_modes; ++i) {
      emit_this_mode[i] = emit_this_mode_[i];
    }

    mam4::aero_model_emissions::calc_marine_organic_numflux(
        fi, ocean_frac, emis_scalefactor, om_seasalt, emit_this_mode, cflux);

    std::vector<Real> cflux_out;

    // NOTE: the only entries that are changed are (c++ indexing): 13, 18
    // i.e.,
    // cflux[num_mode_idx];
    //    Where:
    //      num_mode_idx = seasalt_indices[nsalt + nsalt_om + om_num_idx],
    //      om_num_idx = organic_num_idx[{0, 1}] == {0, 1}
    //      om_num_idx = organic_num_idx[3 + 3 + {0, 1}]
    //      == cflux[seasalt_indices[{6, 7}]]
    //      == cflux[13, 18]
    cflux_out.push_back(cflux[13]);
    cflux_out.push_back(cflux[18]);

    output.set("cflx", cflux_out);
  });
}
