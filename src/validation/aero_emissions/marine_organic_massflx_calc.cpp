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
void marine_organic_massflx_calc(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    // Ensemble parameters
    // Declare array of strings for input names
    std::string input_arrays[] = {"fi",
                                  "ocnfrc",
                                  "emis_scale",
                                  "om_ssa",
                                  "mass_frac_bub_section",
                                  "emit_this_mode",
                                  "nsections",
                                  "Dg",
                                  "dns_aer_sst",
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
    const int n_organic_species_max =
        mam4::aero_model_emissions::n_organic_species_max;
    const int organic_num_modes = mam4::aero_model_emissions::organic_num_modes;

    const auto fi_ = input.get_array("fi");
    const Real ocean_frac = input.get_array("ocnfrc")[0];
    const Real emis_scalefactor = input.get_array("emis_scale")[0];
    const auto om_seasalt_ = input.get_array("om_ssa");
    const auto mass_frac_bub_section_ =
        input.get_array("mass_frac_bub_section");
    const auto emit_this_mode_ = input.get_array("emit_this_mode");

    Real cflux[salt_nsection];

    Real fi[salt_nsection];
    Real om_seasalt[salt_nsection];
    Real mass_frac_bub_section[n_organic_species_max][salt_nsection];
    bool emit_this_mode[organic_num_modes];

    for (int i = 0; i < salt_nsection; ++i) {
      fi[i] = fi_[i];
      om_seasalt[i] = om_seasalt_[i];
    }

    int ll = 0;
    for (int i = 0; i < salt_nsection; ++i) {
      for (int j = 0; j < n_organic_species_max; ++j, ++ll) {
        mass_frac_bub_section[j][i] = mass_frac_bub_section_[ll];
      }
    }

    for (int i = 0; i < organic_num_modes; ++i) {
      emit_this_mode[i] = emit_this_mode_[i];
    }

    mam4::aero_model_emissions::SeasaltSectionData data;
    mam4::aero_model_emissions::init_seasalt(data);
    mam4::aero_model_emissions::calc_marine_organic_massflux(
        fi, ocean_frac, emis_scalefactor, om_seasalt, mass_frac_bub_section,
        emit_this_mode, data.rdry, cflux);

    std::vector<Real> cflux_out;

    // NOTE: the only entries that are changed are (c++ indexing): 12, 17, 29
    // i.e.,
    // cflux[seasalt_indices[nsalt + ispec]]
    //      == cflux[seasalt_indices[3 + {0, 1, 2}]]
    //      == cflux[12, 17, 29]
    cflux_out.push_back(cflux[12]);
    cflux_out.push_back(cflux[17]);
    cflux_out.push_back(cflux[29]);

    output.set("cflx", cflux_out);
  });
}
