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
void calc_om_seasalt(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    // Ensemble parameters
    // Declare array of strings for input names
    std::string input_arrays[] = {"ncol",
                                  "mpoly_in",
                                  "mprot_in",
                                  "mlip_in",
                                  "nsections",
                                  "omfrac_max",
                                  "liter_to_m3",
                                  "OM_to_OC_in",
                                  "alpha_org",
                                  "mw_org",
                                  "dens_srf_org",
                                  "mw_carbon",
                                  "dens_vol_NaCl_in_seawat",
                                  "l_bub",
                                  "n_org_max",
                                  "n_org",
                                  "small_oceanorg"};

    // Iterate over input_arrays and error if not in input
    for (std::string name : input_arrays) {
      if (!input.has_array(name.c_str())) {
        std::cerr << "Required name for array: " << name << std::endl;
        exit(1);
      }
    }

    const int n_organic_species_max =
        mam4::aero_model_emissions::n_organic_species_max;
    const int salt_nsection = mam4::aero_model_emissions::salt_nsection;

    Real mass_frac_bub_section[n_organic_species_max][salt_nsection] = {0.0};
    Real om_ssa[salt_nsection] = {0.0};

    mam4::aero_model_emissions::SeasaltEmissionsData data;
    mam4::aero_model_emissions::init_seasalt(data);
    data.mpoly = input.get_array("mpoly_in")[0];
    data.mprot = input.get_array("mprot_in")[0];
    data.mlip = input.get_array("mlip_in")[0];

    mam4::aero_model_emissions::calc_org_matter_seasalt(
        data, mass_frac_bub_section, om_ssa);

    std::vector<Real> mfb_out;
    for (int i = 0; i < salt_nsection; ++i) {
      for (int j = 0; j < n_organic_species_max; ++j) {
        mfb_out.push_back(mass_frac_bub_section[j][i]);
      }
    }

    std::vector<Real> om_seasalt_out;
    for (int i = 0; i < salt_nsection; ++i) {
      om_seasalt_out.push_back(om_ssa[i]);
    }

    output.set("mass_frac_bub_section", mfb_out);
    output.set("om_ssa", om_seasalt_out);
  });
}
