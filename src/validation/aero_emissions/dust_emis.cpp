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
void dust_emis(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    // Ensemble parameters
    // Declare array of strings for input names
    std::string input_arrays[] = {"ncol",
                                  "lchnk",
                                  "dust_flux_in",
                                  "soil_erod_fact",
                                  "soil_erodibility",
                                  "dust_density",
                                  "pi",
                                  "soil_erod_threshold",
                                  "dust_dmt_vwr",
                                  "dust_indices",
                                  "dust_emis_sclfctr"};

    // Iterate over input_arrays and error if not in input
    for (std::string name : input_arrays) {
      if (!input.has_array(name.c_str())) {
        std::cerr << "Required name for array: " << name << std::endl;
        exit(1);
      }
    }

    const int dust_nflux_in = mam4::aero_model_emissions::dust_nflux_in;
    const int dust_nbin = mam4::aero_model_emissions::dust_nbin;
    int dust_indices[4];
    auto dust_idx_ = mam4::aero_model_emissions::dust_indices;

    for (int i = 0; i < 4; ++i) {
      dust_indices[i] = dust_idx_[i];
    }

    const Real dust_density = input.get_array("dust_density")[0];
    const auto dust_flux_in_ = input.get_array("dust_flux_in");
    const auto dust_dmt_vwr_ = input.get_array("dust_dmt_vwr");
    Real soil_erodibility = input.get_array("soil_erodibility")[0];

    constexpr int pcnst = mam4::pcnst;
    Real cflux[pcnst] = {0.0};

    Real dust_flux_in[dust_nflux_in];
    for (int i = 0; i < dust_nflux_in; ++i) {
      dust_flux_in[i] = dust_flux_in_[i];
    }
    mam4::aero_model_emissions::DustEmissionsData data;
    for (int i = 0; i < dust_nbin; ++i) {
      data.dust_dmt_vwr[i] = dust_dmt_vwr_[i];
    }

    mam4::aero_model_emissions::dust_emis(dust_indices, dust_density,
                                          dust_flux_in, data, soil_erodibility,
                                          cflux);

    std::vector<Real> cflux_out;
    // NOTE: the only entries that are changed are (c++ indexing):
    //       19, 28, 22, 35
    // i.e.,
    // cflux[dust_indices[{0, 1}]]
    //      == cflux[19, 28]
    // AND:
    // cflux[dust_indices[{0, 1} + dust_nbin]]
    //      == cflux[dust_indices[{0, 1} + 2]]
    //      == cflux[22, 35]
    cflux_out.push_back(cflux[19]);
    cflux_out.push_back(cflux[28]);
    cflux_out.push_back(cflux[22]);
    cflux_out.push_back(cflux[35]);

    output.set("cflx", cflux_out);
    output.set("soil_erod", soil_erodibility);
  });
}
