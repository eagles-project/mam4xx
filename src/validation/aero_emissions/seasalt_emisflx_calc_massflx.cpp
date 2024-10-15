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
void seasalt_emisflx_calc_massflx(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    // Ensemble parameters
    // Declare array of strings for input names
    std::string input_arrays[] = {"ncol",
                                  "fi",
                                  "ocnfrc",
                                  "emis_scale",
                                  "flx_type",
                                  "nsections",
                                  "Dg",
                                  "rdry",
                                  "dns_aer_sst",
                                  "pi",
                                  "nslt",
                                  "nslt_om",
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

    const auto fi_ = input.get_array("fi");
    const Real ocean_frac = input.get_array("ocnfrc")[0];
    const Real emis_scalefactor = input.get_array("emis_scale")[0];

    constexpr int pcnst = mam4::pcnst;
    Real cflux[pcnst] = {0.0};

    Real fi[salt_nsection];
    for (int i = 0; i < salt_nsection; ++i) {
      fi[i] = fi_[i];
    }

    const mam4::aero_model_emissions::FluxType flux_type =
        mam4::aero_model_emissions::FluxType::MassFlux;

    mam4::aero_model_emissions::SeasaltEmissionsData data;
    mam4::aero_model_emissions::init_seasalt(data);
    mam4::aero_model_emissions::seasalt_emis_flux_calc(
        fi, ocean_frac, emis_scalefactor, flux_type, data, cflux);

    std::vector<Real> cflux_out;
    // NOTE: the only entries that are changed are (c++ indexing): 20, 25, 29
    // i.e.,
    // cflux[seasalt_indices[num_idx_append + ispec]]
    //      == cflux[seasalt_indices[0 + {0, 1, 2}]]
    //      == cflux[20, 25, 29]
    cflux_out.push_back(cflux[20]);
    cflux_out.push_back(cflux[25]);
    cflux_out.push_back(cflux[29]);

    output.set("cflx", cflux_out);
  });
}
