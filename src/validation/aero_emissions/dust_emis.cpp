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
  //   std::string input_arrays[] = {"ncol", "lchnk", "dust_flux_in", "soil_erod_fact", "soil_erodibility", "dust_density", "pi", "soil_erod_threshold", "dust_dmt_vwr", "dust_indices", "dust_emis_sclfctr"};

  //   // Iterate over input_arrays and error if not in input
  //   for (std::string name : input_arrays) {
  //     if (!input.has_array(name.c_str())) {
  //       std::cerr << "Required name for array: " << name << std::endl;
  //       exit(1);
  //     }
  //   }

  //   const int salt_nsection = mam4::aero_model_emissions::salt_nsection;
  //   const int dust_nflux_in = mam4::aero_model_emissions::dust_nflux_in;

  //   const auto fi_ = input.get_array("fi");
  //   const Real dust_density = input.get_array("dust_density")[0];
  //   const auto dust_flux_in_ = input.get_array("dust_flux_in");
  //   const Real soil_erodibility = input.get_array("soil_erodibility")[0];

  //   Real cflux[salt_nsection] = {0.0};

  //   Real fi[salt_nsection];
  //   for (int i = 0; i < salt_nsection; ++i) {
  //     fi[i] = fi_[i];
  //   }

  //   Real dust_flux_in[dust_nflux_in];
  //   for (int i = 0; i < dust_nflux_in; ++i) {
  //     dust_flux_in[i] = dust_flux_in_[i];
  //   }

    // mam4::aero_model_emissions::SeasaltEmissionsData data;
    // mam4::aero_model_emissions::init_seasalt(data);
  //   mam4::aero_model_emissions::dust_emis(const Real dust_density,
              //  const Real (&dust_flux_in)[dust_nflux_in],
              //  //  out
              //  Real &soil_erodibility,
              //  //  inout
              //  Real (&cflux)[salt_nsection])

  //   std::vector<Real> cflux_out;
  //   // NOTE: the only entries that are changed are (c++ indexing):
  //   //       11, 16, 20, 13, 18, 26
  //   // see dust_emisflx_calc_numflx.cpp and dust_emisflx_calc_massflx.cpp
  //   // to get a deeper explanation on this
  //   cflux_out.push_back(cflux[11]);
  //   cflux_out.push_back(cflux[16]);
  //   cflux_out.push_back(cflux[20]);
  //   cflux_out.push_back(cflux[13]);
  //   cflux_out.push_back(cflux[18]);
  //   cflux_out.push_back(cflux[26]);

  //   output.set("cflx", cflux_out);
  });
}
