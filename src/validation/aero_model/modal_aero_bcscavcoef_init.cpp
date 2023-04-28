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

void modal_aero_bcscavcoef_init(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    auto dgnum_amode = input.get_array("dgnum_amode");
    auto sigmag_amode = input.get_array("sigmag_amode");

    Real scavimptblnum[aero_model::nimptblgrow_total][AeroConfig::num_modes()] =
        {{}};
    Real scavimptblvol[aero_model::nimptblgrow_total][AeroConfig::num_modes()] =
        {{}};

    Real aerosol_dry_density[AeroConfig::num_modes()] = {};
    // Note: Original code uses the following aerosol densities.
    // sulfate, sulfate, dust, p-organic
    aerosol_dry_density[0] = mam4::mam4_density_so4;
    aerosol_dry_density[1] = mam4::mam4_density_so4;
    aerosol_dry_density[2] = mam4::mam4_density_dst;
    aerosol_dry_density[3] = mam4::mam4_density_pom;

    aero_model::modal_aero_bcscavcoef_init(
        dgnum_amode.data(), sigmag_amode.data(), aerosol_dry_density,
        scavimptblnum, scavimptblvol);
    // Note:  scavimptblnum and scavimptblvol were written in row-major order.
    std::vector<Real> values_scavimptblvol;
    std::vector<Real> values_scavimptblnum;
    for (int imode = 0; imode < AeroConfig::num_modes(); ++imode) {
      for (int m = 0; m < aero_model::nimptblgrow_total; ++m) {
        values_scavimptblnum.push_back(scavimptblnum[m][imode]);
        values_scavimptblvol.push_back(scavimptblvol[m][imode]);
      }
    }

    output.set("scavimptblnum", values_scavimptblnum);
    output.set("scavimptblvol", values_scavimptblvol);
  });
}