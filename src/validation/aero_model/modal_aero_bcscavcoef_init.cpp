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
    auto specdens_amode = input.get_array("specdens_amode");
    auto lspectype_amode_1d = input.get_array("lspectype_amode");

    Real zero = 0;

    Real scavimptblnum[aero_model_od::nimptblgrow_total]
                      [AeroConfig::num_modes()] = {{zero}};
    Real scavimptblvol[aero_model_od::nimptblgrow_total]
                      [AeroConfig::num_modes()] = {{zero}};

    int lspectype_amode[aero_model_od::maxd_aspectype]
                       [AeroConfig::num_modes()] = {{0}};

    int count = 0;
    for (int imode = 0; imode < AeroConfig::num_modes(); ++imode) {
      for (int m = 0; m < aero_model_od::maxd_aspectype; ++m) {
        // Fortran to C++ indexing conversion
        lspectype_amode[m][imode] = lspectype_amode_1d[count] - 1;
        count += 1;
      }
    }

    aero_model_od::modal_aero_bcscavcoef_init(
        dgnum_amode.data(), sigmag_amode.data(), specdens_amode.data(),
        lspectype_amode, scavimptblnum, scavimptblvol);

    std::vector<Real> values_scavimptblvol;
    std::vector<Real> values_scavimptblnum;
    for (int imode = 0; imode < AeroConfig::num_modes(); ++imode) {
      for (int m = 0; m < aero_model_od::nimptblgrow_total; ++m) {
        values_scavimptblnum.push_back(scavimptblnum[m][imode]);
        values_scavimptblvol.push_back(scavimptblvol[m][imode]);
      }
    }

    output.set("scavimptblnum", values_scavimptblnum);
    output.set("scavimptblvol", values_scavimptblvol);
  });
}