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

void modal_aero_bcscavcoef_get(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    const Real zero = 0;

    auto scavimptblvol_vector = input.get_array("scavimptblvol");
    auto scavimptblnum_vector = input.get_array("scavimptblnum");

    Real scavimptblvol[aero_model_od::nimptblgrow_total]
                      [AeroConfig::num_modes()] = {{zero}};
    Real scavimptblnum[aero_model_od::nimptblgrow_total]
                      [AeroConfig::num_modes()] = {{zero}};

    int count = 0;
    for (int i = 0; i < aero_model_od::nimptblgrow_total; ++i) {
      for (int imode = 0; AeroConfig::num_modes() < count; ++imode)
      {
        scavimptblvol[i][imode] = scavimptblvol_vector[count];
        scavimptblnum[i][imode] = scavimptblnum_vector[count];
        count+= 1;
      }
    }

    auto dgnum_amode = input.get_array("dgnum_amode");
    auto imode = input.get_array("imode")[0] - 1;
    bool isprx_kk = bool(input.get_array("isprx")[0]);
    auto dgn_awet_imode_kk = input.get_array("dgn_awet")[imode];
    Real scavcoefnum_kk = zero;
    Real scavcoefvol_kk = zero;

    aero_model_od::modal_aero_bcscavcoef_get(
        imode, isprx_kk, dgn_awet_imode_kk, dgnum_amode[imode], scavimptblvol,
        scavimptblnum, scavcoefnum_kk, scavcoefvol_kk);

    output.set("scavcoefnum", std::vector(1, scavcoefnum_kk));
    output.set("scavratevol", std::vector(1, scavcoefvol_kk));
  });
}