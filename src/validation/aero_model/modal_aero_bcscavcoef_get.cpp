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

    const int ncol = int(input.get_array("ncol")[0]);

    auto scavimptblvol_vector = input.get_array("scavimptblvol");
    auto scavimptblnum_vector = input.get_array("scavimptblnum");

    Real scavimptblvol[aero_model::nimptblgrow_total][AeroConfig::num_modes()] =
        {{}};
    Real scavimptblnum[aero_model::nimptblgrow_total][AeroConfig::num_modes()] =
        {{}};

    // Note:  scavimptblvol_vector and scavimptblnum_vector were written in
    // row-major order.
    int count = 0;
    for (int imode = 0; imode < AeroConfig::num_modes(); ++imode) {
      for (int i = 0; i < aero_model::nimptblgrow_total; ++i) {
        scavimptblvol[i][imode] = scavimptblvol_vector[count];
        scavimptblnum[i][imode] = scavimptblnum_vector[count];
        count++;
      }
    }

    auto dgn_awet_vector = input.get_array("dgn_awet");
    std::vector<std::vector<Real>> dgn_awet;

    count = 0;
    for (int imode = 0; imode < AeroConfig::num_modes(); ++imode) {
      std::vector<Real> tmp;
      for (int icol = 0; icol < ncol; ++icol) {
        tmp.push_back(dgn_awet_vector[count]);
        count++;
      }
      dgn_awet.push_back(tmp);
    }

    auto dgnum_amode = input.get_array("dgnum_amode");
    auto imode = input.get_array("imode")[0] - 1;
    auto isprx_vector = input.get_array("isprx");

    std::vector<Real> scavcoefnum, scavcoefvol;
    scavcoefnum = std::vector(ncol, zero);
    scavcoefvol = std::vector(ncol, zero);

    for (int icol = 0; icol < ncol; ++icol) {
      aero_model::modal_aero_bcscavcoef_get(
          imode, isprx_vector[icol], dgn_awet[imode][icol], dgnum_amode[imode],
          scavimptblvol, scavimptblnum, scavcoefnum[icol], scavcoefvol[icol]);
    }

    output.set("scavcoefnum", scavcoefnum);
    output.set("scavcoefvol", scavcoefvol);
  });
}