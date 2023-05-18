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

void explmix(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    // number of vertical points.
    const Real zero = 0;
    const int maxd_aspectype = 14;
    const int ntot_amode = 4;
    const int nvars = 40;

    const int pver = input.get_array("pver")[0];
    const Real dt = input.get("dt");
    const Real dtmix = input.get_array("dtmix")[0];
    const Real is_unact = input.get_array("is_unact")[0];

    const auto qold_db = input.get_array("qold");
    const auto src_db = input.get_array("src");
    const auto ekkp_db = input.get_array("ekkp");
    const auto ekkm_db = input.get_array("ekkm");
    const auto overlapp_db = input.get_array("overlapp");
    const auto overlapm_db = input.get_array("overlapm");

    const int top_lev = 6;
    /*
        ColumnView qold = haero::testing::create_column_view(pver);
        ColumnView src = haero::testing::create_column_view(pver);
        ColumnView ekkp = haero::testing::create_column_view(pver);
        ColumnView ekkm = haero::testing::create_column_view(pver);
        ColumnView overlapp = haero::testing::create_column_view(pver);
        ColumnView overlapm = haero::testing::create_column_view(pver);
        for (int kk = 0; kk < pver; ++kk)
        {
          qold(kk) = qold_db[kk];
          src(kk) = src_db[kk];
          ekkp(kk) = ekkp_db[kk];
          ekkm(kk) = ekkm_db[kk];
          overlapp(kk) = overlapp_db[kk];
          overlapm(kk) = overlapm_db[kk];
        }
        */

    Real q[pver];

    for (int k = 1; k < pver - 1; k++) {

      Real qold_km1 = qold_db[k - 1];
      Real qold_k = qold_db[k];
      Real qold_kp1 = qold_db[k + 1];

      Real src = src_db[k];
      Real ek_km1 = ekkm_db[k]; // or -1 ??
      Real ek_kp1 = ekkp_db[k]; // or +1 ??

      Real overlap_km1 = overlapm_db[k]; // or -1 ??
      Real overlap_kp1 = overlapp_db[k]; // or +1 ??

      ndrop::explmix(qold_km1, qold_k, qold_kp1, q[k], src, ek_kp1, ek_km1,
                     overlap_kp1, overlap_km1, dt, is_unact);
    }

    for (int i = 0; i < pver; ++i) {
      output.set("qnew_" + std::to_string(i + 1), q[i]);
    }
  });
}
