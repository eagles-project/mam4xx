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
    const int top_lev = ndrop::top_lev - 1;
    const int pver = ndrop::pver;
    const Real dtmix = input.get_array("dtmix")[0];
    const Real is_unact = input.get_array("is_unact")[0];

    const auto qold_db = input.get_array("qold");
    const auto src_db = input.get_array("src");
    const auto ekkp_db = input.get_array("ekkp");
    const auto ekkm_db = input.get_array("ekkm");
    const auto overlapp_db = input.get_array("overlapp");
    const auto overlapm_db = input.get_array("overlapm");

    std::vector<Real> qactold;
    if (input.has_array("qactold")) {
      qactold = input.get_array("qactold");
    }

    const Real zero = 0.0;

    Real q[pver];

    for (int k = top_lev; k < pver; k++) {
      // add logic for km1 and kp1 from fortran
      int kp1 = haero::min(k + 1, pver - 1);
      int km1 = haero::max(k - 1, top_lev);

      Real qold_km1 = qold_db[km1];
      Real qold_k = qold_db[k];
      Real qold_kp1 = qold_db[kp1];

      Real src = src_db[k];
      Real ekkm = ekkm_db[k];
      Real ekkp = ekkp_db[k];

      Real overlapm = overlapm_db[k];
      Real overlapp = overlapp_db[k];

      Real qactold_km1 = zero;
      Real qactold_kp1 = zero;
      if (is_unact) {
        qactold_km1 = qactold[km1];
        qactold_kp1 = qactold[kp1];
        ndrop::explmix(qold_km1, qold_k, qold_kp1, q[k], src, ekkp, ekkm,
                       overlapp, overlapm, dtmix, qactold_km1, qactold_kp1);
      } else {
        ndrop::explmix(qold_km1, qold_k, qold_kp1, q[k], src, ekkp, ekkm,
                       overlapp, overlapm, dtmix);
      }
    }

    std::vector<Real> qnew(pver, 0.0);
    for (int k = top_lev; k < pver; ++k) {
      qnew[k] = q[k];
    }
    output.set("qnew", qnew);
  });
}
