// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/coagulation.hpp>
#include <skywalker.hpp>
#include <validation.hpp>

using namespace skywalker;
using namespace mam4;

void coag_aer_update(Ensemble *ensemble) {

  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    if (!input.has_array("deltat")) {
      std::cerr << "Required name: "
                << "deltat" << std::endl;
      exit(1);
    }
    if (!input.has_array("ybetaij3")) {
      std::cerr << "Required name: "
                << "ybetaij3" << std::endl;
      exit(1);
    }
    if (!input.has_array("qnum_tavg")) {
      std::cerr << "Required name: "
                << "qnum_tavg" << std::endl;
      exit(1);
    }
    if (!input.has_array("qaer_bgn")) {
      std::cerr << "Required name: "
                << "qaer_bgn" << std::endl;
      exit(1);
    }
    if (!input.has_array("qaer_end")) {
      std::cerr << "Required name: "
                << "qaer_end" << std::endl;
      exit(1);
    }

    auto deltat = input.get_array("deltat");
    auto ybetaij3 = input.get_array("ybetaij3");
    auto qnum_tavg = input.get_array("qnum_tavg");
    auto qaer_bgn_f = input.get_array("qaer_bgn");
    auto qaer_end_f = input.get_array("qaer_end");

    const int num_modes = AeroConfig::num_modes();
    const int num_aero = AeroConfig::num_aerosol_ids();
    const int max_agepair = AeroConfig::max_agepair();

    Real qaer_bgn_c[num_aero][num_modes];
    Real qaer_end_c[num_aero][num_modes];

    int n = 0;
    for (int ispec = 0; ispec < num_aero; ++ispec) {
      for (int imode = 0; imode < num_modes; ++imode) {
        qaer_bgn_c[ispec][imode] = qaer_bgn_f[n];
        qaer_end_c[ispec][imode] = qaer_end_f[n];
      }
    }

    Real qaer_del_coag_out_c[num_aero][max_agepair] = {{0}};
    coagulation::mam_coag_aer_update(ybetaij3.data(), deltat[0],
                                     qnum_tavg.data(), qaer_bgn_c, qaer_end_c,
                                     qaer_del_coag_out_c);

    n = 0;
    for (int ispec = 0; ispec < num_aero; ++ispec) {
      for (int imode = 0; imode < num_modes; ++imode) {
        qaer_end_f[n] = qaer_end_c[ispec][imode];
      }
    }

    n = 0;
    std::vector<Real> qaer_del_coag_out_f(num_aero * max_agepair);
    for (int ispec = 0; ispec < num_aero; ++ispec) {
      for (int imode = 0; imode < max_agepair; ++imode) {
        qaer_del_coag_out_f[n] = qaer_del_coag_out_c[ispec][imode];
      }
    }

    output.set("qaer_end", qaer_end_f);
    output.set("qaer_del_coag_out", qaer_del_coag_out_f);
  });
}