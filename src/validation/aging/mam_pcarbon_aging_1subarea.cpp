// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/aging.hpp>
#include <skywalker.hpp>
#include <validation.hpp>

using namespace skywalker;
using namespace mam4;

void mam_pcarbon_aging_1subarea(Ensemble *ensemble) {

  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    if (!input.has_array("dgn_a")) {
      std::cerr << "Required name: "
                << "dgn_a" << std::endl;
      exit(1);
    }
    if (!input.has_array("qnum_cur")) {
      std::cerr << "Required name: "
                << "qnum_cur" << std::endl;
      exit(1);
    }
    if (!input.has_array("qnum_del_cond")) {
      std::cerr << "Required name: "
                << "qnum_del_cond" << std::endl;
      exit(1);
    }
    if (!input.has_array("qnum_del_coag")) {
      std::cerr << "Required name: "
                << "qnum_del_coag" << std::endl;
      exit(1);
    }
    if (!input.has_array("qaer_cur")) {
      std::cerr << "Required name: "
                << "qaer_cur" << std::endl;
      exit(1);
    }
    if (!input.has_array("qaer_del_cond")) {
      std::cerr << "Required name: "
                << "qaer_del_cond" << std::endl;
      exit(1);
    }
    if (!input.has_array("qaer_del_coag")) {
      std::cerr << "Required name: "
                << "qaer_del_coag" << std::endl;
      exit(1);
    }
    if (!input.has_array("qaer_del_coag_in")) {
      std::cerr << "Required name: "
                << "qaer_del_coag_in" << std::endl;
      exit(1);
    }

    auto dgn_a_f = input.get_array("dgn_a");
    auto qnum_cur_f = input.get_array("qnum_cur");
    auto qnum_del_cond_f = input.get_array("qnum_del_cond");
    auto qnum_del_coag_f = input.get_array("qnum_del_coag");
    auto qaer_cur_f = input.get_array("qaer_cur");
    auto qaer_del_cond_f = input.get_array("qaer_del_cond");
    auto qaer_del_coag_f = input.get_array("qaer_del_coag");
    auto qaer_del_coag_in_f = input.get_array("qaer_del_coag_in");

    const int num_modes = AeroConfig::num_modes();
    const int num_aero = AeroConfig::num_aerosol_ids();

    Real qaer_cur_c[num_aero][num_modes];
    Real qaer_del_cond_c[num_aero][num_modes];
    Real qaer_del_coag_c[num_aero][num_modes];
    Real qaer_del_coag_in_c[num_aero][AeroConfig::max_agepair()];

    int n = 0;
    for (int imode = 0; imode < AeroConfig::max_agepair(); ++imode) {
      for (int ispec = 0; ispec < num_aero; ++ispec) {
        qaer_del_coag_in_c[ispec][imode] = qaer_del_coag_in_f[n];
        n += 1;
      }
    }

    n = 0;
    for (int imode = 0; imode < num_modes; ++imode) {
      for (int ispec = 0; ispec < num_aero; ++ispec) {
        qaer_cur_c[ispec][imode] = qaer_cur_f[n];
        qaer_del_cond_c[ispec][imode] = qaer_del_cond_f[n];
        qaer_del_coag_c[ispec][imode] = qaer_del_coag_f[n];
        n += 1;
      }
    }

    aging::mam_pcarbon_aging_1subarea(
        dgn_a_f.data(), qnum_cur_f.data(), qnum_del_cond_f.data(),
        qnum_del_coag_f.data(), qaer_cur_c, qaer_del_cond_c, qaer_del_coag_c,
        qaer_del_coag_in_c);

    n = 0;
    for (int imode = 0; imode < AeroConfig::max_agepair(); ++imode) {
      for (int ispec = 0; ispec < num_aero; ++ispec) {
        qaer_del_coag_in_f[n] = qaer_del_coag_in_c[ispec][imode];
        n += 1;
      }
    }

    n = 0;
    for (int imode = 0; imode < num_modes; ++imode) {
      for (int ispec = 0; ispec < num_aero; ++ispec) {
        qaer_cur_f[n] = qaer_cur_c[ispec][imode];
        qaer_del_cond_f[n] = qaer_del_cond_c[ispec][imode];
        qaer_del_coag_f[n] = qaer_del_coag_c[ispec][imode];
        n += 1;
      }
    }

    output.set("qnum_cur", qnum_cur_f);
    output.set("qnum_del_cond", qnum_del_cond_f);
    output.set("qnum_del_coag", qnum_del_coag_f);
    output.set("qaer_cur", qaer_cur_f);
    output.set("qaer_del_cond", qaer_del_cond_f);
    output.set("qaer_del_coag", qaer_del_coag_f);
    output.set("qaer_del_coag_in", qaer_del_coag_in_f);
  });
}
