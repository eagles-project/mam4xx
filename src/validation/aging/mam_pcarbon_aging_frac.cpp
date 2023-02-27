// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/aging.hpp>
#include <skywalker.hpp>
#include <validation.hpp>

using namespace skywalker;
using namespace mam4;

void mam_pcarbon_aging_frac(Ensemble *ensemble) {

  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    // Ensemble parameters
    if (!input.has_array("dgn_a")) {
      std::cerr << "Required name: "
                << "dgn_a" << std::endl;
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
    if (!input.has_array("qaer_del_coag_in")) {
      std::cerr << "Required name: "
                << "qaer_del_coag_in" << std::endl;
      exit(1);
    }
    auto dgn_a_f = input.get_array("dgn_a");
    auto qaer_cur_f = input.get_array("qaer_cur");
    auto qaer_del_cond_f = input.get_array("qaer_del_cond");
    auto qaer_del_coag_in_f = input.get_array("qaer_del_coag_in");

    const int num_modes = AeroConfig::num_modes();
    const int num_aero = AeroConfig::num_aerosol_ids();

    Real qaer_cur_c[num_aero][num_modes];
    Real qaer_del_cond_c[num_aero][num_modes];
    Real qaer_del_coag_in_c[num_aero][AeroConfig::max_agepair()];

    int n = 0;
    for (int imode = 0; imode < num_modes; ++imode) {
      for (int ispec = 0; ispec < num_aero; ++ispec) {
        qaer_cur_c[ispec][imode] = qaer_cur_f[n];
        qaer_del_cond_c[ispec][imode] = qaer_del_cond_f[n];
        n += 1;
      }
    }

    n = 0;
    for (int imode = 0; imode < AeroConfig::max_agepair(); ++imode) {
      for (int ispec = 0; ispec < num_aero; ++ispec) {
        qaer_del_coag_in_c[ispec][imode] = qaer_del_coag_in_f[n];
        n += 1;
      }
    }

    Real xferfrac_pcage;
    Real frac_cond;
    Real frac_coag;

    aging::mam_pcarbon_aging_frac(dgn_a_f.data(), qaer_cur_c, qaer_del_cond_c,
                                  qaer_del_coag_in_c, xferfrac_pcage, frac_cond,
                                  frac_coag);

    n = 0;
    for (int imode = 0; imode < num_modes; ++imode) {
      for (int ispec = 0; ispec < num_aero; ++ispec) {
        qaer_cur_f[n] = qaer_cur_c[ispec][imode];
        qaer_del_cond_f[n] = qaer_del_cond_c[ispec][imode];
        n += 1;
      }
    }

    n = 0;
    for (int imode = 0; imode < AeroConfig::max_agepair(); ++imode) {
      for (int ispec = 0; ispec < num_aero; ++ispec) {
        qaer_del_coag_in_f[n] = qaer_del_coag_in_c[ispec][imode];
        n += 1;
      }
    }

    output.set("frac_cond", frac_cond);
    output.set("frac_coag", frac_coag);
    output.set("xferfrac_pcage", xferfrac_pcage);
    output.set("qaer_cur", qaer_cur_f);
    output.set("qaer_del_coag_in", qaer_del_coag_in_f);
    output.set("qaer_del_cond", qaer_del_cond_f);
  });
}
