// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/coagulation.hpp>
#include <skywalker.hpp>
#include <validation.hpp>

using namespace skywalker;
using namespace mam4;

void coag_1subarea(Ensemble *ensemble) {

  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    if (!input.has_array("deltat")) {
      std::cerr << "Required name: "
                << "deltat" << std::endl;
      exit(1);
    }
    if (!input.has_array("pmid")) {
      std::cerr << "Required name: "
                << "pmid" << std::endl;
      exit(1);
    }
    if (!input.has_array("aircon")) {
      std::cerr << "Required name: "
                << "aircon" << std::endl;
      exit(1);
    }
    if (!input.has_array("dgn_a")) {
      std::cerr << "Required name: "
                << "dgn_a" << std::endl;
      exit(1);
    }
    if (!input.has_array("dgn_awet")) {
      std::cerr << "Required name: "
                << "dgn_awet" << std::endl;
      exit(1);
    }
    if (!input.has_array("wetdens")) {
      std::cerr << "Required name: "
                << "wetdens" << std::endl;
      exit(1);
    }
    if (!input.has_array("qnum_cur")) {
      std::cerr << "Required name: "
                << "qnum_cur" << std::endl;
      exit(1);
    }
    if (!input.has_array("qaer_cur")) {
      std::cerr << "Required name: "
                << "qaer_cur" << std::endl;
      exit(1);
    }

    // read input data
    auto deltat_f = input.get_array("deltat");
    auto temp_f = input.get_array("temp");
    auto pmid_f = input.get_array("pmid");
    auto aircon_f = input.get_array("aircon");
    auto dgn_a_f = input.get_array("dgn_a");
    auto dgn_awet_f = input.get_array("dgn_awet");
    auto wetdens_f = input.get_array("wetdens");
    auto qnum_cur_f = input.get_array("qnum_cur");
    auto qaer_cur_f = input.get_array("qaer_cur");

    const int num_modes = AeroConfig::num_modes();
    const int num_aero = AeroConfig::num_aerosol_ids();
    const int max_agepair = AeroConfig::max_agepair();
    Real qaer_cur_c[num_aero][num_modes];
    int n = 0;
    for (int imode = 0; imode < num_modes; ++imode) {
      for (int ispec = 0; ispec < num_aero; ++ispec) {
        qaer_cur_c[ispec][imode] = qaer_cur_f[n];
        n += 1;
      }
    }

    Real qaer_del_coag_out_c[num_aero][max_agepair] = {{0}};
    coagulation::mam_coag_1subarea(
        deltat_f[0], temp_f[0], pmid_f[0], aircon_f[0], dgn_a_f.data(),
        dgn_awet_f.data(), wetdens_f.data(), qnum_cur_f.data(), qaer_cur_c,
        qaer_del_coag_out_c);

    n = 0;
    for (int imode = 0; imode < num_modes; ++imode) {
      for (int ispec = 0; ispec < num_aero; ++ispec) {
        qaer_cur_f[n] = qaer_cur_c[ispec][imode];
        n += 1;
      }
    }

    std::vector<Real> qaer_del_coag_out_f(num_aero * max_agepair);
    n = 0;
    for (int imode = 0; imode < num_modes; ++imode) {
      for (int ispec = 0; ispec < max_agepair; ++ispec) {

        qaer_del_coag_out_f[n] = qaer_del_coag_out_c[ispec][imode];
        n += 1;
      }
    }

    output.set("qnum_cur", qnum_cur_f);
    output.set("qaer_cur", qaer_cur_f);
    output.set("qaer_del_coag_out", qaer_del_coag_out_f);
  });
}