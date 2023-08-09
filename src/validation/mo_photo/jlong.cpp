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
using namespace haero;
using namespace mo_photo;

void jlong(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    // validation test from standalone mo_photo.
    const auto alb_in = input.get_array("alb_in");
    const auto sza_in = input.get_array("sza_in")[0];
    const auto p_in = input.get_array("p_in");
    const auto colo3_in = input.get_array("colo3_in");
    const auto t_in = input.get_array("t_in");

    // FIXME; ask for following variables
    const auto sza = input.get_array("sza");
    const auto del_sza = input.get_array("del_sza");
    const auto alb = input.get_array("alb");
    const auto press = input.get_array("press");
    const auto del_p = input.get_array("del_p");
    const auto colo3 = input.get_array("colo3");
    const auto o3rat = input.get_array("o3rat");
    const auto del_alb = input.get_array("del_alb");
    const auto del_o3rat = input.get_array("del_o3rat");
    const auto etfphot = input.get_array("etfphot");
    const auto rsf_tab_1d = input.get_array("rsf_tab");
    const auto xsqy_1d = input.get_array("xsqy");
    const auto prs = input.get_array("prs");
    const auto dprs = input.get_array("dprs");
    //

    View5D rsf_tab;
    View2D rsf("rsf",nw, nlev);
    View4D xsqy("xsqy",numj, nw, nt, np_xs);
    View2D xswk("xswk",numj, nw);

    // Real rsf[nw][nlev] = {};
    Real psum_l[nw] = {};
    Real psum_u[nw] = {};

    // Real xswk[numj][nw] = {};
    Real j_long = {};

    jlong(sza_in, alb_in.data(), p_in.data(), t_in.data(), colo3_in.data(),
          xsqy, sza.data(), del_sza.data(), alb.data(), press.data(),
          del_p.data(), colo3.data(), o3rat.data(), del_alb.data(),
          del_o3rat.data(), etfphot.data(),
          rsf_tab, // in
          prs.data(), dprs.data(),
          j_long, // output
          // work arrays
          rsf, xswk, psum_l, psum_u);
  });
}
