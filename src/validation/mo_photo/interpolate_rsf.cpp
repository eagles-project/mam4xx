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
void interpolate_rsf(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    // validation test from standalone mo_photo.
    const auto alb_in = input.get_array("alb_in");
    const auto sza_in = input.get_array("sza_in")[0];
    const auto p_in = input.get_array("p_in");
    const auto colo3_in = input.get_array("colo3_in");

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

    Real rsf_tab[nw][nump][numsza][numcolo3][numalb] = {};

    Real rsf[nw][nlev] = {};
    Real psum_l[nw] = {};
    Real psum_u[nw] = {};

    interpolate_rsf(alb_in.data(), sza_in, p_in.data(), colo3_in.data(),
                    pver, //  in
                    sza.data(), del_sza.data(), alb.data(), press.data(),
                    del_p.data(), colo3.data(), o3rat.data(), del_alb.data(),
                    del_o3rat.data(), etfphot.data(), rsf_tab,
                    rsf, // out
                    // work array
                    psum_l, psum_u);
  });
}
