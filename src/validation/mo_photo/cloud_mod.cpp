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
void cloud_mod(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    // number of vertical points.
    // validation test from standalone mo_photo.
    const auto zen_angle = input.get_array("zen_angle")[0];
    const auto clouds = input.get_array("clouds");
    const auto lwc = input.get_array("lwc");
    const auto delp = input.get_array("delp");
    const auto srf_alb = input.get_array("srf_alb")[0];
    // const auto  = input.get_array("");
    const Real zero = 0;

    std::vector<Real> eff_alb(pver, zero);
    std::vector<Real> cld_mult(pver, zero);

    cloud_mod(zen_angle, clouds.data(), lwc.data(), delp.data(),
              srf_alb, //  in
              eff_alb.data(), cld_mult.data());

    output.set("eff_alb", eff_alb);
    output.set("cld_mult", cld_mult);
  });
}
