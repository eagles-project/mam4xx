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
using namespace modal_aer_opt;

void calc_parameterized(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {


    EKAT_REQUIRE_MSG(input.has_array("coef"), "Required name: coef");
    EKAT_REQUIRE_MSG(input.has_array("cheb_k"), "Required name: cheb_k");
    const auto coef = input.get_array("coef");
    const auto cheb_k = input.get_array("cheb_k");
    constexpr Real zero =0;

    Real para = zero;
    calc_parameterized(coef.data(), cheb_k.data(),para);
    output.set("para", std::vector<Real>(1, para));



  });
}    