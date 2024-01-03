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

void update_aod_spec(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    const auto scath2o = input.get_array("scath2o")[0];
    const auto sumhygro = input.get_array("sumhygro")[0];
    const auto absh2o = input.get_array("absh2o")[0];
    const auto sumscat = input.get_array("sumscat")[0];
    const auto sumabs = input.get_array("sumabs")[0];
    const auto hygro_s = input.get_array("hygro_s")[0];
    const auto palb = input.get_array("palb")[0];
    const auto dopaer = input.get_array("dopaer")[0];
    auto scat_s = input.get_array("scat_s")[0];
    auto abs_s = input.get_array("abs_s")[0];
    auto aod_s = input.get_array("aod_s")[0];

    update_aod_spec(scath2o,
                    absh2o, // in
                    sumhygro, sumscat,
                    sumabs, // in
                    hygro_s, palb,
                    dopaer, // in
                    scat_s, abs_s, aod_s);

    output.set("scat_s", std::vector<Real>(1, scat_s));
    output.set("abs_s", std::vector<Real>(1, abs_s));
    output.set("aod_s", std::vector<Real>(1, aod_s));
  });
}