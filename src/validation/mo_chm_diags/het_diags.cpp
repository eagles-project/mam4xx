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
using namespace mo_chm_diags;

constexpr int gas_pcnst = gas_chemistry::gas_pcnst;

void het_diags(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
  
    const auto ncol = input.get_array("ncol")[0];
    const auto het_rates_in = input.get_array("het_rates");
    const auto mmr_in = input.get_array("mmr");
    const auto pdel_in = input.get_array("pdel");
    const auto wght = input.get_array("wght")[0];

    Real wrk_wd[gas_pcnst];
    Real sox_wk[gas_pcnst];
    Real adv_mass[gas_pcnst];

    mo_chm_diags::het_diags(het_rates, mmr, pdel, wght, wrk_wd, sox_wk, adv_mass); 

    output.set("wrk_wd_mm", wrk_wd);
    output.set("sox_wk", sox_wk);

  });
}
