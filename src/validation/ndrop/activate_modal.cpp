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

void activate_modal(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    const Real zero = 0;
    const int ntot_amode = AeroConfig::num_modes();
    const Real w_in = input.get_array("w_in")[0];
    const Real wmaxf = input.get_array("wmaxf")[0];
    const Real tair = input.get_array("tair")[0];
    const Real rhoair = input.get_array("rhoair")[0];
    std::vector<Real> na = input.get_array("na");
    const auto volume = input.get_array("volume");
    const auto hygro = input.get_array("hygro");

    Real exp45logsig[AeroConfig::num_modes()], alogsig[AeroConfig::num_modes()],
        num2vol_ratio_min_nmodes[AeroConfig::num_modes()],
        num2vol_ratio_max_nmodes[AeroConfig::num_modes()] = {};

    Real aten = zero;

    ndrop::ndrop_init(exp45logsig, alogsig, aten,
                      num2vol_ratio_min_nmodes,  // voltonumbhi_amode
                      num2vol_ratio_max_nmodes); // voltonumblo_amode

    std::vector<Real> fn(ntot_amode, zero), fm(ntot_amode, zero),
        fluxn(ntot_amode, zero), fluxm(ntot_amode, zero);
    Real flux_fullact = zero;

    ndrop::activate_modal(w_in, wmaxf, tair, rhoair, na.data(), volume.data(),
                          hygro.data(), exp45logsig, alogsig, aten, fn.data(),
                          fm.data(), fluxn.data(), fluxm.data(), flux_fullact);

    output.set("flux_fullact", flux_fullact);
    output.set("fn", fn);
    output.set("fm", fm);
    output.set("fluxn", fluxn);
    output.set("fluxm", fluxm);
  });
}
