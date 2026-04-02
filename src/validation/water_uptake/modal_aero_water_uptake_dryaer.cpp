// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/mam4.hpp>
#include <validation.hpp>

using namespace skywalker;

void modal_aero_water_uptake_dryaer(Ensemble *ensemble) {
  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    EKAT_REQUIRE_MSG(input.has_array("dgncur_a"), "Required name: dgncur_a");
    EKAT_REQUIRE_MSG(input.has_array("state_q"), "Required name: state_q");

    auto dgncur_a = input.get_array("dgncur_a");
    auto state_q = input.get_array("state_q");

    int nspec_amode[mam4::AeroConfig::num_modes()];
    int lspectype_amode[mam4::water_uptake::maxd_aspectype]
                       [mam4::AeroConfig::num_modes()];
    Real specdens_amode[mam4::water_uptake::maxd_aspectype];
    Real spechygro[mam4::water_uptake::maxd_aspectype];

    mam4::water_uptake::get_e3sm_parameters(nspec_amode, lspectype_amode,
                                            specdens_amode, spechygro);

    std::vector<Real> hygro(mam4::AeroConfig::num_modes(), 0.0);
    std::vector<Real> naer(mam4::AeroConfig::num_modes(), 0.0);
    std::vector<Real> dryrad(mam4::AeroConfig::num_modes(), 0.0);
    std::vector<Real> dryvol(mam4::AeroConfig::num_modes(), 0.0);
    std::vector<Real> drymass(mam4::AeroConfig::num_modes(), 0.0);
    std::vector<Real> rhcrystal(mam4::AeroConfig::num_modes(), 0.0);
    std::vector<Real> rhdeliques(mam4::AeroConfig::num_modes(), 0.0);
    std::vector<Real> specdens_1(mam4::AeroConfig::num_modes(), 0.0);

    mam4::water_uptake::modal_aero_water_uptake_dryaer(
        nspec_amode, specdens_amode, spechygro, lspectype_amode, state_q.data(),
        dgncur_a.data(), hygro.data(), naer.data(), dryrad.data(),
        dryvol.data(), drymass.data(), rhcrystal.data(), rhdeliques.data(),
        specdens_1.data());

    output.set("hygro", hygro);
    output.set("naer", naer);
    output.set("dryrad", dryrad);
    output.set("dryvol", dryvol);
    output.set("drymass", drymass);
    output.set("rhcrystal", rhcrystal);
    output.set("rhdeliques", rhdeliques);
    output.set("specdens_1", specdens_1);
  });
}
