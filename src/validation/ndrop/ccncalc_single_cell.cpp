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

void ccncalc_single_cell(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    // number of vertical points.
    // validation test from standalone ndrop.
    const Real zero = 0;
    const int maxd_aspectype = 14;
    const int ntot_amode = 4;
    const int psat = ndrop_od::psat;

    const auto state_q = input.get_array("state_q");

    const Real tair = input.get_array("tair")[0];


    const auto qcldbrn_1d = input.get_array("qcldbrn");
    const auto qcldbrn_num = input.get_array("qcldbrn_num");

    Real qcldbrn[maxd_aspectype][ntot_amode] = {{zero}};

    const auto lspectype_amode_db = input.get_array("lspectype_amode");
    int lspectype_amode[maxd_aspectype][ntot_amode] = {};

    const auto lmassptr_amode_db = input.get_array("lmassptr_amode");
    int lmassptr_amode[maxd_aspectype][ntot_amode] = {};

    int count = 0;
    for (int i = 0; i < ntot_amode; ++i) {
      for (int j = 0; j < maxd_aspectype; ++j) {
        lspectype_amode[j][i] = lspectype_amode_db[count];
        lmassptr_amode[j][i] = lmassptr_amode_db[count];
        qcldbrn[j][i] =qcldbrn_1d[count];
        count++;
      }
    }
    const Real air_density = input.get_array("cs")[0];

    const auto specdens_amode_db = input.get_array("specdens_amode");
    const auto spechygro_db = input.get_array("spechygro");

    const auto specdens_amode = specdens_amode_db.data();
    const auto spechygro = spechygro_db.data();

    const auto voltonumbhi_amode_db = input.get_array("voltonumbhi_amode");
    const auto voltonumblo_amode_db = input.get_array("voltonumblo_amode");
    const auto numptr_amode_db = input.get_array("numptr_amode");
    const auto nspec_amode_db = input.get_array("nspec_amode");

    const auto voltonumbhi_amode = voltonumbhi_amode_db.data();
    const auto voltonumblo_amode = voltonumblo_amode_db.data();


    Real exp45logsig[AeroConfig::num_modes()], alogsig[AeroConfig::num_modes()],
    num2vol_ratio_min_nmodes[AeroConfig::num_modes()],
    num2vol_ratio_max_nmodes[AeroConfig::num_modes()] = {};

    Real aten = zero;

    ndrop_od::ndrop_int(exp45logsig, alogsig, aten,
                        num2vol_ratio_min_nmodes,  // voltonumbhi_amode
                        num2vol_ratio_max_nmodes); // voltonumblo_amode

    int numptr_amode[ntot_amode];
    int nspec_amode[ntot_amode];
    for (int i = 0; i < ntot_amode; ++i) {
      numptr_amode[i] = numptr_amode_db[i];
      nspec_amode[i] = nspec_amode_db[i];
    }
    std::vector<Real> ccn(psat,zero);
    ndrop_od::ccncalc(state_q.data(), tair, qcldbrn, qcldbrn_num.data(),
                      air_density, lspectype_amode, specdens_amode,
                      spechygro, lmassptr_amode, voltonumbhi_amode,
                      voltonumblo_amode, numptr_amode, nspec_amode,
                      exp45logsig, alogsig, ccn.data());

    output.set("ccn", ccn);

   }); 

}
