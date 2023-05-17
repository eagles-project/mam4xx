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

void loadaer(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    // number of vertical points.
    const Real zero = 0;
    const int ntot_amode = 4;
    const int maxd_aspectype = 14;

    const auto state_q = input.get_array("state_q");
    const Real air_density = input.get_array("cs")[0];
    const Real phase = input.get_array("phase")[0];
    const auto qcldbrn1d_1d = input.get_array("qcldbrn1d");
    const auto qcldbrn1d_num = input.get_array("qcldbrn1d_num");

    // const auto  = input.get_array("");
    std::vector<Real> naerosol(ntot_amode, zero), vaerosol(ntot_amode, zero),
        hygro(ntot_amode, zero);
    // from ccncalc_input_ts_1400.yaml
    const int nspec_amode[ntot_amode] = {7, 4, 7, 3};
    const int lspectype_amode_1d[ntot_amode * maxd_aspectype] = {
        1, 4, 5, 6, 8, 7, 9, 0, 0, 0, 0, 0, 0, 0, 1, 5, 7, 9, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 7, 1, 6, 4, 5, 9, 0, 0, 0,
        0, 0, 0, 0, 4, 6, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    const int lmassptr_amode_1d[ntot_amode * maxd_aspectype] = {
        16, 17, 18, 19, 20, 21, 22, 0, 0, 0,  0,  0,  0,  0,  24, 25, 26, 27, 0,
        0,  0,  0,  0,  0,  0,  0,  0, 0, 29, 30, 31, 32, 33, 34, 35, 0,  0,  0,
        0,  0,  0,  0,  37, 38, 39, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0};

    int lspectype_amode[maxd_aspectype][ntot_amode] = {};
    int lmassptr_amode[maxd_aspectype][ntot_amode] = {};
    Real qcldbrn1d[maxd_aspectype][ntot_amode] = {};

    int count = 0;
    for (int i = 0; i < ntot_amode; ++i) {
      for (int j = 0; j < maxd_aspectype; ++j) {
        lspectype_amode[j][i] = lspectype_amode_1d[count];
        lmassptr_amode[j][i] = lmassptr_amode_1d[count];
        qcldbrn1d[j][i] = qcldbrn1d_1d[count];
        count++;
      }
    }
    const int numptr_amode[ntot_amode] = {23, 28, 36, 40};
    const Real specdens_amode[maxd_aspectype] = {
        0.1770000000E+04, 0.1797693135 + 309, 0.1797693135 + 309,
        0.1000000000E+04, 0.1000000000E+04,   0.1700000000E+04,
        0.1900000000E+04, 0.2600000000E+04,   0.1601000000E+04,
        0.0000000000E+00, 0.0000000000E+00,   0.0000000000E+00,
        0.0000000000E+00, 0.0000000000E+00};
    const Real spechygro[maxd_aspectype] = {
        0.5070000000E+00, 0.1797693135 + 309, 0.1797693135 + 309,
        0.1000000083E-09, 0.1400000000E+00,   0.1000000013E-09,
        0.1160000000E+01, 0.6800000000E-01,   0.1000000015E+00,
        0.0000000000E+00, 0.0000000000E+00,   0.0000000000E+00,
        0.0000000000E+00, 0.0000000000E+00};

    Real exp45logsig[AeroConfig::num_modes()], alogsig[AeroConfig::num_modes()],
        num2vol_ratio_min_nmodes[AeroConfig::num_modes()],
        num2vol_ratio_max_nmodes[AeroConfig::num_modes()] = {};

    Real aten = zero;

    ndrop_od::ndrop_int(exp45logsig, alogsig, aten,
                        num2vol_ratio_min_nmodes,  // voltonumbhi_amode
                        num2vol_ratio_max_nmodes); // voltonumblo_amode

    ndrop_od::loadaer(
        state_q.data(), nspec_amode, air_density, phase, lspectype_amode,
        specdens_amode, spechygro, lmassptr_amode, num2vol_ratio_min_nmodes,
        num2vol_ratio_max_nmodes, numptr_amode, qcldbrn1d, qcldbrn1d_num.data(),
        naerosol.data(), vaerosol.data(), hygro.data());

    output.set("naerosol", naerosol);
    output.set("vaerosol", vaerosol);
    output.set("hygro", hygro);
  });
}
