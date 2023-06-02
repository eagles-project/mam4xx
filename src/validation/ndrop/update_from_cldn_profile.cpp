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

void update_from_cldn_profile(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    // number of vertical points.
    // validation test from standalone ndrop.
    const Real zero = 0;
    const int maxd_aspectype = 14;
    const int ntot_amode = 4;
    const int nspec_max = 8;

    const Real cldn_col_in = input.get_array("cldn_col_in_kk")[0];
    const Real cldn_col_in_kp1 = input.get_array("cldn_col_in_kp1")[0];
    const Real dtinv = input.get_array("dtinv")[0];
    const Real wtke_col_in = input.get_array("wtke_col_in")[0];
    const Real temp_col_in = input.get_array("temp_col_in")[0];
    const Real air_density = input.get_array("cs_col_in_kk")[0];
    const Real air_density_kp1 = input.get_array("cs_col_in_kp1")[0];

    const auto state_q_col_in_kp1 = input.get_array("state_q_col_in_kp1");
    Real qcld = input.get_array("qcld")[0];

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

    int count = 0;
    for (int i = 0; i < ntot_amode; ++i) {
      for (int j = 0; j < maxd_aspectype; ++j) {
        lspectype_amode[j][i] = lspectype_amode_1d[count];
        lmassptr_amode[j][i] = lmassptr_amode_1d[count];
        count++;
      }
    }
    const int numptr_amode[ntot_amode] = {23, 28, 36, 40};

    const Real specdens_amode[maxd_aspectype] = {0.1770000000E+004,
                                                 -1,
                                                 -1,
                                                 0.1000000000E+004,
                                                 0.1000000000E+004,
                                                 0.1700000000E+004,
                                                 0.1900000000E+004,
                                                 0.2600000000E+004,
                                                 0.1601000000E+004,
                                                 0.0000000000E+000,
                                                 0.0000000000E+000,
                                                 0.0000000000E+000,
                                                 0.0000000000E+000,
                                                 0.0000000000E+000};
    const Real spechygro[maxd_aspectype] = {0.5070000000E+000,
                                            -1,
                                            -1,
                                            0.1000000083E-009,
                                            0.1400000000E+000,
                                            0.1000000013E-009,
                                            0.1160000000E+001,
                                            0.6800000000E-001,
                                            0.1000000015E+000,
                                            0.0000000000E+000,
                                            0.0000000000E+000,
                                            0.0000000000E+000,
                                            0.0000000000E+000,
                                            0.0000000000E+000};

    Real exp45logsig[AeroConfig::num_modes()], alogsig[AeroConfig::num_modes()],
        num2vol_ratio_min_nmodes[AeroConfig::num_modes()],
        num2vol_ratio_max_nmodes[AeroConfig::num_modes()] = {};

    Real aten = zero;

    ndrop_od::ndrop_int(exp45logsig, alogsig, aten,
                        num2vol_ratio_min_nmodes,  // voltonumbhi_amode
                        num2vol_ratio_max_nmodes); // voltonumblo_amode

    const int mam_idx_db[nspec_max * ntot_amode] = {
        1, 9,  14, 22, 2, 10, 15, 23, 3, 11, 16, 24, 4, 12, 17, 25,
        5, 13, 18, 0,  6, 0,  19, 0,  7, 0,  20, 0,  8, 0,  21, 0};

    count = 0;
    int mam_idx[ntot_amode][nspec_max];
    for (int i = 0; i < nspec_max; ++i) {
      for (int j = 0; j < ntot_amode; ++j) {
        mam_idx[j][i] = mam_idx_db[count];
        count++;
      }
    }

    auto raercol_nsav = input.get_array("raercol_nsav_kk");
    auto raercol_nsav_kp1 = input.get_array("raercol_nsav_kp1");
    auto raercol_cw_nsav = input.get_array("raercol_cw_nsav");
    auto nsource_col = input.get_array("nsource_col")[0];
    auto factnum_col = input.get_array("factnum_col");
    auto nact = input.get_array("nact");
    auto mact = input.get_array("mact");

    Real ekd = input.get_array("ekd")[0];
    const Real csbot_cscen = input.get_array("csbot_cscen")[0];
    auto dz = input.get_array("dz")[0];
    auto zs = input.get_array("zs")[0];

    ndrop_od::update_from_cldn_profile(
        cldn_col_in, cldn_col_in_kp1, dtinv, wtke_col_in, zs,
        dz, // ! in
        temp_col_in, air_density, air_density_kp1, csbot_cscen,
        state_q_col_in_kp1.data(), // ! in
        lspectype_amode, specdens_amode, spechygro, lmassptr_amode,
        num2vol_ratio_min_nmodes, num2vol_ratio_max_nmodes, numptr_amode,
        nspec_amode, exp45logsig, alogsig, aten, mam_idx, raercol_nsav.data(),
        raercol_nsav_kp1.data(), raercol_cw_nsav.data(),
        nsource_col, // inout
        qcld, factnum_col.data(),
        ekd, // out
        nact.data(), mact.data());

    output.set("qcld", qcld);
    output.set("nsource_col", nsource_col);
    output.set("raercol_nsav", raercol_nsav);
    output.set("raercol_cw_nsav", raercol_cw_nsav);
    output.set("factnum_col", factnum_col);
    output.set("ekd", ekd);
    output.set("nact", nact);
    output.set("mact", mact);
  });
}
