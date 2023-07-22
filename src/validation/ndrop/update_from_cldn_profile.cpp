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
using namespace ndrop;
void update_from_cldn_profile(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    const Real zero = 0;
    const int ntot_amode = AeroConfig::num_modes();

    const Real cldn_col_in = input.get_array("cldn_col_in_kk")[0];
    const Real cldn_col_in_kp1 = input.get_array("cldn_col_in_kp1")[0];
    const Real dtinv = input.get_array("dtinv")[0];
    const Real wtke_col_in = input.get_array("wtke_col_in")[0];
    const Real temp_col_in = input.get_array("temp_col_in")[0];
    const Real air_density = input.get_array("cs_col_in_kk")[0];
    const Real air_density_kp1 = input.get_array("cs_col_in_kp1")[0];

    const auto state_q_col_in_kp1 = input.get_array("state_q_col_in_kp1");
    Real qcld = input.get_array("qcld")[0];

    Real exp45logsig[AeroConfig::num_modes()], alogsig[AeroConfig::num_modes()],
        num2vol_ratio_min_nmodes[AeroConfig::num_modes()],
        num2vol_ratio_max_nmodes[AeroConfig::num_modes()] = {};

    Real aten = zero;

    ndrop::ndrop_init(exp45logsig, alogsig, aten,
                      num2vol_ratio_min_nmodes,  // voltonumbhi_amode
                      num2vol_ratio_max_nmodes); // voltonumblo_amode

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

    int nspec_amode[ntot_amode];
    int lspectype_amode[maxd_aspectype][ntot_amode];
    int lmassptr_amode[maxd_aspectype][ntot_amode];
    Real specdens_amode[maxd_aspectype];
    Real spechygro[maxd_aspectype];
    int numptr_amode[ntot_amode];
    int mam_idx[ntot_amode][nspec_max];
    int mam_cnst_idx[ntot_amode][nspec_max];

    ndrop::get_e3sm_parameters(nspec_amode, lspectype_amode, lmassptr_amode,
                               numptr_amode, specdens_amode, spechygro, mam_idx,
                               mam_cnst_idx);

    ndrop::update_from_cldn_profile(
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
