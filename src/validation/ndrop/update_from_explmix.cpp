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

void update_from_explmix(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    // number of vertical points.
    const int top_lev = input.get_array("top_lev")[0] - 1;
    const int pver = input.get_array("pver")[0];
    // const int ntot_amode = input.get_array("ntot_amode")[0];
    const auto mam_idx_db = input.get_array("mam_idx");
    const auto nspec_amode_db = input.get_array("nspec_amode");

    int nnew = input.get_array("nnew")[0];
    int nsav = input.get_array("nsav")[0];
    nnew -= 1;
    nsav -= 1;

    const auto raercol_1 = input.get_array("raercol_1");
    const auto raercol_cw_1 = input.get_array("raercol_cw_1");
    const auto raercol_2 = input.get_array("raercol_2");
    const auto raercol_cw_2 = input.get_array("raercol_cw_2");

    auto qcld_db = input.get_array("qcld");
    auto mact_db = input.get_array("mact");
    auto nact_db = input.get_array("nact");
    const auto ekd_db = input.get_array("ekd");
    const auto zs_db = input.get_array("zs");
    const auto zn_db = input.get_array("zn");
    const auto cldn_col_db = input.get_array("cldn_col");
    const auto csbot_db = input.get_array("csbot");

    const Real dtmicro = input.get_array("dtmicro")[0];

    const Real zero = 0.0;
    const int nmodes = AeroConfig::num_modes();
    const int ncnst_tot = 25;
    const int nspec_max = 8;
    int raer_len = pver * ncnst_tot;
    int act_len = pver * nmodes;

    using ColumnHostView = typename HostType::view_1d<Real>;

    std::vector<Real> nact_out(act_len, zero);
    std::vector<Real> mact_out(act_len, zero);
    std::vector<Real> qcld_out(pver);

    std::vector<Real> raercol_1_out(raer_len, 0.0);
    std::vector<Real> raercol_2_out(raer_len, 0.0);
    std::vector<Real> raercol_cw_1_out(raer_len, 0.0);
    std::vector<Real> raercol_cw_2_out(raer_len, 0.0);

    std::vector<Real> nnew_out(1);
    std::vector<Real> nsav_out(1);
    int counter = 0;

    ColumnView ekd;
    ekd = haero::testing::create_column_view(pver);

    ColumnView zn, csbot, zs, overlapp, overlapm, ekkp, ekkm, qncld, srcn,
        source, qcld, cldn;

    zn = haero::testing::create_column_view(pver);
    csbot = haero::testing::create_column_view(pver);
    zs = haero::testing::create_column_view(pver);
    overlapp = haero::testing::create_column_view(pver);
    overlapm = haero::testing::create_column_view(pver);
    ekkp = haero::testing::create_column_view(pver);
    ekkm = haero::testing::create_column_view(pver);
    qncld = haero::testing::create_column_view(pver);
    qcld = haero::testing::create_column_view(pver);
    cldn = haero::testing::create_column_view(pver);
    srcn = haero::testing::create_column_view(pver);
    source = haero::testing::create_column_view(pver);

    ColumnView nact[pver];
    ColumnView mact[pver];
    ColumnHostView nact_host[pver];
    ColumnHostView mact_host[pver];
    for (int kk = 0; kk < pver; ++kk) {
      nact[kk] = haero::testing::create_column_view(nmodes);
      mact[kk] = haero::testing::create_column_view(nmodes);
      nact_host[kk] = ColumnHostView("nact_host", nmodes);
      mact_host[kk] = ColumnHostView("mact_host", nmodes);
    }

    ndrop::Ndrop_ColumnView raercol[pver][2];
    ndrop::Ndrop_ColumnView raercol_cw[pver][2];
    ColumnHostView raercol_host[pver][2];
    ColumnHostView raercol_cw_host[pver][2];
    for (int i = 0; i < pver; ++i) {
      raercol[i][0] = haero::testing::create_column_view(ncnst_tot);
      raercol[i][1] = haero::testing::create_column_view(ncnst_tot);
      raercol_cw[i][0] = haero::testing::create_column_view(ncnst_tot);
      raercol_cw[i][1] = haero::testing::create_column_view(ncnst_tot);
      raercol_host[i][0] = ColumnHostView("raercol_host", ncnst_tot);
      raercol_host[i][1] = ColumnHostView("raercol_host", ncnst_tot);
      raercol_cw_host[i][0] = ColumnHostView("raercol_cw_host", ncnst_tot);
      raercol_cw_host[i][1] = ColumnHostView("raercol_cw_host", ncnst_tot);
    }

    auto csbot_host = Kokkos::create_mirror_view(csbot);
    auto cldn_host = Kokkos::create_mirror_view(cldn);
    auto zn_host = Kokkos::create_mirror_view(zn);
    auto zs_host = Kokkos::create_mirror_view(zs);
    auto ekd_host = Kokkos::create_mirror_view(ekd);
    auto qncld_host = Kokkos::create_mirror_view(qncld);
    auto overlapp_host = Kokkos::create_mirror_view(overlapp);
    auto overlapm_host = Kokkos::create_mirror_view(overlapm);

    auto nact_host = Kokkos::create_mirror_view(nact);
    auto mact_host = Kokkos::create_mirror_view(mact);
    auto raercol_host = Kokkos::create_mirror_view(raercol);
    auto raercol_cw_host = Kokkos::create_mirror_view(raercol_cw);

    // FIXME: is this, and below, still open?
    // // FIXME. Find a better way:
    for (int kk = 0; kk < pver; ++kk) {
      qcld_host(kk) = qcld_db[kk];
      qncld_host(kk) = qncld_db[kk]; // TODO: qncld_db doesn't exist, ah qncld
                                     // is the ouput var?
      nact_host(kk) = nact_db[kk];
      mact_host(kk) = mact_db[kk];
      ekd_host(kk) = ekd_db[kk];
      zn_host(kk) = zn_db[kk];
      zs_host(kk) = zs_db[kk];
      cldn_host(kk) = cldn_col_db[kk];
      csbot_host(kk) = zs_db[kk];
    }

    Kokkos::deep_copy(qcld, qcld_host);
    Kokkos::deep_copy(qncld, qncld_host);
    Kokkos::deep_copy(nact, nact_host);
    Kokkos::deep_copy(mact, mact_host);
    Kokkos::deep_copy(ekd, ekd_host);
    Kokkos::deep_copy(zn, zn_host);
    Kokkos::deep_copy(zs, zs_host);
    Kokkos::deep_copy(cldn, cldn_host);
    Kokkos::deep_copy(csbod, csbot_host);

    counter = 0;
    for (int n = 0; n < ncnst_tot; n++) {
      for (int k = 0; k < pver; k++) {
        raercol[k][0](n) = raercol_1[counter];
        raercol_cw[k][0](n) = raercol_cw_1[counter];

        raercol[k][1](n) = raercol_2[counter];
        raercol_cw[k][1](n) = raercol_cw_2[counter];
        counter++;
      }
    }

    for (int k = 0; k < pver; k++) {
      Kokkos::deep_copy(raercol[k][0], raercol_host[k][0]);
      Kokkos::deep_copy(raercol[k][1], raercol_host[k][1]);
      Kokkos::deep_copy(raercol_cw[k][0], raercol_cw_host[k][0]);
      Kokkos::deep_copy(raercol_cw[k][1], raercol_cw_host[k][1]);
    }

    counter = 0;
    for (int m = 0; m < nmodes; m++) {
      for (int k = 0; k < pver; k++) {
        nact[k](m) = nact_db[counter];
        mact[k](m) = mact_db[counter];
        counter++;
      }
    }
    for (int k = 0; k < pver; k++) {
      Kokkos::deep_copy(nact[k], nact_host[k]);
      Kokkos::deep_copy(mact[k], mact_host[k]);
    }

    int nspec_amode[nmodes];
    int mam_idx[nmodes][nspec_max];
    for (int m = 0; m < nmodes; m++) {
      nspec_amode[m] = nspec_amode_db[m];
    }
    counter = 0;
    for (int n = 0; n < nspec_max; n++) {
      for (int m = 0; m < nmodes; m++) {
        mam_idx[m][n] = mam_idx_db[counter];
        counter++;
      }
    }

    // FIXME: done?
    // TODO: need thread team here
    ndrop::update_from_explmix(team, dtmicro, csbot, cldn, zn, zs, ekd, nact,
                               mact, qcld, raercol, raercol_cw, nsav, nnew,
                               nspec_amode, mam_idx, overlapp, overlapm, ekkp,
                               ekkm, qncld, srcn, source);

    // TODO: ColumnView-ify the output sequence

    nnew_out[0] = nnew + 1;
    nsav_out[0] = nsav + 1;
    counter = 0;
    for (int n = 0; n < ncnst_tot; n++) {
      for (int k = 0; k < pver; k++) {
        raercol_1_out[counter] = raercol[k][n][0];
        raercol_cw_1_out[counter] = raercol_cw[k][n][0];

        raercol_2_out[counter] = raercol[k][n][1];
        raercol_cw_2_out[counter] = raercol_cw[k][n][1];
        counter++;
      }
    }
    for (int k = 0; k < pver; k++) {
      qcld_out[k] = qcld[k];
    }
    counter = 0;
    for (int m = 0; m < nmodes; m++) {
      for (int k = 0; k < pver; k++) {
        nact[counter] = _nact[k][m];
        mact[counter] = _mact[k][m];
        counter++;
      }
    }

    output.set("nact", nact_out);
    output.set("mact", mact_out);
    output.set("qcld", qcld_out);

    output.set("raercol_1", raercol_1_out);
    output.set("raercol_cw_1", raercol_cw_1_out);
    output.set("raercol_2", raercol_2_out);
    output.set("raercol_cw_2", raercol_cw_2_out);

    output.set("nnew", nnew_out);
    output.set("nsav", nsav_out);
  });
}
