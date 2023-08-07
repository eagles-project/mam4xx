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

void update_from_explmix(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    // number of vertical points.
    const int ntot_amode = AeroConfig::num_modes();
    const int pver = ndrop::pver;
    const auto mam_idx_db = input.get_array("mam_idx");
    const auto nspec_amode_db = input.get_array("nspec_amode");

    /* int nnew_in = */ input.get_array("nnew")[0];
    /* int nsav_in = */ input.get_array("nsav")[0];

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

    using View1D = ndrop::View1D;
    using View2D = ndrop::View2D;

    using View1DHost = typename HostType::view_1d<Real>;

    View1D indexes = haero::testing::create_column_view(2);
    auto indexes_host = View1DHost("nnew_nsav", 2);
    Kokkos::deep_copy(indexes, indexes_host);

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

    ColumnView zn, csbot, zs, ekd, overlapp, overlapm, ekkp, ekkm, qncld, srcn,
        source, qcld, cldn;

    ekd = haero::testing::create_column_view(pver);
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

    auto csbot_host = View1DHost((Real *)csbot_db.data(), pver);
    auto cldn_host = View1DHost((Real *)cldn_col_db.data(), pver);
    auto zn_host = View1DHost((Real *)zn_db.data(), pver);
    auto zs_host = View1DHost((Real *)zs_db.data(), pver);
    auto ekd_host = View1DHost((Real *)ekd_db.data(), pver);
    auto qcld_host = View1DHost((Real *)qcld_db.data(), pver);

    Kokkos::deep_copy(qcld, qcld_host);
    Kokkos::deep_copy(ekd, ekd_host);
    Kokkos::deep_copy(zn, zn_host);
    Kokkos::deep_copy(zs, zs_host);
    Kokkos::deep_copy(cldn, cldn_host);
    Kokkos::deep_copy(csbot, csbot_host);

    View2D nact("nact", pver, ntot_amode);
    View2D mact("mact", pver, ntot_amode);
    auto nact_host = Kokkos::create_mirror_view(nact);
    auto mact_host = Kokkos::create_mirror_view(mact);
    counter = 0;
    for (int i = 0; i < ntot_amode; ++i) {
      // input data is store on the cpu.
      for (int kk = 0; kk < pver; ++kk) {
        nact_host(kk, i) = nact_db[counter];
        mact_host(kk, i) = mact_db[counter];
        counter++;
      }
    }

    View1D raercol_cw[pver][2];
    View1D raercol[pver][2];
    View1DHost raercol_host[pver][2];
    View1DHost raercol_cw_host[pver][2];
    for (int i = 0; i < pver; ++i) {
      raercol[i][0] = View1D("raercol_0", ncnst_tot);
      raercol[i][1] = View1D("raercol_1", ncnst_tot);
      raercol_cw[i][0] = View1D("raercol_cw_0", ncnst_tot);
      raercol_cw[i][1] = View1D("raercol_cw_0", ncnst_tot);
      raercol_host[i][0] = View1DHost("raercol_host", ncnst_tot);
      raercol_host[i][1] = View1DHost("raercol_host", ncnst_tot);
      raercol_cw_host[i][0] = View1DHost("raercol_cw_host", ncnst_tot);
      raercol_cw_host[i][1] = View1DHost("raercol_cw_host", ncnst_tot);
    }

    counter = 0;
    for (int n = 0; n < ncnst_tot; n++) {
      for (int k = 0; k < pver; k++) {
        raercol_host[k][0][n] = raercol_1[counter];
        raercol_cw_host[k][0][n] = raercol_cw_1[counter];

        raercol_host[k][1][n] = raercol_2[counter];
        raercol_cw_host[k][1][n] = raercol_cw_2[counter];
        counter++;
      }
    }

    for (int k = 0; k < pver; k++) {
      Kokkos::deep_copy(raercol[k][0], raercol_host[k][0]);
      Kokkos::deep_copy(raercol[k][1], raercol_host[k][1]);
      Kokkos::deep_copy(raercol_cw[k][0], raercol_cw_host[k][0]);
      Kokkos::deep_copy(raercol_cw[k][1], raercol_cw_host[k][1]);
    }

    Kokkos::deep_copy(nact, nact_host);
    Kokkos::deep_copy(mact, mact_host);

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

    auto team_policy = ThreadTeamPolicy(1u, Kokkos::AUTO);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
          int nnew = 1;
          int nsav = 0;
          ndrop::update_from_explmix(team, dtmicro, csbot, cldn, zn, zs, ekd,
                                     nact, mact, qcld, raercol, raercol_cw,
                                     nsav, nnew, nspec_amode, mam_idx, overlapp,
                                     overlapm, ekkp, ekkm, qncld, srcn, source);
          indexes(0) = nnew;
          indexes(1) = nsav;
        });

    Kokkos::deep_copy(qcld_host, qcld);
    Kokkos::deep_copy(nact_host, nact);
    Kokkos::deep_copy(mact_host, mact);
    Kokkos::deep_copy(indexes_host, indexes);

    counter = 0;
    for (int i = 0; i < ntot_amode; ++i) {
      // input data is store on the cpu.
      for (int kk = 0; kk < pver; ++kk) {
        nact_out[counter] = nact_host(kk, i);
        mact_out[counter] = mact_host(kk, i);
        counter++;
      }
    }

    counter = 0;
    for (int k = 0; k < pver; k++) {
      qcld_out[counter] = qcld_host[k];
      counter++;
    }

    for (int k = 0; k < pver; k++) {
      Kokkos::deep_copy(raercol_host[k][0], raercol[k][0]);
      Kokkos::deep_copy(raercol_host[k][1], raercol[k][1]);
      Kokkos::deep_copy(raercol_cw_host[k][0], raercol_cw[k][0]);
      Kokkos::deep_copy(raercol_cw_host[k][1], raercol_cw[k][1]);
    }

    nnew_out[0] = indexes_host(0) + 1;
    nsav_out[0] = indexes_host(1) + 1;
    counter = 0;
    for (int n = 0; n < ncnst_tot; n++) {
      for (int k = 0; k < pver; k++) {
        raercol_1_out[counter] = raercol_host[k][0](n);
        raercol_cw_1_out[counter] = raercol_cw_host[k][0](n);

        raercol_2_out[counter] = raercol_host[k][1](n);
        raercol_cw_2_out[counter] = raercol_cw_host[k][1](n);
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
