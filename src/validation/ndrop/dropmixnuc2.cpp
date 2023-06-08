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

void dropmixnuc2(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    // number of vertical points.
    // validation test from standalone ndrop.
    const Real zero = 0;
    const int maxd_aspectype = 14;
    const int ntot_amode = 4;
    const int nvars = 40;
    const int psat = ndrop_od::psat;
    const int ncnst_tot = ndrop_od::ncnst_tot;
    const int nspec_max = mam4::ndrop_od::nspec_max;
    const int nvar_ptend_q = mam4::ndrop_od::nvar_ptend_q;

    const int pver = 72; // input.get_array("pver")[0];
    const auto state_q_db = input.get_array("state_q");

    const auto tair_db = input.get_array("temp");
    const auto pmid_db = input.get_array("pmid");
    const auto pint_db = input.get_array("pint");

    const auto pdel_db = input.get_array("pdel");
    const auto rpdel_db = input.get_array("rpdel");
    const auto zm_db = input.get_array("zm");
    const auto ncldwtr_db = input.get_array("ncldwtr");
    const auto kvh_db = input.get_array("kvh");
    const auto cldn_db = input.get_array("cldn");
    const auto wsub_db = input.get_array("wsub");
    const auto cldo_db = input.get_array("cldo");
    const auto qqcw_db = input.get_array("qqcw");

    const int top_lev = 6;
    ColumnView state_q[nvars];

    int count = 0;
    for (int i = 0; i < nvars; ++i) {
      state_q[i] = haero::testing::create_column_view(pver);
      // input data is store on the cpu.
      auto state_q_i_host = Kokkos::create_mirror_view(state_q[i]);
      for (int kk = 0; kk < pver; ++kk) {
        state_q_i_host(kk) = state_q_db[count];
        count++;
      }
      // transfer data to GPU.
      Kokkos::deep_copy(state_q[i], state_q_i_host);
    }

    ColumnView tair;
    ColumnView pmid;
    ColumnView pint;
    ColumnView pdel;
    ColumnView rpdel;
    ColumnView zm;
    ColumnView ncldwtr;
    ColumnView kvh;
    ColumnView cldn;
    ColumnView wsub;
    ColumnView cldo;
    tair = haero::testing::create_column_view(pver);
    pmid = haero::testing::create_column_view(pver);
    pint = haero::testing::create_column_view(pver);
    pdel = haero::testing::create_column_view(pver);
    rpdel = haero::testing::create_column_view(pver);
    zm = haero::testing::create_column_view(pver);
    ncldwtr = haero::testing::create_column_view(pver);
    kvh = haero::testing::create_column_view(pver);
    cldn = haero::testing::create_column_view(pver);
    wsub = haero::testing::create_column_view(pver);
    cldo = haero::testing::create_column_view(pver);

    auto tair_host = Kokkos::create_mirror_view(tair);
    auto pmid_host = Kokkos::create_mirror_view(pmid);
    auto pint_host = Kokkos::create_mirror_view(pint);

    auto pdel_host = Kokkos::create_mirror_view(pdel);
    auto rpdel_host = Kokkos::create_mirror_view(rpdel);
    auto zm_host = Kokkos::create_mirror_view(zm);
    auto ncldwtr_host = Kokkos::create_mirror_view(ncldwtr);
    auto kvh_host = Kokkos::create_mirror_view(kvh);
    auto cldn_host = Kokkos::create_mirror_view(cldn);
    auto wsub_host = Kokkos::create_mirror_view(wsub);
    auto cldo_host = Kokkos::create_mirror_view(cldo);

    // // FIXME. Find a better way:
    for (int kk = 0; kk < pver; ++kk) {
      tair_host(kk) = tair_db[kk];
      pmid_host(kk) = pmid_db[kk];
      pint_host(kk) = pint_db[kk];
      pdel_host(kk) = pdel_db[kk];
      rpdel_host(kk) = rpdel_db[kk];
      zm_host(kk) = zm_db[kk];
      ncldwtr_host(kk) = ncldwtr_db[kk];
      kvh_host(kk) = kvh_db[kk];
      cldn_host(kk) = cldn_db[kk];
      wsub_host(kk) = wsub_db[kk];
      cldo_host(kk) = cldo_db[kk];
    }
    Kokkos::deep_copy(tair, tair_host);
    Kokkos::deep_copy(pmid, pmid_host);
    Kokkos::deep_copy(pint, pint_host);

    Kokkos::deep_copy(pdel, pdel_host);
    Kokkos::deep_copy(rpdel, rpdel_host);
    Kokkos::deep_copy(zm, zm_host);
    Kokkos::deep_copy(ncldwtr, ncldwtr_host);
    Kokkos::deep_copy(kvh, kvh_host);
    Kokkos::deep_copy(cldn, cldn_host);
    Kokkos::deep_copy(wsub, wsub_host);
    Kokkos::deep_copy(cldo, cldo_host);

    ColumnView qqcw[ncnst_tot];

    count = 0;
    for (int i = 0; i < ncnst_tot; ++i) {
      qqcw[i] = haero::testing::create_column_view(pver);
      // input data is store on the cpu.
      auto qqcw_i_host = Kokkos::create_mirror_view(qqcw[i]);
      for (int kk = 0; kk < pver; ++kk) {
        qqcw_i_host(kk) = qqcw_db[count];
        count++;
      }
      // transfer data to GPU.
      Kokkos::deep_copy(qqcw[i], qqcw_i_host);
    }

    const auto lspectype_amode_db = input.get_array("lspectype_amode");
    int lspectype_amode[maxd_aspectype][ntot_amode] = {};

    const auto lmassptr_amode_db = input.get_array("lmassptr_amode");
    int lmassptr_amode[maxd_aspectype][ntot_amode] = {};

    count = 0;
    for (int i = 0; i < ntot_amode; ++i) {
      for (int j = 0; j < maxd_aspectype; ++j) {
        lspectype_amode[j][i] = lspectype_amode_db[count];
        lmassptr_amode[j][i] = lmassptr_amode_db[count];
        count++;
      }
    }

    const auto specdens_amode_db = input.get_array("specdens_amode");
    const auto spechygro_db = input.get_array("spechygro");

    const auto specdens_amode = specdens_amode_db.data();
    const auto spechygro = spechygro_db.data();

    const auto numptr_amode_db = input.get_array("numptr_amode");
    const auto nspec_amode_db = input.get_array("nspec_amode");

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

    const Real dtmicro = input.get_array("dtmicro")[0];

    const auto mam_idx_db = input.get_array("mam_idx");

    count = 0;
    int mam_idx[ntot_amode][nspec_max];
    for (int i = 0; i < nspec_max; ++i) {
      for (int j = 0; j < ntot_amode; ++j) {
        mam_idx[j][i] = mam_idx_db[count];
        count++;
      }
    }

    const auto mam_cnst_idx_db = input.get_array("mam_cnst_idx");
    count = 0;
    int mam_cnst_idx[ntot_amode][nspec_max];
    for (int i = 0; i < nspec_max; ++i) {
      for (int j = 0; j < ntot_amode; ++j) {
        mam_cnst_idx[j][i] = mam_cnst_idx_db[count];
        count++;
      }
    }

    // ColumnView ccn[psat];

    // for (int i = 0; i < psat; ++i) {
    //   ccn[i] = haero::testing::create_column_view(pver);
    // }

    // output
    ColumnView qcld;
    ColumnView tendnd;
    ColumnView ndropcol;
    ColumnView ndropmix;
    ColumnView nsource;
    ColumnView wtke;
    qcld = haero::testing::create_column_view(pver);
    tendnd = haero::testing::create_column_view(pver);
    ndropcol = haero::testing::create_column_view(pver);
    ndropmix = haero::testing::create_column_view(pver);
    nsource = haero::testing::create_column_view(pver);
    wtke = haero::testing::create_column_view(pver);

    ColumnView ptend_q[nvar_ptend_q];

    count = 0;
    for (int i = 0; i < nvar_ptend_q; ++i) {
      ptend_q[i] = haero::testing::create_column_view(pver);
    }

    ColumnView factnum[ntot_amode];

    for (int i = 0; i < ntot_amode; ++i) {
      factnum[i] = haero::testing::create_column_view(pver);
    }

    Real state_q_v[pver][nvars] = {{zero}};
    Real qqcw_v[pver][ncnst_tot] = {{zero}};
    for (int k = 0; k < pver; ++k) {
      for (int i = 0; i < nvars; ++i) {
        state_q_v[k][i] = state_q[i](k);
      }

      for (int i = 0; i < nvars; ++i) {
        qqcw_v[k][i] = qqcw[i](k);
      }
    }

    Real ptend_q_kk[pver][nvar_ptend_q] = {{zero}};
    Real factnum_kk[pver][ntot_amode] = {{zero}};

    Real ccn[pver][psat] = {};
    Real coltend[pver][ncnst_tot] = {};
    Real coltend_cw[pver][ncnst_tot] = {};

    ndrop_od::dropmixnuc2(
        dtmicro, tair.data(), pmid.data(), pint.data(), pdel.data(),
        rpdel.data(),
        zm.data(), //  ! in zm[kk] - zm[kk+1], for pver zm[kk-1] - zm[kk]
        state_q_v, ncldwtr.data(),
        kvh.data(), // kvh[kk+1]
        cldn.data(), lspectype_amode, specdens_amode, spechygro, lmassptr_amode,
        num2vol_ratio_min_nmodes, num2vol_ratio_max_nmodes, numptr_amode,
        nspec_amode, exp45logsig, alogsig, aten, mam_idx, mam_cnst_idx,
        qcld.data(), //
        wsub.data(),
        cldo.data(), // in
        qqcw_v,      // inout
        ptend_q_kk, tendnd.data(), factnum_kk, ndropcol.data(), ndropmix.data(),
        nsource.data(), wtke.data(), ccn, coltend, coltend_cw);

#if 0
    Kokkos::parallel_for(
        "dropmixnuc", pver - top_lev, KOKKOS_LAMBDA(int k) {
          // k begins at 0
          const int kk = k + top_lev;
          // FIXME
          const int kp1 = kk + 1;
          Real state_q_kk[nvars] = {zero};
          Real state_q_kp1[nvars] = {zero};
          for (int i = 0; i < nvars; ++i) {
            state_q_kk[i] = state_q[i](kk);
            state_q_kp1[i] = state_q[i](kp1);
          }

          Real qqcw_kk[ncnst_tot] = {zero};
          for (int i = 0; i < ncnst_tot; ++i) {
            qqcw_kk[i] = qqcw[i](kk);
          }

          const Real air_density =
              conversions::density_of_ideal_gas(tair(kk), pmid(kk));

          const Real air_density_kp1 =
              conversions::density_of_ideal_gas(tair(kp1), pmid(kp1));

          Real delta_zm = zero;

          if (kk < pver) {
            delta_zm = zm(kk - 1) - zm(kk);
          } else {
            delta_zm = zm(kk) - zm(kk + 1);
          }

          Real factnum_kk[ntot_amode] = {zero};
          Real ccn[psat] = {zero};
          Real ptend_q_kk[nvar_ptend_q] = {zero};
          Real coltend_kk[ncnst_tot] = {zero};
          Real coltend_cw_kk[ncnst_tot] = {zero};

          ndrop_od::dropmixnuc(
              kk, top_lev, dtmicro, tair(kk), tair(kp1), air_density,
              air_density_kp1, pint(kp1), pdel(kk), rpdel(kk),
              delta_zm, //  ! in zm[kk] - zm[kk+1], for pver zm[kk-1] - zm[kk]
              state_q_kk, state_q_kp1, ncldwtr(kk),
              kvh(kp1), // kvh[kk+1]
              cldn(kk), cldn(kp1), lspectype_amode, specdens_amode, spechygro,
              lmassptr_amode, num2vol_ratio_min_nmodes,
              num2vol_ratio_max_nmodes, numptr_amode, nspec_amode, exp45logsig,
              alogsig, aten, mam_idx, mam_cnst_idx, qcld(kk), //
              wsub(kk),
              cldo(kk), // in
              qqcw_kk,  // inout
              ptend_q_kk, tendnd(kk), factnum_kk, ndropcol(kk), ndropmix(kk),
              nsource(kk), wtke(kk), ccn, coltend_kk, coltend_cw_kk);

          for (int i = 0; i < ncnst_tot; ++i) {
            qqcw[i](kk) = qqcw_kk[i];
          }
          for (int i = 0; i < nvar_ptend_q; ++i) {
            ptend_q[i](kk) = ptend_q_kk[i];
          }

          for (int i = 0; i < ntot_amode; ++i) {
            factnum[i](kk) = factnum_kk[i];
          }
        });

    auto host = Kokkos::create_mirror_view(tendnd);
    Kokkos::deep_copy(host, tendnd);
    std::vector<Real> v_host(pver);
    for (int kk = 0; kk < pver; ++kk) {
      v_host[kk] = host(kk);
    }

    output.set("tendnd", v_host);

    std::vector<Real> output_qqcw;

    for (int i = 0; i < ncnst_tot; ++i) {
      Kokkos::deep_copy(host, qqcw[i]);
      for (int kk = 0; kk < pver; ++kk) {
        output_qqcw.push_back(host(kk));
      }
    }

    std::vector<Real> output_ptend_q;
    for (int i = 0; i < nvar_ptend_q; ++i) {
      Kokkos::deep_copy(host, ptend_q[i]);
      for (int kk = 0; kk < pver; ++kk) {
        output_ptend_q.push_back(host(kk));
      }
    }

    std::vector<Real> output_factnum;

    for (int i = 0; i < ntot_amode; ++i) {
      Kokkos::deep_copy(host, factnum[i]);
      for (int kk = 0; kk < pver; ++kk) {
        output_factnum.push_back(host(kk));
      }
    }

    output.set("qqcw", output_qqcw);
    output.set("ptend_q", output_ptend_q);
    output.set("factnum", output_factnum);

#endif
  });
}
