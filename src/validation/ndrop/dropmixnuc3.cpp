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

void dropmixnuc3(Ensemble *ensemble) {
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
     } 
     
    for (int kk = 0; kk < pver; ++kk) {
    for (int i = 0; i < ncnst_tot; ++i) {
      // input data is store on the cpu.
      // auto qqcw_i_host = Kokkos::create_mirror_view(qqcw[i]);
        // qqcw_i_host(kk) = qqcw_db[count];
      qqcw[i](kk)= qqcw_db[count];
        count++;
      }
      // transfer data to GPU.
      // Kokkos::deep_copy(qqcw[i], qqcw_i_host);
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

    ColumnView coltend[ncnst_tot]; 
    ColumnView coltend_cw[ncnst_tot];

    for (int i = 0; i < ncnst_tot; ++i) {
      coltend[i] = haero::testing::create_column_view(pver);
      coltend_cw[i] = haero::testing::create_column_view(pver);
    }


    ColumnView ccn[psat];

    for (int i = 0; i < psat; ++i) {
      ccn[i] = haero::testing::create_column_view(pver);
    }

    ColumnView raercol_cw[2][ncnst_tot];
    ColumnView raercol[2][ncnst_tot];

    for (int i = 0; i < ncnst_tot; ++i) {
      raercol[0][i] = haero::testing::create_column_view(pver);
      raercol[1][i] = haero::testing::create_column_view(pver);
      raercol_cw[0][i] = haero::testing::create_column_view(pver);
      raercol_cw[1][i] = haero::testing::create_column_view(pver);
    }

    ColumnView nact[ntot_amode];
    ColumnView mact[ntot_amode];

    for (int i = 0; i < ntot_amode; ++i) {
      nact[i] = haero::testing::create_column_view(pver);
      mact[i] = haero::testing::create_column_view(pver);
    }

    ndrop_od::dropmixnuc3(
        dtmicro, tair, pmid, pint, pdel,
        rpdel,
        zm, //  ! in zm[kk] - zm[kk+1], for pver zm[kk-1] - zm[kk]
        state_q, ncldwtr,
        kvh, // kvh[kk+1]
        cldn, lspectype_amode, specdens_amode, spechygro, lmassptr_amode,
        num2vol_ratio_min_nmodes, num2vol_ratio_max_nmodes, numptr_amode,
        nspec_amode, exp45logsig, alogsig, aten, mam_idx, mam_cnst_idx,
        qcld, //
        wsub,
        cldo, // in
        qqcw,      // inout
        ptend_q, tendnd, factnum, ndropcol, ndropmix,
        nsource, wtke, ccn, coltend, coltend_cw,
        raercol_cw,
        raercol,
        nact, mact
        );


    auto host = Kokkos::create_mirror_view(tendnd);
    Kokkos::deep_copy(host, tendnd);
    std::vector<Real> v_host(pver);
    for (int kk = 0; kk < pver; ++kk) {
      v_host[kk] = host(kk);
    }

    output.set("tendnd", v_host);

    std::vector<Real> output_qqcw;

    for (int i = 0; i < ncnst_tot; ++i) {
      for (int kk = 0; kk < pver; ++kk) {
        output_qqcw.push_back(qqcw[i](kk));
      }
    }
    output.set("qqcw", output_qqcw);


    std::vector<Real> output_ptend_q;
    // count =0;
    for (int i = 0; i < nvar_ptend_q; ++i) {
      // Kokkos::deep_copy(host, ptend_q[i]);
    for (int kk = 0; kk < pver; ++kk) {
        output_ptend_q.push_back(ptend_q[i](kk));
        // printf("%d ptend_q_kk[%d][%d] %e \n ",count, kk,i, ptend_q_kk[kk][i]);
        // count++;
      }
    }
    output.set("ptend_q", output_ptend_q);

    std::vector<Real> output_factnum;

    for (int i = 0; i < ntot_amode; ++i) {
      // Kokkos::deep_copy(host, factnum[i]);
      for (int kk = 0; kk < pver; ++kk) {
        output_factnum.push_back(factnum[i](kk));
      }
    }

    output.set("factnum", output_factnum);

    // std::vector<Real> output_raercol_cw;

    // for (int i = 0; i < pver; ++i) {
    //   output_raercol_cw.push_back(raercol_cw[0][0](i));
    // }  

    // output.set("raercol_cw_out", output_raercol_cw);


  });
}
