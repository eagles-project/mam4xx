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
void vtest_test(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    // Ensemble parameters
    // Declare array of strings for input names
    std::string input_variables[] = {"dt"};

    std::string input_arrays[] = {
        "ncol", "loffset", "dtime",  "press", "pdel", "tfld", "mbar",
        "lwc",  "cldfrc",  "cldnum", "xhnm",  "qcw",  "qin"};

    // Iterate over input_variables and error if not in input
    for (std::string name : input_variables) {
      if (!input.has(name.c_str())) {
        std::cerr << "Required name for variable: " << name << std::endl;
        exit(1);
      }
    }
    // Iterate over input_arrays and error if not in input
    for (std::string name : input_arrays) {
      if (!input.has_array(name.c_str())) {
        std::cerr << "Required name for array: " << name << std::endl;
        exit(1);
      }
    }

    // Parse input
    const Real dt = input.get("dt");
    const int ncol = input.get_array("ncol")[0];
    const int loffset = input.get_array("loffset")[0];
    Real dtime = input.get_array("dtime")[0];
    Real press = input.get_array("press")[0];
    Real pdel = input.get_array("pdel")[0];
    Real tfld = input.get_array("tfld")[0];
    Real mbar = input.get_array("mbar")[0];
    Real lwc = input.get_array("lwc")[0];
    Real cldfrc = input.get_array("cldfrc")[0];
    Real cldnum = input.get_array("cldnum")[0];
    Real xhnm = input.get_array("xhnm")[0];

    auto qcw = input.get_array("qcw");
    auto qin = input.get_array("qin");

    mam4::mo_setsox::sox_init(dt);

    output.set("qcw", qcw);
    output.set("qin", qin);

    // dt:    0.00
    // ncol: [         4]
    // loffset: [         9]
    // dtime: [ 0.3600000000E+04]
    // press: [ 0.6753476429E+05]
    // pdel: [ 0.3395589227E+04]
    // tfld: [ 0.2804313949E+03]
    // mbar: [ 0.2896600000E+02]
    // lwc: [ 0.6161223535E-19]
    // cldfrc: [ 0.4131862680E-01]
    // cldnum: [ 0.1000000000E-11]
    // xhnm: [ 0.1744283847E+20]
    // qcw: [ 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00,
    // 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00,
    // 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00,
    // 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00,
    // 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00,
    // 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00,
    // 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00,
    // 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00]
    // qin: [ 0.3535379752E-07, 0.1118556121E-09, 0.1995769752E-13,
    // 0.2433004510E-11, 0.2291127136E-12, 0.3717964795E-10, 0.1138542516E-09,
    // 0.2491717570E-09, 0.2253095070E-08, 0.3390091237E-10, 0.4585812463E-12,
    // 0.8946720770E-11, 0.3891580054E-15, 0.6483274310E+10, 0.3520384006E-11,
    // 0.1098978968E-09, 0.1239028229E-14, 0.6875694162E-19, 0.1609234212E+11,
    // 0.1461002261E-11, 0.6937543758E-10, 0.1337569100E-10, 0.4186728267E-11,
    // 0.2656753863E-10, 0.1666223857E-09, 0.5118951031E-16, 0.8038981136E+06,
    // 0.5376641139E-11, 0.1230094574E-11, 0.2714275891E-21, 0.4978976956E+08]

    // const Real zero = 0;
    // const int maxd_aspectype = ndrop::maxd_aspectype;
    // const int ntot_amode = AeroConfig::num_modes();
    // const int nvars = ndrop::nvars;
    // const int psat = ndrop::psat;
    // const int ncnst_tot = ndrop::ncnst_tot;
    // const int nspec_max = mam4::ndrop::nspec_max;
    // const int nvar_ptend_q = mam4::ndrop::nvar_ptend_q;

    // const int pver = ndrop::pver;
    // const auto state_q_db = input.get_array("state_q");

    // const auto tair_db = input.get_array("temp");
    // const auto pmid_db = input.get_array("pmid");
    // const auto pint_db = input.get_array("pint");

    // const auto pdel_db = input.get_array("pdel");
    // const auto rpdel_db = input.get_array("rpdel");
    // const auto zm_db = input.get_array("zm");
    // const auto ncldwtr_db = input.get_array("ncldwtr");
    // const auto kvh_db = input.get_array("kvh");
    // const auto cldn_db = input.get_array("cldn");
    // const auto wsub_db = input.get_array("wsub");
    // const auto cldo_db = input.get_array("cldo");
    // const auto qqcw_db = input.get_array("qqcw");

    // using View1D = ndrop::View1D;
    // using View2D = ndrop::View2D;

    // using View1DHost = typename HostType::view_1d<Real>;

    // int count = 0;

    // View2D state_q("state_q", pver, nvars);
    // auto state_host = Kokkos::create_mirror_view(state_q);

    // for (int i = 0; i < nvars; ++i) {
    //   // input data is store on the cpu.
    //   for (int kk = 0; kk < pver; ++kk) {
    //     state_host(kk, i) = state_q_db[count];
    //     count++;
    //   }
    // }

    // Kokkos::deep_copy(state_q, state_host);

    // ColumnView qqcw[ncnst_tot];
    // View1DHost qqcw_host[ncnst_tot];

    // count = 0;
    // for (int i = 0; i < ncnst_tot; ++i) {
    //   qqcw[i] = haero::testing::create_column_view(pver);
    //   qqcw_host[i] = View1DHost("qqcw_host", pver);
    // }

    // for (int kk = 0; kk < pver; ++kk) {
    //   for (int i = 0; i < ncnst_tot; ++i) {
    //     qqcw_host[i](kk) = qqcw_db[count];
    //     count++;
    //   }
    // }

    // // transfer data to GPU.
    // for (int i = 0; i < ncnst_tot; ++i) {
    //   Kokkos::deep_copy(qqcw[i], qqcw_host[i]);
    // }

    // ColumnView tair;
    // ColumnView pmid;
    // ColumnView pint;
    // ColumnView pdel;
    // ColumnView rpdel;
    // ColumnView zm;
    // ColumnView ncldwtr;
    // ColumnView kvh;
    // ColumnView cldn;
    // ColumnView wsub;
    // ColumnView cldo;
    // tair = haero::testing::create_column_view(pver);
    // pmid = haero::testing::create_column_view(pver);
    // pint = haero::testing::create_column_view(pver);
    // pdel = haero::testing::create_column_view(pver);
    // rpdel = haero::testing::create_column_view(pver);
    // zm = haero::testing::create_column_view(pver);
    // ncldwtr = haero::testing::create_column_view(pver);
    // kvh = haero::testing::create_column_view(pver);
    // cldn = haero::testing::create_column_view(pver);
    // wsub = haero::testing::create_column_view(pver);
    // cldo = haero::testing::create_column_view(pver);

    // auto tair_host = View1DHost((Real *)tair_db.data(), pver);
    // auto pmid_host = View1DHost((Real *)pmid_db.data(), pver);
    // auto pint_host = View1DHost((Real *)pint_db.data(), pver);
    // auto pdel_host = View1DHost((Real *)pdel_db.data(), pver);
    // auto rpdel_host = View1DHost((Real *)rpdel_db.data(), pver);
    // auto zm_host = View1DHost((Real *)zm_db.data(), pver);
    // auto ncldwtr_host = View1DHost((Real *)ncldwtr_db.data(), pver);
    // auto kvh_host = View1DHost((Real *)kvh_db.data(), pver);
    // auto cldn_host = View1DHost((Real *)cldn_db.data(), pver);
    // auto wsub_host = View1DHost((Real *)wsub_db.data(), pver);
    // auto cldo_host = View1DHost((Real *)cldo_db.data(), pver);

    // Kokkos::deep_copy(tair, tair_host);
    // Kokkos::deep_copy(pmid, pmid_host);
    // Kokkos::deep_copy(pint, pint_host);

    // Kokkos::deep_copy(pdel, pdel_host);
    // Kokkos::deep_copy(rpdel, rpdel_host);
    // Kokkos::deep_copy(zm, zm_host);
    // Kokkos::deep_copy(ncldwtr, ncldwtr_host);
    // Kokkos::deep_copy(kvh, kvh_host);
    // Kokkos::deep_copy(cldn, cldn_host);
    // Kokkos::deep_copy(wsub, wsub_host);
    // Kokkos::deep_copy(cldo, cldo_host);

    // const Real dtmicro = input.get_array("dtmicro")[0];

    // // output
    // ColumnView qcld;
    // ColumnView tendnd;
    // ColumnView ndropcol;
    // ColumnView ndropmix;
    // ColumnView nsource;
    // ColumnView wtke;
    // qcld = haero::testing::create_column_view(pver);
    // tendnd = haero::testing::create_column_view(pver);
    // ndropcol = haero::testing::create_column_view(pver);
    // ndropmix = haero::testing::create_column_view(pver);
    // nsource = haero::testing::create_column_view(pver);
    // wtke = haero::testing::create_column_view(pver);

    // ColumnView ptend_q[nvar_ptend_q];

    // count = 0;
    // for (int i = 0; i < nvar_ptend_q; ++i) {
    //   ptend_q[i] = haero::testing::create_column_view(pver);
    // }

    // View2D factnum("factnum", pver, ntot_amode);

    // ColumnView coltend[ncnst_tot];
    // ColumnView coltend_cw[ncnst_tot];

    // for (int i = 0; i < ncnst_tot; ++i) {
    //   coltend[i] = haero::testing::create_column_view(pver);
    //   coltend_cw[i] = haero::testing::create_column_view(pver);
    // }

    // View2D ccn("ccn", pver, psat);

    // View1D raercol_cw[pver][2];
    // View1D raercol[pver][2];
    // for (int i = 0; i < pver; ++i) {
    //   raercol[i][0] = View1D("raercol_0", ncnst_tot);
    //   raercol[i][1] = View1D("raercol_1", ncnst_tot);
    //   raercol_cw[i][0] = View1D("raercol_cw_0", ncnst_tot);
    //   raercol_cw[i][1] = View1D("raercol_cw_0", ncnst_tot);
    // }

    // View2D nact("nact", pver, ntot_amode);
    // View2D mact("mact", pver, ntot_amode);

    // ColumnView ekd;
    // ekd = haero::testing::create_column_view(pver);

    // ColumnView zn, csbot, zs, overlapp, overlapm, ekkp, ekkm, qncld, srcn,
    //     source;

    // zn = haero::testing::create_column_view(pver);
    // csbot = haero::testing::create_column_view(pver);
    // zs = haero::testing::create_column_view(pver);
    // overlapp = haero::testing::create_column_view(pver);
    // overlapm = haero::testing::create_column_view(pver);
    // ekkp = haero::testing::create_column_view(pver);
    // ekkm = haero::testing::create_column_view(pver);
    // qncld = haero::testing::create_column_view(pver);
    // srcn = haero::testing::create_column_view(pver);
    // source = haero::testing::create_column_view(pver);

    // ColumnView dz, csbot_cscen;
    // dz = haero::testing::create_column_view(pver);
    // csbot_cscen = haero::testing::create_column_view(pver);

    // ColumnView raertend, qqcwtend;

    // raertend = haero::testing::create_column_view(pver);
    // qqcwtend = haero::testing::create_column_view(pver);

    // auto team_policy = ThreadTeamPolicy(1u, Kokkos::AUTO);
    // Kokkos::parallel_for(
    //     team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
    //       int nspec_amode[ntot_amode];
    //       int lspectype_amode[maxd_aspectype][ntot_amode];
    //       int lmassptr_amode[maxd_aspectype][ntot_amode];
    //       Real specdens_amode[maxd_aspectype];
    //       Real spechygro[maxd_aspectype];
    //       int numptr_amode[ntot_amode];
    //       int mam_idx[ntot_amode][nspec_max];
    //       int mam_cnst_idx[ntot_amode][nspec_max];

    //       ndrop::get_e3sm_parameters(
    //           nspec_amode, lspectype_amode, lmassptr_amode, numptr_amode,
    //           specdens_amode, spechygro, mam_idx, mam_cnst_idx);

    //       Real exp45logsig[AeroConfig::num_modes()],
    //           alogsig[AeroConfig::num_modes()],
    //           num2vol_ratio_min_nmodes[AeroConfig::num_modes()],
    //           num2vol_ratio_max_nmodes[AeroConfig::num_modes()] = {};

    //       Real aten = zero;

    //       ndrop::ndrop_init(exp45logsig, alogsig, aten,
    //                         num2vol_ratio_min_nmodes,  // voltonumbhi_amode
    //                         num2vol_ratio_max_nmodes); // voltonumblo_amode

    //       ndrop::vtest_test(
    //           team, dtmicro, tair, pmid, pint, pdel, rpdel,
    //           zm, //  ! in zm[kk] - zm[kk+1], for pver zm[kk-1] - zm[kk]
    //           state_q, ncldwtr,
    //           kvh, // kvh[kk+1]
    //           cldn, lspectype_amode, specdens_amode, spechygro,
    //           lmassptr_amode, num2vol_ratio_min_nmodes,
    //           num2vol_ratio_max_nmodes, numptr_amode, nspec_amode,
    //           exp45logsig, alogsig, aten, mam_idx, mam_cnst_idx, qcld, //
    //           wsub,
    //           cldo, // in
    //           qqcw, // inout
    //           ptend_q, tendnd, factnum, ndropcol, ndropmix, nsource, wtke,
    //           ccn, coltend, coltend_cw, raercol_cw, raercol, nact, mact, ekd,
    //           // work arrays
    //           zn, csbot, zs, overlapp, overlapm, ekkp, ekkm, qncld, srcn,
    //           source, dz, csbot_cscen, raertend, qqcwtend);
    //     });

    // auto host = Kokkos::create_mirror_view(tendnd);
    // Kokkos::deep_copy(host, tendnd);
    // std::vector<Real> v_host(pver);
    // for (int kk = 0; kk < pver; ++kk) {
    //   v_host[kk] = host(kk);
    // }

    // output.set("tendnd", v_host);

    // std::vector<Real> output_qqcw;

    // // transfer data to host
    // for (int i = 0; i < ncnst_tot; ++i) {
    //   Kokkos::deep_copy(qqcw_host[i], qqcw[i]);
    // }

    // for (int kk = 0; kk < pver; ++kk) {
    //   for (int i = 0; i < ncnst_tot; ++i) {
    //     output_qqcw.push_back(qqcw_host[i](kk));
    //   }
    // }

    // output.set("qqcw", output_qqcw);

    // auto ptend_q_host = Kokkos::create_mirror_view(ptend_q[0]);
    // std::vector<Real> output_ptend_q;
    // for (int i = 0; i < nvar_ptend_q; ++i) {
    //   Kokkos::deep_copy(ptend_q_host, ptend_q[i]);
    //   for (int kk = 0; kk < pver; ++kk) {
    //     output_ptend_q.push_back(ptend_q_host(kk));
    //   }
    // }
    // output.set("ptend_q", output_ptend_q);

    // auto factnum_host = Kokkos::create_mirror_view(factnum);
    // Kokkos::deep_copy(factnum_host, factnum);

    // std::vector<Real> output_factnum;
    // for (int i = 0; i < ntot_amode; ++i) {
    //   for (int kk = 0; kk < pver; ++kk) {
    //     output_factnum.push_back(factnum_host(kk, i));
    //   }
    // }

    // output.set("factnum", output_factnum);
  });
}