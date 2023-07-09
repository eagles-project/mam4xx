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

void ccncalc(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    // number of vertical points.
    // validation test from standalone ndrop.
    const Real zero = 0;
    const int maxd_aspectype = 14;
    const int ntot_amode = 4;
    const int nvars = 40;
    const int psat = ndrop::psat;

    const int pver = input.get_array("pver")[0];
    const auto state_q_db = input.get_array("state_q");

    const auto tair_db = input.get_array("temp");
    const auto pmid_db = input.get_array("pmid");

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
    tair = haero::testing::create_column_view(pver);
    pmid = haero::testing::create_column_view(pver);

    auto tair_host = Kokkos::create_mirror_view(tair);
    auto pmid_host = Kokkos::create_mirror_view(pmid);

    // // FIXME. Find a better way:
    for (int kk = 0; kk < pver; ++kk) {
      tair_host(kk) = tair_db[kk];
      pmid_host(kk) = pmid_db[kk];
    }
    Kokkos::deep_copy(tair, tair_host);
    Kokkos::deep_copy(pmid, pmid_host);

    Real qcldbrn[maxd_aspectype][ntot_amode] = {{zero}};
    Real qcldbrn_num[ntot_amode] = {zero};

    ColumnView ccn[psat];

    for (int i = 0; i < psat; ++i) {
      ccn[i] = haero::testing::create_column_view(pver);
    }

    Kokkos::parallel_for(
        "ccncalc", pver - top_lev, KOKKOS_LAMBDA(int k) {
          // k begins at 0
          const int kk = k + top_lev;
          Real state_q_kk[nvars] = {zero};
          for (int i = 0; i < nvars; ++i) {
            state_q_kk[i] = state_q[i](kk);
          }

          const Real air_density =
              conversions::density_of_ideal_gas(tair(kk), pmid(kk));

          Real exp45logsig[4] = {0.4733757267E+01, 0.2702197556E+01,
                                 0.4733757267E+01, 0.2702197727E+01};
          Real voltonumbhi_amode[4] = {0.4736279937E+19, 0.5026599108E+22,
                                       0.6303988596E+16, 0.7067799781E+21};
          Real voltonumblo_amode[4] = {0.2634717443E+22, 0.1073313330E+25,
                                       0.4034552701E+18, 0.7067800158E+24};
          const int nspec_amode[ntot_amode] = {7, 4, 7, 3};
          const int lspectype_amode_1d[ntot_amode * maxd_aspectype] = {
            1, 4, 5, 6, 8, 7, 9, 0, 0, 0, 0, 0, 0, 0, 1, 5, 7, 9, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 7, 1, 6, 4, 5, 9, 0, 0, 0,
            0, 0, 0, 0, 4, 6, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
          const int lmassptr_amode_1d[ntot_amode * maxd_aspectype] = {
            16, 17, 18, 19, 20, 21, 22, 0, 0, 0, 0, 0, 0, 0, 24, 25, 26, 27, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 29, 30, 31, 32, 33, 34, 35, 0, 0, 0,
            0, 0, 0, 0, 37, 38, 39, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
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
          const Real specdens_amode[maxd_aspectype] = {
            0.1770000000E+04, 0.1797693135 + 309, 0.1797693135 + 309,
            0.1000000000E+04, 0.1000000000E+04,  0.1700000000E+04,
            0.1900000000E+04, 0.2600000000E+04,  0.1601000000E+04,
            0.0000000000E+00, 0.0000000000E+00,  0.0000000000E+00,
            0.0000000000E+00, 0.0000000000E+00};
          const Real spechygro[maxd_aspectype] = {
            0.5070000000E+00, 0.1797693135 + 309, 0.1797693135 + 309,
            0.1000000083E-09, 0.1400000000E+00,  0.1000000013E-09,
            0.1160000000E+01, 0.6800000000E-01,  0.1000000015E+00,
            0.0000000000E+00, 0.0000000000E+00,  0.0000000000E+00,
            0.0000000000E+00, 0.0000000000E+00};
          Real alogsig[ntot_amode] ={0.5877866649E+00, 0.4700036292E+00,
                                    0.5877866649E+00, 0.4700036441E+00};

          Real ccn_kk[psat] = {zero};
          ndrop::ccncalc(state_q_kk, tair(kk), qcldbrn, qcldbrn_num,
                         air_density, lspectype_amode, specdens_amode,
                         spechygro, lmassptr_amode, voltonumbhi_amode,
                         voltonumblo_amode, numptr_amode, nspec_amode,
                         exp45logsig, alogsig, ccn_kk);
          for (int i = 0; i < psat; ++i) {
            ccn[i](kk) = ccn_kk[i];
          }
        });

    for (int i = 0; i < psat; ++i) {
      auto ccn_i_host = Kokkos::create_mirror_view(ccn[i]);
      Kokkos::deep_copy(ccn_i_host, ccn[i]);
      std::vector<Real> ccn_v(pver);
      for (int kk = 0; kk < pver; ++kk) {
        ccn_v[kk] = ccn_i_host(kk);
      }

      output.set("ccn_" + std::to_string(i + 1), ccn_v);
    }
  });
}
