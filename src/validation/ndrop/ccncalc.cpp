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

    const auto voltonumbhi_amode_db = input.get_array("voltonumbhi_amode");
    const auto voltonumblo_amode_db = input.get_array("voltonumblo_amode");
    const auto numptr_amode_db = input.get_array("numptr_amode");
    const auto nspec_amode_db = input.get_array("nspec_amode");

    const auto voltonumbhi_amode = voltonumbhi_amode_db.data();
    const auto voltonumblo_amode = voltonumblo_amode_db.data();

    const auto exp45logsig_db = input.get_array("exp45logsig");
    const auto alogsig_db = input.get_array("alogsig");

    const auto exp45logsig = exp45logsig_db.data();
    const auto alogsig = alogsig_db.data();

    int numptr_amode[ntot_amode];
    int nspec_amode[ntot_amode];
    for (int i = 0; i < ntot_amode; ++i) {
      numptr_amode[i] = numptr_amode_db[i];
      nspec_amode[i] = nspec_amode_db[i];
    }

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
