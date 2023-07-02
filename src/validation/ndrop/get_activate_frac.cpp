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

void get_activate_frac(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    // number of vertical points.
    const Real zero = 0;
    const int maxd_aspectype = 14;
    const int ntot_amode = 4;
    const int nvars = 40;

    const int pver = input.get_array("pver")[0];
    const auto state_q_db = input.get_array("state_q");

    const auto tair_db = input.get_array("temp");
    const auto pmid_db = input.get_array("pmid");
    const auto wsub_db = input.get_array("wsub");

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
    ColumnView wsub;
    tair = haero::testing::create_column_view(pver);
    pmid = haero::testing::create_column_view(pver);
    wsub = haero::testing::create_column_view(pver);

    auto tair_host = Kokkos::create_mirror_view(tair);
    auto pmid_host = Kokkos::create_mirror_view(pmid);
    auto wsub_host = Kokkos::create_mirror_view(wsub);

    // // FIXME. Find a better way:
    for (int kk = 0; kk < pver; ++kk) {
      tair_host(kk) = tair_db[kk];
      pmid_host(kk) = pmid_db[kk];
      wsub_host(kk) = wsub_db[kk];
    }

    Kokkos::deep_copy(tair, tair_host);
    Kokkos::deep_copy(pmid, pmid_host);
    Kokkos::deep_copy(wsub, wsub_host);

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

    const Real aten = input.get_array("aten")[0];

    int numptr_amode[ntot_amode];
    int nspec_amode[ntot_amode];
    for (int i = 0; i < ntot_amode; ++i) {
      numptr_amode[i] = numptr_amode_db[i];
      nspec_amode[i] = nspec_amode_db[i];
    }

    ColumnView fn[ntot_amode];
    ColumnView fm[ntot_amode];
    ColumnView fluxn[ntot_amode];
    ColumnView fluxm[ntot_amode];
    ColumnView flux_fullact = haero::testing::create_column_view(pver);

    for (int i = 0; i < ntot_amode; ++i) {
      fn[i] = haero::testing::create_column_view(pver);
      fm[i] = haero::testing::create_column_view(pver);
      fluxn[i] = haero::testing::create_column_view(pver);
      fluxm[i] = haero::testing::create_column_view(pver);
    }

    Kokkos::parallel_for(
        "get_activate_frac", pver, KOKKOS_LAMBDA(int kk) {
          Real state_q_kk[nvars] = {zero};
          for (int i = 0; i < nvars; ++i) {
            state_q_kk[i] = state_q[i](kk);
          }

          Real fn_kk[ntot_amode] = {zero};
          Real fm_kk[ntot_amode] = {zero};
          Real fluxn_kk[ntot_amode] = {zero};
          Real fluxm_kk[ntot_amode] = {zero};

          const Real air_density =
              conversions::density_of_ideal_gas(tair(kk), pmid(kk));

          ndrop::get_activate_frac(
              state_q_kk, air_density, air_density, wsub(kk), tair(kk), // in
              lspectype_amode, specdens_amode, spechygro, lmassptr_amode,
              voltonumbhi_amode, voltonumblo_amode, numptr_amode, nspec_amode,
              exp45logsig, alogsig, aten, fn_kk, fm_kk, fluxn_kk, fluxm_kk,
              flux_fullact(kk));

          for (int i = 0; i < ntot_amode; ++i) {
            fn[i](kk) = fn_kk[i];
            fm[i](kk) = fm_kk[i];
            fluxn[i](kk) = fluxn_kk[i];
            fluxm[i](kk) = fluxm_kk[i];
          }
        });

    auto host = Kokkos::create_mirror_view(fn[0]);
    std::vector<Real> host_v(pver);

    for (int i = 0; i < ntot_amode; ++i) {
      Kokkos::deep_copy(host, fn[i]);
      for (int kk = 0; kk < pver; ++kk) {
        host_v[kk] = host(kk);
      }

      output.set("fn_" + std::to_string(i + 1), host_v);

      Kokkos::deep_copy(host, fm[i]);
      for (int kk = 0; kk < pver; ++kk) {
        host_v[kk] = host(kk);
      }

      output.set("fm_" + std::to_string(i + 1), host_v);

      Kokkos::deep_copy(host, fluxn[i]);
      for (int kk = 0; kk < pver; ++kk) {
        host_v[kk] = host(kk);
      }

      output.set("fluxn_" + std::to_string(i + 1), host_v);

      Kokkos::deep_copy(host, fluxm[i]);
      for (int kk = 0; kk < pver; ++kk) {
        host_v[kk] = host(kk);
      }
      output.set("fluxm_" + std::to_string(i + 1), host_v);
    }

    Kokkos::deep_copy(host, flux_fullact);
    for (int kk = 0; kk < pver; ++kk) {
      host_v[kk] = host(kk);
    }

    output.set("flux_fullact", host_v);
  });
}
