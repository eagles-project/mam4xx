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

void get_activate_frac(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    // number of vertical points.
    const int maxd_aspectype = ndrop::maxd_aspectype;
    const int ntot_amode = 4;
    const int nvars = ndrop::nvars;

    const int pver = ndrop::pver;
    const auto state_q_db = input.get_array("state_q");

    const auto tair_db = input.get_array("temp");
    const auto pmid_db = input.get_array("pmid");
    const auto wsub_db = input.get_array("wsub");

    using ColumnHostView = typename HostType::view_1d<Real>;

    int count = 0;
    ColumnView state_q[pver];
    ColumnHostView state_host[pver];

    for (int kk = 0; kk < pver; ++kk) {
      state_q[kk] = haero::testing::create_column_view(nvars);
      state_host[kk] = ColumnHostView("state_host", nvars);
    } // kk

    for (int i = 0; i < nvars; ++i) {
      // input data is store on the cpu.
      for (int kk = 0; kk < pver; ++kk) {
        state_host[kk](i) = state_q_db[count];
        count++;
      }
    }

    for (int kk = 0; kk < pver; ++kk) {
      // transfer data to GPU.
      Kokkos::deep_copy(state_q[kk], state_host[kk]);
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

    ColumnView fn[pver];
    ColumnView fm[pver];
    ColumnView fluxn[pver];
    ColumnView fluxm[pver];
    ColumnView flux_fullact = haero::testing::create_column_view(pver);

    for (int i = 0; i < pver; ++i) {
      fn[i] = haero::testing::create_column_view(ntot_amode);
      fm[i] = haero::testing::create_column_view(ntot_amode);
      fluxn[i] = haero::testing::create_column_view(ntot_amode);
      fluxm[i] = haero::testing::create_column_view(ntot_amode);
    }

    Kokkos::parallel_for(
        "get_activate_frac", pver, KOKKOS_LAMBDA(int kk) {
          const Real air_density =
              conversions::density_of_ideal_gas(tair(kk), pmid(kk));

          ndrop::get_activate_frac(
              state_q[kk].data(), air_density, air_density, wsub(kk),
              tair(kk), // in
              lspectype_amode, specdens_amode, spechygro, lmassptr_amode,
              voltonumbhi_amode, voltonumblo_amode, numptr_amode, nspec_amode,
              exp45logsig, alogsig, aten, fn[kk].data(), fm[kk].data(),
              fluxn[kk].data(), fluxm[kk].data(), flux_fullact(kk));
        });

    std::vector<Real> host_v(pver);

    ColumnHostView fn_host[pver];
    ColumnHostView fm_host[pver];
    ColumnHostView fluxn_host[pver];
    ColumnHostView fluxm_host[pver];

    for (int kk = 0; kk < pver; ++kk) {
      fn_host[kk] = ColumnHostView("fn_host", ntot_amode);
      Kokkos::deep_copy(fn_host[kk], fn[kk]);

      fm_host[kk] = ColumnHostView("fm_host", ntot_amode);
      Kokkos::deep_copy(fm_host[kk], fm[kk]);

      fluxn_host[kk] = ColumnHostView("fluxn_host", ntot_amode);
      Kokkos::deep_copy(fluxn_host[kk], fluxn[kk]);

      fluxm_host[kk] = ColumnHostView("fluxm_host", ntot_amode);
      Kokkos::deep_copy(fluxm_host[kk], fluxm[kk]);
    }

    for (int i = 0; i < ntot_amode; ++i) {

      for (int kk = 0; kk < pver; ++kk) {
        host_v[kk] = fn_host[kk](i);
      } // k

      output.set("fn_" + std::to_string(i + 1), host_v);

      for (int kk = 0; kk < pver; ++kk) {
        host_v[kk] = fm_host[kk](i);
      } // k

      output.set("fm_" + std::to_string(i + 1), host_v);

      for (int kk = 0; kk < pver; ++kk) {
        host_v[kk] = fluxn_host[kk](i);
      } // k

      output.set("fluxn_" + std::to_string(i + 1), host_v);

      for (int kk = 0; kk < pver; ++kk) {
        host_v[kk] = fluxm_host[kk](i);
      }
      output.set("fluxm_" + std::to_string(i + 1), host_v);
    } // i

    auto host = Kokkos::create_mirror_view(flux_fullact);

    Kokkos::deep_copy(host, flux_fullact);
    for (int kk = 0; kk < pver; ++kk) {
      host_v[kk] = host(kk);
    }

    output.set("flux_fullact", host_v);
  });
}
