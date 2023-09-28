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
using namespace modal_aer_opt;
void binterp(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    constexpr Real zero = 0;
    using View1D = DeviceType::view_1d<Real>;

    const auto table_db = input.get_array("table");
    const auto ref_real = input.get_array("ref_real")[0];
    const auto ref_img = input.get_array("ref_img")[0];
    const auto ref_real_tab = input.get_array("ref_real_tab");
    const auto ref_img_tab = input.get_array("ref_img_tab");

    int itab_1 = 0;
    int ncoef = 5;
    int prefr = 7;
    int prefi = 10;

    View3D table("table", ncoef, prefr, prefi);
    auto table_host = Kokkos::create_mirror_view(table);

    int N1 = ncoef;
    int N2 = prefr;
    int N3 = prefi;

    for (int d1 = 0; d1 < N1; ++d1)
      for (int d2 = 0; d2 < N2; ++d2)
        for (int d3 = 0; d3 < N3; ++d3) {
          const int offset = d1 + N1 * (d2 + d3 * N2);
          table_host(d1, d2, d3) = table_db[offset];

        } // d3

    Kokkos::deep_copy(table, table_host);

    View1D coef("coef", ncoef);
    View1D tab("tab", 4);

    auto team_policy = ThreadTeamPolicy(1u, Kokkos::AUTO);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
          int itab = 0;
          int jtab = 0;
          Real ttab = zero;
          Real utab = zero;
          binterp(table, ref_real, ref_img, ref_real_tab.data(),
                  ref_img_tab.data(), itab, jtab, ttab, utab, coef.data(),
                  itab_1);
          tab(0) = Real(itab);
          tab(1) = Real(jtab);
          tab(2) = ttab;
          tab(3) = utab;
        });

    auto tab_host = Kokkos::create_mirror_view(tab);
    Kokkos::deep_copy(tab_host, tab);

    auto coef_host = Kokkos::create_mirror_view(coef);
    Kokkos::deep_copy(coef_host, coef);
    std::vector<Real> coef_out(coef_host.data(), coef_host.data() + ncoef);

    output.set("itab", std::vector<Real>(1, tab_host(0) + 1));
    output.set("jtab", std::vector<Real>(1, tab_host(1) + 1));
    output.set("ttab", std::vector<Real>(1, tab(2)));
    output.set("utab", std::vector<Real>(1, tab(3)));
    output.set("coef", coef_out);
  });
}
