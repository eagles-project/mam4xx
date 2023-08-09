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
using namespace mo_chm_diags;

// constexpr const int gas_pcnst = gas_chemistry::gas_pcnst;

void het_diags(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    using View1DHost = typename HostType::view_1d<Real>;
    using View1D = typename DeviceType::view_1d<Real>;
    using ColumnView = haero::ColumnView;

    const auto het_rates_in = input.get_array("het_rates");
    const auto mmr_in = input.get_array("mmr");
    const auto pdel_in = input.get_array("pdel");
    const auto wght_in = input.get_array("wght");

    Real wght = wght_in[0];

    ColumnView het_rates[gas_pcnst];
    ColumnView mmr[gas_pcnst];
    View1DHost het_rates_host[gas_pcnst];
    View1DHost mmr_host[gas_pcnst];

    for (int mm = 0; mm < gas_pcnst; ++mm) {
      het_rates[mm] = haero::testing::create_column_view(pver);
      mmr[mm] = haero::testing::create_column_view(pver);

      het_rates_host[mm] = View1DHost("het_rates_host", pver);
      mmr_host[mm] = View1DHost("mmr_host", pver);
    }

    int count = 0;
    for (int mm = 0; mm < gas_pcnst; ++mm) {
      for (int kk = 0; kk < pver; ++kk) {
        het_rates_host[mm](kk) = het_rates_in[count];
        mmr_host[mm](kk) = mmr_in[count];
        count++;
      }
    }

    // transfer data to GPU.
    for (int mm = 0; mm < gas_pcnst; ++mm) {
      Kokkos::deep_copy(het_rates[mm], het_rates_host[mm]);
      Kokkos::deep_copy(mmr[mm], mmr_host[mm]);
    }

    ColumnView pdel;
    auto pdel_host =
        View1DHost((Real *)pdel_in.data(), pver); // puts data into host
    pdel = haero::testing::create_column_view(pver);
    Kokkos::deep_copy(pdel, pdel_host);

    std::vector<Real> vector0(gas_pcnst, 0);
    std::vector<Real> single_vector0(1, 0);

    auto wrk_wd_host = View1DHost(vector0.data(), gas_pcnst);
    const auto wrk_wd = View1D("wrk_wd", gas_pcnst);
    Kokkos::deep_copy(wrk_wd, wrk_wd_host);

    auto sox_wk_host = View1DHost(single_vector0.data(), 1);
    const auto sox_wk = View1D("sox_wk", 1);
    Kokkos::deep_copy(sox_wk, sox_wk_host);

    const Real sox_species[3] = {4, -1, 3};
    const Real adv_mass[gas_pcnst] = {
        47.998200,     34.013600,  98.078400,     64.064800, 62.132400,
        12.011000,     115.107340, 12.011000,     12.011000, 12.011000,
        135.064039,    58.442468,  250092.672000, 1.007400,  115.107340,
        12.011000,     58.442468,  250092.672000, 1.007400,  135.064039,
        58.442468,     115.107340, 12.011000,     12.011000, 12.011000,
        250092.672000, 1.007400,   12.011000,     12.011000, 250092.672000,
        1.007400};

    auto team_policy = ThreadTeamPolicy(1u, Kokkos::AUTO);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
          mo_chm_diags::het_diags(team, het_rates, mmr, pdel, wght, wrk_wd,
                                  sox_wk, adv_mass, sox_species);
        });

    std::vector<Real> wrk_wd_mm(gas_pcnst);
    Kokkos::deep_copy(wrk_wd_host, wrk_wd);
    for (int mm = 0; mm < gas_pcnst; mm++) {
      wrk_wd_mm[mm] = wrk_wd_host(mm);
    }

    Kokkos::deep_copy(sox_wk_host, sox_wk);

    std::vector<Real> sox_wk_out(1);

    output.set("wrk_wd_mm", wrk_wd_mm);
    output.set("sox_wk", sox_wk_host[0]);
  });
}
