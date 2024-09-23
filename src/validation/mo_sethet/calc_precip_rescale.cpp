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
using namespace mo_sethet;

void calc_precip_rescale(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    using View1DHost = typename HostType::view_1d<Real>;
    using ColumnView = haero::ColumnView;
    constexpr int pver = mam4::nlev;

    const auto cmfdqr_in = input.get_array("cmfdqr");
    const auto nrain_in = input.get_array("nrain");
    const auto nevapr_in = input.get_array("nevapr");

    ColumnView cmfdqr, nrain, nevapr;
    auto cmfdqr_host = View1DHost((Real *)cmfdqr_in.data(), pver);
    auto nrain_host = View1DHost((Real *)nrain_in.data(), pver);
    auto nevapr_host = View1DHost((Real *)nevapr_in.data(), pver);
    cmfdqr = haero::testing::create_column_view(pver);
    nrain = haero::testing::create_column_view(pver);
    nevapr = haero::testing::create_column_view(pver);
    Kokkos::deep_copy(cmfdqr, cmfdqr_host);
    Kokkos::deep_copy(nrain, nrain_host);
    Kokkos::deep_copy(nevapr, nevapr_host);

    std::vector<Real> vector0(pver, 0);
    ColumnView precip;
    auto precip_host = View1DHost(vector0.data(), pver);
    precip = haero::testing::create_column_view(pver);
    Kokkos::deep_copy(precip, precip_host);
    auto team_policy = ThreadTeamPolicy(1u, Kokkos::AUTO);

    //Real total_rain = 0.0;
    //Real total_pos = 0.0;
    std::vector<Real> vector01(1, 5.577886392e-07);
    auto total_rain_host = View1DHost(vector01.data(), 1);
    auto total_pos_host = View1DHost(vector01.data(), 1);
    ColumnView total_rain = haero::testing::create_column_view(1);
    ColumnView total_pos = haero::testing::create_column_view(1);
    Kokkos::deep_copy(total_rain, total_rain_host);
    Kokkos::deep_copy(total_pos, total_pos_host);

    Kokkos::parallel_for(
        "calc_precip_rescale", pver, KOKKOS_LAMBDA(int kk) {
      total_rain[0] +=precip(kk);
      total_pos[0] += haero::max(precip(kk), 0.0);
    });

    Kokkos::parallel_for(
        "calc_precip_rescale", pver, KOKKOS_LAMBDA(int kk) {
          calc_precip_rescale_kk(cmfdqr(kk), nrain(kk), nevapr(kk), total_rain[0], total_pos[0], precip(kk));
        });

    Kokkos::deep_copy(precip_host, precip);
    std::vector<Real> precip_out(pver);
    for (int k = 0; k < pver; k++)
      precip_out[k] = precip_host(k);

    output.set("precip", precip_out);
  });
}