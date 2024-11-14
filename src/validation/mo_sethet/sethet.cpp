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

void sethet(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    using View1DHost = typename HostType::view_1d<Real>;
    using ColumnView = haero::ColumnView;
    constexpr int nlev = mam4::nlev;
    constexpr int gas_pcnst = mam4::gas_chemistry::gas_pcnst;
    constexpr int nfs = mam4::gas_chemistry::nfs;
    // index of total atm density in invariant array
    constexpr int indexm = mam4::gas_chemistry::indexm;

    // non-ColumnView input values
    const Real rlat = input.get_array("rlat")[0]; // need
    const Real phis = input.get_array("phis")[0];
    const Real delt = input.get_array("delt")[0];

    const int spc_h2o2_ndx = input.get_array("spc_h2o2_ndx")[0] - 1;
    const int spc_so2_ndx = input.get_array("spc_so2_ndx")[0] - 1;
    const int h2o2_ndx = input.get_array("h2o2_ndx")[0] - 1;
    const int so2_ndx = input.get_array("so2_ndx")[0] - 1;
    const int h2so4_ndx = input.get_array("h2so4_ndx")[0] - 1;
    const int gas_wetdep_cnt = input.get_array("gas_wetdep_cnt")[0];

    int wetdep_map[3];
    const auto wetdep_map_in = input.get_array("wetdep_map");
    wetdep_map[0] = wetdep_map_in[0] - 1;
    wetdep_map[1] = wetdep_map_in[1] - 1;
    wetdep_map[2] = wetdep_map_in[2] - 1;

    const auto press_in = input.get_array("press");
    const auto zmid_in = input.get_array("zmid");
    const auto tfld_in = input.get_array("tfld");
    const auto cmfdqr_in = input.get_array("cmfdqr");
    const auto nrain_in = input.get_array("nrain");
    const auto nevapr_in = input.get_array("nevapr");
    const auto xhnm_in = input.get_array("xhnm");
    const auto qin_in = input.get_array("qin");

    // ColumnView input values
    ColumnView press, zmid, tfld, cmfdqr, nrain, nevapr, xhnm;

    auto press_host = View1DHost((Real *)press_in.data(), nlev);
    auto zmid_host = View1DHost((Real *)zmid_in.data(), nlev);
    auto tfld_host = View1DHost((Real *)tfld_in.data(), nlev);
    auto cmfdqr_host = View1DHost((Real *)cmfdqr_in.data(), nlev);
    auto nrain_host = View1DHost((Real *)nrain_in.data(), nlev);
    auto nevapr_host = View1DHost((Real *)nevapr_in.data(), nlev);
    auto xhnm_host = View1DHost((Real *)xhnm_in.data(), nlev);

    press = haero::testing::create_column_view(nlev);
    zmid = haero::testing::create_column_view(nlev);
    tfld = haero::testing::create_column_view(nlev);
    cmfdqr = haero::testing::create_column_view(nlev);
    nrain = haero::testing::create_column_view(nlev);
    nevapr = haero::testing::create_column_view(nlev);
    xhnm = haero::testing::create_column_view(nlev);

    Kokkos::deep_copy(press, press_host);
    Kokkos::deep_copy(zmid, zmid_host);
    Kokkos::deep_copy(tfld, tfld_host);
    Kokkos::deep_copy(cmfdqr, cmfdqr_host);
    Kokkos::deep_copy(nrain, nrain_host);
    Kokkos::deep_copy(nevapr, nevapr_host);
    Kokkos::deep_copy(xhnm, xhnm_host);

    View2D invariants("invariants", pver, nfs);

    // working var inputs
    ColumnView xgas2, xgas3, delz, xh2o2, xso2, xliq, rain, precip, xhen_h2o2,
        xhen_hno3, xhen_so2, t_factor, xk0_hno3, xk0_so2, so2_diss;

    xgas2 = haero::testing::create_column_view(nlev);
    xgas3 = haero::testing::create_column_view(nlev);
    delz = haero::testing::create_column_view(nlev);
    xh2o2 = haero::testing::create_column_view(nlev);
    xso2 = haero::testing::create_column_view(nlev);
    xliq = haero::testing::create_column_view(nlev);
    rain = haero::testing::create_column_view(nlev);
    precip = haero::testing::create_column_view(nlev);
    xhen_h2o2 = haero::testing::create_column_view(nlev);
    xhen_hno3 = haero::testing::create_column_view(nlev);
    xhen_so2 = haero::testing::create_column_view(nlev);
    t_factor = haero::testing::create_column_view(nlev);
    xk0_hno3 = haero::testing::create_column_view(nlev);
    xk0_so2 = haero::testing::create_column_view(nlev);
    so2_diss = haero::testing::create_column_view(nlev);

    ColumnView tmp_hetrates[gas_pcnst];
    ColumnView qin[gas_pcnst];
    View1DHost tmp_hetrates_host[gas_pcnst];
    View1DHost qin_host[gas_pcnst];

    View2D het_rates("het_rates", pver, gas_pcnst);
    auto het_rates_host = Kokkos::create_mirror_view(het_rates);

    for (int mm = 0; mm < gas_pcnst; ++mm) {

      tmp_hetrates[mm] = haero::testing::create_column_view(nlev);
      qin[mm] = haero::testing::create_column_view(nlev);
      tmp_hetrates_host[mm] = View1DHost("tmp_hetrates_host", nlev);
      qin_host[mm] = View1DHost("qin_host", nlev);
    }

    int count = 0;
    for (int mm = 0; mm < gas_pcnst; ++mm) {
      for (int kk = 0; kk < nlev; ++kk) {
        qin_host[mm](kk) = qin_in[count];
        count++;
      }
    }

    // transfer data to GPU.
    for (int mm = 0; mm < gas_pcnst; ++mm) {
      Kokkos::deep_copy(tmp_hetrates[mm], 0.0);
      Kokkos::deep_copy(qin[mm], qin_host[mm]);
    }

    auto team_policy = ThreadTeamPolicy(1u, Kokkos::AUTO);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
          Kokkos::parallel_for(
              Kokkos::TeamVectorRange(team, pver),
              [&](int kk) { invariants(kk, indexm) = xhnm(kk); });
          team.team_barrier();
          mo_sethet::sethet_detail(
              team, het_rates, rlat, press, zmid, phis, tfld, cmfdqr, nrain,
              nevapr, delt, invariants, qin, t_factor, xk0_hno3, xk0_so2,
              so2_diss, xgas2, xgas3, delz, xh2o2, xso2, xliq, rain, precip,
              xhen_h2o2, xhen_hno3, xhen_so2, tmp_hetrates, spc_h2o2_ndx,
              spc_so2_ndx, h2o2_ndx, so2_ndx, h2so4_ndx, gas_wetdep_cnt,
              wetdep_map, indexm);
        });

    // transfer data to GPU.
    Kokkos::deep_copy(het_rates_host, het_rates);

    std::vector<Real> het_rates_out(nlev * gas_pcnst);
    count = 0;
    for (int mm = 0; mm < gas_pcnst; ++mm) {
      for (int kk = 0; kk < nlev; ++kk) {
        het_rates_out[count] = het_rates_host(kk, mm);
        count++;
      }
    }

    output.set("het_rates", het_rates_out);
  });
}
