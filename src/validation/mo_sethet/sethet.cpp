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
    constexpr int pver = mam4::nlev;
    constexpr int gas_pcnst = mam4::gas_chemistry::gas_pcnst;

    // non-ColumnView input values
    const Real rlat = input.get_array("rlat")[0]; // need
    // const Real rlat = -.2320924702;
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

    auto press_host = View1DHost((Real *)press_in.data(), pver);
    auto zmid_host = View1DHost((Real *)zmid_in.data(), pver);
    auto tfld_host = View1DHost((Real *)tfld_in.data(), pver);
    auto cmfdqr_host = View1DHost((Real *)cmfdqr_in.data(), pver);
    auto nrain_host = View1DHost((Real *)nrain_in.data(), pver);
    auto nevapr_host = View1DHost((Real *)nevapr_in.data(), pver);
    auto xhnm_host = View1DHost((Real *)xhnm_in.data(), pver);

    press = haero::testing::create_column_view(pver);
    zmid = haero::testing::create_column_view(pver);
    tfld = haero::testing::create_column_view(pver);
    cmfdqr = haero::testing::create_column_view(pver);
    nrain = haero::testing::create_column_view(pver);
    nevapr = haero::testing::create_column_view(pver);
    xhnm = haero::testing::create_column_view(pver);
    

    Kokkos::deep_copy(press, press_host);
    Kokkos::deep_copy(zmid, zmid_host);
    Kokkos::deep_copy(tfld, tfld_host);
    Kokkos::deep_copy(cmfdqr, cmfdqr_host);
    Kokkos::deep_copy(nrain, nrain_host);
    Kokkos::deep_copy(nevapr, nevapr_host);
    Kokkos::deep_copy(xhnm, xhnm_host);


    // working var inputs
    ColumnView xgas2, xgas3, delz, xh2o2, xso2, xliq, rain, precip, xhen_h2o2,
        xhen_hno3, xhen_so2, t_factor, xk0_hno3, xk0_so2, so2_diss;

    // initialize internal veriables
    std::vector<Real> vector0(pver, 0);
    auto xgas2_host = View1DHost(vector0.data(), pver);
    auto xgas3_host = View1DHost(vector0.data(), pver);
    auto delz_host = View1DHost(vector0.data(), pver);
    auto xh2o2_host = View1DHost(vector0.data(), pver);
    auto xso2_host = View1DHost(vector0.data(), pver);
    auto xliq_host = View1DHost(vector0.data(), pver);
    auto rain_host = View1DHost(vector0.data(), pver);
    auto precip_host = View1DHost(vector0.data(), pver);
    auto xhen_h2o2_host = View1DHost(vector0.data(), pver);
    auto xhen_hno3_host = View1DHost(vector0.data(), pver);
    auto xhen_so2_host = View1DHost(vector0.data(), pver);
    auto t_factor_host = View1DHost(vector0.data(), pver);
    auto xk0_hno3_host = View1DHost(vector0.data(), pver);
    auto xk0_so2_host = View1DHost(vector0.data(), pver);
    auto so2_diss_host = View1DHost(vector0.data(), pver);
    

    xgas2 = haero::testing::create_column_view(pver);
    xgas3 = haero::testing::create_column_view(pver);
    delz = haero::testing::create_column_view(pver);
    xh2o2 = haero::testing::create_column_view(pver);
    xso2 = haero::testing::create_column_view(pver);
    xliq = haero::testing::create_column_view(pver);
    rain = haero::testing::create_column_view(pver);
    precip = haero::testing::create_column_view(pver);
    xhen_h2o2 = haero::testing::create_column_view(pver);
    xhen_hno3 = haero::testing::create_column_view(pver);
    xhen_so2 = haero::testing::create_column_view(pver);
    t_factor = haero::testing::create_column_view(pver);
    xk0_hno3 = haero::testing::create_column_view(pver);
    xk0_so2 = haero::testing::create_column_view(pver);
    so2_diss = haero::testing::create_column_view(pver);

    Kokkos::deep_copy(xgas2, xgas2_host);
    Kokkos::deep_copy(xgas3, xgas3_host);
    Kokkos::deep_copy(delz, delz_host);
    Kokkos::deep_copy(xh2o2, xh2o2_host);
    Kokkos::deep_copy(xso2, xso2_host);
    Kokkos::deep_copy(xliq, xliq_host);
    Kokkos::deep_copy(rain, rain_host);
    Kokkos::deep_copy(precip, precip_host);
    Kokkos::deep_copy(xhen_h2o2, xhen_h2o2_host);
    Kokkos::deep_copy(xhen_hno3, xhen_hno3_host);
    Kokkos::deep_copy(xhen_so2, xhen_so2_host);
    Kokkos::deep_copy(t_factor, t_factor_host);
    Kokkos::deep_copy(xk0_hno3, xk0_hno3_host);
    Kokkos::deep_copy(xk0_so2, xk0_so2_host);
    Kokkos::deep_copy(so2_diss, so2_diss_host);

    ColumnView het_rates[gas_pcnst];
    ColumnView tmp_hetrates[gas_pcnst];
    ColumnView qin[gas_pcnst];
    View1DHost het_rates_host[gas_pcnst];
    View1DHost tmp_hetrates_host[gas_pcnst];
    View1DHost qin_host[gas_pcnst];

    for (int mm = 0; mm < gas_pcnst; ++mm) {
      het_rates[mm] = haero::testing::create_column_view(pver);
      tmp_hetrates[mm] = haero::testing::create_column_view(pver);
      qin[mm] = haero::testing::create_column_view(pver);

      het_rates_host[mm] = View1DHost("het_rates_host", pver);
      tmp_hetrates_host[mm] = View1DHost("tmp_hetrates_host", pver);
      qin_host[mm] = View1DHost("qin_host", pver);
    }

    int count = 0;
    for (int mm = 0; mm < gas_pcnst; ++mm) {
      for (int kk = 0; kk < pver; ++kk) {
        het_rates_host[mm](kk) = 0.0;
        tmp_hetrates_host[mm](kk) = 0.0;
        qin_host[mm](kk) = qin_in[count];
        count++;
      }
    }

    // transfer data to GPU.
    for (int mm = 0; mm < gas_pcnst; ++mm) {
      Kokkos::deep_copy(het_rates[mm], het_rates_host[mm]);
      Kokkos::deep_copy(tmp_hetrates[mm], tmp_hetrates_host[mm]);
      Kokkos::deep_copy(qin[mm], qin_host[mm]);
    }

    auto team_policy = ThreadTeamPolicy(1u, Kokkos::AUTO);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
          mo_sethet::sethet(team, het_rates, rlat, press, zmid, phis, tfld,
                            cmfdqr, nrain, nevapr, delt, xhnm, qin, t_factor,
                            xk0_hno3, xk0_so2, so2_diss, xgas2,
                            xgas3, delz, xh2o2, xso2, xliq, rain, precip,
                            xhen_h2o2, xhen_hno3, xhen_so2, tmp_hetrates,
                            spc_h2o2_ndx, spc_so2_ndx, h2o2_ndx, so2_ndx,
                            h2so4_ndx, gas_wetdep_cnt, wetdep_map);
        });

    // transfer data to GPU.
    for (int mm = 0; mm < gas_pcnst; ++mm) {
      Kokkos::deep_copy(het_rates_host[mm], het_rates[mm]);
    }
    std::vector<Real> het_rates_out(pver * gas_pcnst);
    count = 0;
    for (int mm = 0; mm < gas_pcnst; ++mm) {
      for (int kk = 0; kk < pver; ++kk) {
        het_rates_out[count] = het_rates_host[mm](kk);
        count++;
      }
    }

    output.set("het_rates", het_rates_out);
  });
}