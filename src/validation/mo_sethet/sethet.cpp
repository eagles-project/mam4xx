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

//constexpr Real m2km = 1.0e-3;    // convert m to km
//constexpr Real rga = 1.0 / haero::Constants::gravity;

void sethet(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    using View1DHost = typename HostType::view_1d<Real>;
    using View2D = DeviceType::view_2d<Real>;
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
    ColumnView delz, xh2o2, xso2, xliq, rain, precip, xhen_h2o2,
        xhen_hno3, xhen_so2, t_factor, xk0_hno3, xk0_so2, so2_diss;

    // initialize internal veriables
    std::vector<Real> vector0(pver, 0);
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

    // Real het_rates[pver][gas_pcnst] = {0.0};
    // Real tmp_hetrates[pver][gas_pcnst] = {0.0};
    Real qin[pver][gas_pcnst];
    //View1DHost het_rates_host[pver];
    //View1DHost tmp_hetrates_host[pver];
    //View1DHost qin_host[gas_pcnst];

    // for (int mm = 0; mm < gas_pcnst; ++mm) {
    //   qin[mm] = haero::testing::create_column_view(pver);
    //   qin_host[mm] = View1DHost("qin_host", pver);
    // }

    // for (int kk = 0; kk < pver; ++kk) {
    //   for (int mm = 0; mm < gas_pcnst; ++mm) {
    //     het_rates[kk][mm] = 0.0;
    //     tmp_hetrates[kk][mm] = 0.0; 
    //   }
    // }

    int count = 0;
    for (int mm = 0; mm < gas_pcnst; ++mm) {
      for (int kk = 0; kk < pver; ++kk) {
        //het_rates_host[kk](mm) = 0.0;
        //tmp_hetrates_host[kk](mm) = 0.0;
        qin[kk][mm] = qin_in[count];
        count++;
      }
    }

    // transfer data to GPU.
    // for (int mm = 0; mm < gas_pcnst; ++mm) {
    //   Kokkos::deep_copy(qin[mm], qin_host[mm]);
    // }

    auto team_policy = ThreadTeamPolicy(1u, Kokkos::AUTO);

    auto ktop_out = haero::testing::create_column_view(1);
    auto ktop_out_host = View1DHost("ktop_out_host", 1);
    ktop_out_host(0) = 0;
    Kokkos::deep_copy(ktop_out, ktop_out_host);
    Kokkos::parallel_for(
        "find_ktop", 1, KOKKOS_LAMBDA(int i) {
          int ktop = 0;
          find_ktop(rlat, press, ktop);
          ktop_out(0) = ktop;
        });

    Kokkos::deep_copy(ktop_out_host, ktop_out);
    int ktop = (int) ktop_out_host(0);
    
    Real total_rain = 0.0; // total rain rate (both pos and neg) in the column
    Real total_pos = 0.0;  // total positive rain rate in the column

    auto total_rain_out = haero::testing::create_column_view(1);
    auto total_pos_out = haero::testing::create_column_view(1);
    auto total_rain_host = View1DHost("total_rain_host", 1);
    auto total_pos_host = View1DHost("total_pos_host", 1);
    total_rain_host(0) = 0;
    total_pos_host(0) = 0;

    Kokkos::deep_copy(total_rain_out, total_rain_host);
    Kokkos::deep_copy(total_pos_out, total_pos_host);
    Kokkos::parallel_for(
      team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
        calc_precip_rescale(team, cmfdqr, nrain, nevapr, total_rain_out(0), total_pos_out(0), precip);
      });
    Kokkos::deep_copy(total_rain_host, total_rain_out);
    Kokkos::deep_copy(total_pos_host, total_pos_out);

    total_rain = (Real) total_rain_host(0);
    total_pos = (Real) total_pos_host(0);

    

    Real zsurf = m2km * phis * rga;
    //move to 241 (with sethet call)
    Kokkos::parallel_for(
      team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
          calc_delz(team, ktop, delz, zmid, zsurf);
          team.team_barrier();
      });

    // Kokkos::deep_copy(precip_host, precip);
    // Kokkos::deep_copy(xhnm_host, xhnm);
    for(int m = 0; m < gas_pcnst; m++) {
      for(int k = 0; k < pver; k++) {
      std::cout << "qin @ k " << k << " m " << m << " = " << qin[k][m] << std::endl;
      }
    }
    //barrier TODO
    

    // View1D het_rates_d("het_rates", pver * gas_pcnst);
    // Kokkos::deep_copy(het_rates_d, 0.0);

    View2D het_rates("het_rates", pver, gas_pcnst); 
    View2D tmp_hetrates("het_rates", pver, gas_pcnst); 
    //View2D qin("het_rates", pver, gas_pcnst); 
    //qin and tmp_hetrates TODO
    
    // View2D tmp_hetrates("het_rates", pver, gas_pcnst);
    Kokkos::deep_copy(het_rates, 0.0);
    Kokkos::deep_copy(tmp_hetrates, 0.0);

    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
          // Real het_rates[pver][gas_pcnst];
          // Real tmp_hetrates[pver][gas_pcnst]; 
          //int kk = 1;
          for(int kk = 0; kk < pver; kk++) { 
              auto het_rates_at_kk = ekat::subview(het_rates, kk);
              auto tmp_hetrates_at_kk = ekat::subview(tmp_hetrates, kk);
              mo_sethet::sethet(
                  // team, 
                  het_rates_at_kk.data(), rlat, press(kk), zmid(kk), zsurf, phis, tfld(kk), cmfdqr(kk), nrain(kk),
                  nevapr(kk), delt, xhnm(kk), qin[kk], t_factor(kk), xk0_hno3(kk), xk0_so2(kk), so2_diss(kk),
                  delz(kk), xh2o2(kk), xso2(kk), xliq(kk), rain(kk), precip(kk), xhen_h2o2(kk),
                  xhen_hno3(kk), xhen_so2(kk), tmp_hetrates_at_kk.data(), spc_h2o2_ndx, spc_so2_ndx,
                  h2o2_ndx, so2_ndx, h2so4_ndx, gas_wetdep_cnt, wetdep_map, total_rain, total_pos, ktop);
      }
      
      // int i = 0;
      // for (int mm = 0; mm < gas_pcnst; ++mm) {
      //   for (int kk = 0; kk < pver; ++kk) {
      //     het_rates_d(i) = het_rates[kk][mm];
      //     i++;
      //   }
      // } 
    });
    //team.team_barrier();
    // transfer data to GPU.
    // for (int kk = 0; kk < pver; kk++) {
    //   Kokkos::deep_copy(het_rates_host[kk], het_rates[kk]);
    // }
    // std::vector<Real> het_rates_out(pver * gas_pcnst);
    // count = 0;
    // for (int mm = 0; mm < gas_pcnst; ++mm) {
    //   for (int kk = 0; kk < pver; ++kk) {
    //     het_rates_out[count] = het_rates[kk][mm];
    //     count++;
    //   }
    // }

    std::vector<Real> het_rates_out(gas_pcnst * pver);
    mam4::validation::convert_2d_view_device_to_1d_vector(het_rates,
                                                          het_rates_out);
    // auto het_rates_host = View1DHost((Real *)het_rates_out.data(), gas_pcnst * pver);
    // Kokkos::deep_copy(het_rates_host, het_rates_d);

    output.set("het_rates", het_rates_out);
  });
}