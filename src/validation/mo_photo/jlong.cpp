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
using namespace mo_photo;

void jlong(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    using View1DHost = typename HostType::view_1d<Real>;
    using View1D = typename DeviceType::view_1d<Real>;
    const auto alb_in_db = input.get_array("alb_in");
    const Real sza_in = input.get_array("sza_in")[0];
    const auto p_in_db = input.get_array("p_in");
    const auto colo3_in_db = input.get_array("colo3_in");
    const auto t_in_db = input.get_array("t_in");

    auto alb_in_host = View1DHost((Real *)alb_in_db.data(), pver);
    auto p_in_host = View1DHost((Real *)p_in_db.data(), pver);
    auto colo3_in_host = View1DHost((Real *)colo3_in_db.data(), pver);
    auto t_in_host = View1DHost((Real *)t_in_db.data(), pver);

    const auto alb_in = View1D("alb_in", pver);
    const auto p_in = View1D("p_in", pver);
    const auto colo3_in = View1D("colo3_in", pver);
    const auto t_in = View1D("t_in", pver);

    Kokkos::deep_copy(alb_in, alb_in_host);
    Kokkos::deep_copy(p_in, p_in_host);
    Kokkos::deep_copy(colo3_in, colo3_in_host);
    Kokkos::deep_copy(t_in, t_in_host);

    // FIXME; ask for following variables
    const auto sza_db = input.get_array("sza");
    const auto del_sza_db = input.get_array("del_sza");
    const auto alb_db = input.get_array("alb");
    const auto press_db = input.get_array("press");
    const auto del_p_db = input.get_array("del_p");
    const auto colo3_db = input.get_array("colo3");
    const auto o3rat_db = input.get_array("o3rat");
    const auto del_alb_db = input.get_array("del_alb");
    const auto del_o3rat_db = input.get_array("del_o3rat");
    const auto etfphot_db = input.get_array("etfphot");

    const int nw = 67;
    const int nump = 15;
    const int numsza = 10;
    const int numcolo3 = 10;
    const int numalb = 10;

    auto sza_host = View1DHost((Real *)sza_db.data(), numsza);
    const auto sza = View1D("sza", numsza);
    Kokkos::deep_copy(sza, sza_host);

    auto del_sza_host = View1DHost((Real *)del_sza_db.data(), numsza - 1);
    const auto del_sza = View1D("del_sza", numsza - 1);
    Kokkos::deep_copy(del_sza, del_sza_host);

    auto alb_host = View1DHost((Real *)alb_db.data(), numalb);
    const auto alb = View1D("alb", numalb);
    Kokkos::deep_copy(alb, alb_host);

    auto press_host = View1DHost((Real *)press_db.data(), nump);
    const auto press = View1D("press", nump);
    Kokkos::deep_copy(press, press_host);

    auto del_p_host = View1DHost((Real *)del_p_db.data(), nump - 1);
    const auto del_p = View1D("del_p", nump - 1);
    Kokkos::deep_copy(del_p, del_p_host);

    auto colo3_host = View1DHost((Real *)colo3_db.data(), nump);
    const auto colo3 = View1D("colo3", nump);
    Kokkos::deep_copy(colo3, colo3_host);

    auto o3rat_host = View1DHost((Real *)o3rat_db.data(), numcolo3);
    const auto o3rat = View1D("o3rat", numcolo3);
    Kokkos::deep_copy(o3rat, o3rat_host);

    auto del_alb_host = View1DHost((Real *)del_alb_db.data(), numalb - 1);
    const auto del_alb = View1D("del_alb", numalb - 1);
    Kokkos::deep_copy(del_alb, del_alb_host);

    auto del_o3rat_host = View1DHost((Real *)del_o3rat_db.data(), numcolo3 - 1);
    const auto del_o3rat = View1D("del_o3rat", numcolo3 - 1);
    Kokkos::deep_copy(del_o3rat, del_o3rat_host);

    auto etfphot_host = View1DHost((Real *)etfphot_db.data(), nw);
    const auto etfphot = View1D("etfphot", nw);
    Kokkos::deep_copy(etfphot, etfphot_host);

    // auto _host = View1DHost((Real *).data(), );
    // const auto  = View1D("", );
    // Kokkos::deep_copy(, );

    const int np_xs = 10;
    const int nt = 201;
    const int numj = 10;

    const auto prs_db = input.get_array("prs");
    const auto dprs_db = input.get_array("dprs");

    auto prs_host = View1DHost((Real *)prs_db.data(), np_xs);
    const auto prs = View1D("prs", np_xs);
    Kokkos::deep_copy(prs, prs_host);

    auto dprs_host = View1DHost((Real *)dprs_db.data(), np_xs - 1);
    const auto dprs = View1D("dprs", np_xs - 1);
    Kokkos::deep_copy(dprs, dprs_host);

    View5D rsf_tab("rsf_tab", nw, nump, numsza, numcolo3, numalb);
    auto rsf_tab_1 = Kokkos::subview(rsf_tab, Kokkos::ALL(), 1, Kokkos::ALL(),
                                     Kokkos::ALL(), Kokkos::ALL());

    auto rsf_tab_2 = Kokkos::subview(rsf_tab, Kokkos::ALL(), Kokkos::ALL(), 6,
                                     Kokkos::ALL(), Kokkos::ALL());

    auto rsf_tab_3 = Kokkos::subview(rsf_tab, Kokkos::ALL(), Kokkos::ALL(),
                                     Kokkos::ALL(), 7, Kokkos::ALL());

    auto rsf_tab_4 = Kokkos::subview(rsf_tab, Kokkos::ALL(), Kokkos::ALL(),
                                     Kokkos::ALL(), Kokkos::ALL(), 3);

    auto rsf_tab_5 = Kokkos::subview(rsf_tab, 0, Kokkos::ALL(), Kokkos::ALL(),
                                     Kokkos::ALL(), Kokkos::ALL());

    auto rsf_tab_6 = Kokkos::subview(rsf_tab, 9, Kokkos::ALL(), Kokkos::ALL(),
                                     Kokkos::ALL(), Kokkos::ALL());

    Kokkos::deep_copy(rsf_tab, 0.1);
    Kokkos::deep_copy(rsf_tab_1, 2.0);
    Kokkos::deep_copy(rsf_tab_2, 3.0);
    Kokkos::deep_copy(rsf_tab_3, 1.0);
    Kokkos::deep_copy(rsf_tab_4, 0.8);
    Kokkos::deep_copy(rsf_tab_5, 6.0);
    Kokkos::deep_copy(rsf_tab_6, 1e-2);

    View2D rsf("rsf", nw, nlev);
    View4D xsqy("xsqy", numj, nw, nt, np_xs);
    View2D xswk("xswk", numj, nw);

    Kokkos::deep_copy(xsqy, 0.1);

    View2D j_long("j_long", numj, pver);

    auto team_policy = ThreadTeamPolicy(1u, 1u);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
          Real psum_l[nw] = {};
          Real psum_u[nw] = {};

          jlong(sza_in, alb_in.data(), p_in.data(), t_in.data(),
                colo3_in.data(), xsqy, sza.data(), del_sza.data(), alb.data(),
                press.data(), del_p.data(), colo3.data(), o3rat.data(),
                del_alb.data(), del_o3rat.data(), etfphot.data(),
                rsf_tab, // in
                prs.data(), dprs.data(), nw, nump, numsza, numcolo3, numalb,
                np_xs, numj,
                j_long, // output
                // work arrays
                rsf, xswk, psum_l, psum_u);
        });

    const Real zero = 0;
    std::vector<Real> jlong_out(pver * numj, zero);

    auto j_long_host = Kokkos::create_mirror_view(j_long);
    Kokkos::deep_copy(j_long_host, j_long);

    int count = 0;
    for (int j = 0; j < pver; ++j) {
      for (int i = 0; i < numj; ++i) {
        jlong_out[count] = j_long_host(i, j);
        count += 1;
      }
    }

    output.set("j_long", jlong_out);
  });
}
