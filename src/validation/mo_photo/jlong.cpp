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
    // validation test from standalone mo_photo.
    const auto alb_in = input.get_array("alb_in");
    const auto sza_in = input.get_array("sza_in")[0];
    const auto p_in = input.get_array("p_in");
    const auto colo3_in = input.get_array("colo3_in");
    const auto t_in = input.get_array("t_in");

    // FIXME; ask for following variables
    const auto sza = input.get_array("sza");
    const auto del_sza = input.get_array("del_sza");
    const auto alb = input.get_array("alb");
    const auto press = input.get_array("press");
    const auto del_p = input.get_array("del_p");
    const auto colo3 = input.get_array("colo3");
    const auto o3rat = input.get_array("o3rat");
    const auto del_alb = input.get_array("del_alb");
    const auto del_o3rat = input.get_array("del_o3rat");
    const auto etfphot = input.get_array("etfphot");
    const auto prs = input.get_array("prs");
    const auto dprs = input.get_array("dprs");
    //

    const int nw = 67;
    const int nump = 15;
    const int numsza = 10;
    const int numcolo3 = 10;
    const int numalb = 10;

    View5D rsf_tab("rsf_tab", nw, nump, numsza, numcolo3, numalb);
    auto rsf_tab_1 = Kokkos::subview(rsf_tab, Kokkos::ALL(), 1,
                                     Kokkos::ALL(),Kokkos::ALL(), Kokkos::ALL());

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


    Kokkos::deep_copy(rsf_tab,0.1);
    Kokkos::deep_copy(rsf_tab_1,2.0);
    Kokkos::deep_copy(rsf_tab_2,3.0);
    Kokkos::deep_copy(rsf_tab_3,1.0);
    Kokkos::deep_copy(rsf_tab_4,0.8);
    Kokkos::deep_copy(rsf_tab_5,6.0);
    Kokkos::deep_copy(rsf_tab_6,1e-2);

    const int np_xs = 10;
    const int nt = 201;
    const int numj = 10;
    View2D rsf("rsf",nw, nlev);
    View4D xsqy("xsqy",numj, nw, nt, np_xs);
    View2D xswk("xswk",numj, nw);

    Kokkos::deep_copy(xsqy,0.1);

    Real psum_l[nw] = {};
    Real psum_u[nw] = {};

    View2D j_long("j_long",numj,pver );

    jlong(sza_in, alb_in.data(), p_in.data(), t_in.data(), colo3_in.data(),
          xsqy, sza.data(), del_sza.data(), alb.data(), press.data(),
          del_p.data(), colo3.data(), o3rat.data(), del_alb.data(),
          del_o3rat.data(), etfphot.data(),
          rsf_tab, // in
          prs.data(), dprs.data(),
          nw, nump, numsza, numcolo3, numalb,
          np_xs,numj,
          j_long, // output
          // work arrays
          rsf, xswk, psum_l, psum_u);


    const Real zero=0;
    std::vector<Real> jlong_out(pver*numj,zero);

    int count=0;
    for (int j = 0; j < pver; ++j)
    {
      for (int i = 0; i < numj; ++i)
      {
        jlong_out[count] = j_long(i,j);
        count += 1;
      }
    }

    output.set("j_long", jlong_out);

  });
}
