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
void cloud_mod(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {

    using View1DHost = typename HostType::view_1d<Real>;
    using View1D = typename DeviceType::view_1d<Real>;

    // number of vertical points.
    // validation test from standalone mo_photo.
    const auto zen_angle = input.get_array("zen_angle")[0];
    const auto clouds_db = input.get_array("clouds");
    const auto lwc_db = input.get_array("lwc");
    const auto delp_db = input.get_array("delp");
    const auto srf_alb = input.get_array("srf_alb")[0];
    // const auto  = input.get_array("");
    constexpr Real zero = 0;
    std::cout << "clouds_db.size() " << clouds_db.size()<< "\n";

    auto clouds_host = View1DHost((Real *)clouds_db.data(), clouds_db.size());
    View1D clouds("clouds", clouds_db.size());
    Kokkos::deep_copy(clouds, clouds_host);

    auto lwc_host = View1DHost((Real *)lwc_db.data(), lwc_db.size());
    const auto lwc = View1D("lwc", lwc_db.size());
    Kokkos::deep_copy(lwc, lwc_host);

    auto delp_host = View1DHost((Real *)delp_db.data(), delp_db.size());
    const auto delp = View1D("delp", delp_db.size());
    Kokkos::deep_copy(delp, delp_host);

    View1D eff_alb("eff_alb", pver);
    View1D cld_mult("cld_mult", pver);

    auto team_policy = ThreadTeamPolicy(1u, Kokkos::AUTO);
#if 1
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
    cloud_mod(team, zen_angle, clouds.data(), lwc.data(), delp.data(),
              srf_alb, //  in
              eff_alb.data(), cld_mult.data());
    });
#endif
    std::vector<Real> eff_alb_db(pver, zero);
    std::vector<Real> cld_mult_db(pver, zero);

    auto eff_alb_host = View1DHost((Real *)eff_alb_db.data(), pver);
    auto cld_mult_host = View1DHost((Real *)cld_mult_db.data(), pver);
    Kokkos::deep_copy(eff_alb_host, eff_alb);
    Kokkos::deep_copy(cld_mult_host, cld_mult);


    output.set("eff_alb", eff_alb_db);
    output.set("cld_mult", cld_mult_db);
  });
}
