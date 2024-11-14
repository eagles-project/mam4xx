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
  using mam4::nlev;
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
    auto clouds_host = View1DHost((Real *)clouds_db.data(), nlev);
    View1D clouds("clouds", nlev);
    Kokkos::deep_copy(clouds, clouds_host);

    auto lwc_host = View1DHost((Real *)lwc_db.data(), nlev);
    const auto lwc = View1D("lwc", nlev);
    Kokkos::deep_copy(lwc, lwc_host);

    auto delp_host = View1DHost((Real *)delp_db.data(), nlev);
    const auto delp = View1D("delp", nlev);
    Kokkos::deep_copy(delp, delp_host);

    View1D eff_alb("eff_alb", nlev);
    View1D cld_mult("cld_mult", nlev);

    auto team_policy = ThreadTeamPolicy(1u, Kokkos::AUTO);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
          cloud_mod(zen_angle, clouds, lwc, delp,
                    srf_alb, //  in
                    eff_alb.data(), cld_mult.data());
        });
    std::vector<Real> eff_alb_db(nlev, zero);
    std::vector<Real> cld_mult_db(nlev, zero);

    auto eff_alb_host = View1DHost((Real *)eff_alb_db.data(), nlev);
    auto cld_mult_host = View1DHost((Real *)cld_mult_db.data(), nlev);
    Kokkos::deep_copy(eff_alb_host, eff_alb);
    Kokkos::deep_copy(cld_mult_host, cld_mult);

    output.set("eff_alb", eff_alb_db);
    output.set("cld_mult", cld_mult_db);
  });
}
