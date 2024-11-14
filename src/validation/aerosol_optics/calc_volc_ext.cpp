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

void calc_volc_ext(Ensemble *ensemble) {
  using mam4::nlev;
  ensemble->process([=](const Input &input, Output &output) {
    using View1D = DeviceType::view_1d<Real>;
    using View1DHost = typename HostType::view_1d<Real>;

    const int trop_level = int(input.get_array("trop_level")[0]) - 1;
    const auto state_zm_db = input.get_array("state_zm");
    const auto ext_cmip6_sw_db = input.get_array("ext_cmip6_sw");
    const auto extinct_db = input.get_array("extinct");

    auto state_zm_host = View1DHost((Real *)state_zm_db.data(), nlev);
    auto ext_cmip6_sw_host = View1DHost((Real *)ext_cmip6_sw_db.data(), nlev);
    auto extinct_host = View1DHost((Real *)extinct_db.data(), nlev);

    ColumnView state_zm, ext_cmip6_sw, extinct;
    state_zm = haero::testing::create_column_view(nlev);
    ext_cmip6_sw = haero::testing::create_column_view(nlev);
    extinct = haero::testing::create_column_view(nlev);
    Kokkos::deep_copy(state_zm, state_zm_host);
    Kokkos::deep_copy(ext_cmip6_sw, ext_cmip6_sw_host);
    Kokkos::deep_copy(extinct, extinct_host);

    View1D tropopause_m("tropopause_m", 1);

    auto team_policy = ThreadTeamPolicy(1u, Kokkos::AUTO);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
          calc_volc_ext(trop_level, state_zm, ext_cmip6_sw, extinct,
                        tropopause_m(0));
        });

    auto tropopause_m_host = Kokkos::create_mirror_view(tropopause_m);
    Kokkos::deep_copy(tropopause_m_host, tropopause_m);
    Kokkos::deep_copy(extinct_host, extinct);

    std::vector<Real> extinct_out(extinct_host.data(),
                                  extinct_host.data() + nlev);
    output.set("tropopause_m", std::vector<Real>(1, tropopause_m_host(0)));
    output.set("extinct", extinct_out);
  });
}
