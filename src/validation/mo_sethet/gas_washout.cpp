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

void gas_washout(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    using View1DHost = typename HostType::view_1d<Real>;
    using ColumnView = haero::ColumnView;
    constexpr int pver = mam4::nlev;

    const int plev = input.get_array("plev")[0];
    const Real xkgm = input.get_array("xkgm")[0];
    const Real xliq_ik = input.get_array("xliq_ik")[0];
    const auto xhen_i_in = input.get_array("xhen_i");
    const auto tfld_i_in = input.get_array("tfld_i");
    const auto delz_i_in = input.get_array("delz_i");
    const auto xgas_in = input.get_array("xgas");

    ColumnView xhen_i, tfld_i, delz_i, xgas;
    auto xhen_i_host = View1DHost((Real *)xhen_i_in.data(), pver);
    auto tfld_i_host = View1DHost((Real *)tfld_i_in.data(), pver);
    auto delz_i_host = View1DHost((Real *)delz_i_in.data(), pver);
    auto xgas_host = View1DHost((Real *)xgas_in.data(), pver);
    xhen_i = haero::testing::create_column_view(pver);
    tfld_i = haero::testing::create_column_view(pver);
    delz_i = haero::testing::create_column_view(pver);
    xgas = haero::testing::create_column_view(pver);
    Kokkos::deep_copy(xhen_i, xhen_i_host);
    Kokkos::deep_copy(tfld_i, tfld_i_host);
    Kokkos::deep_copy(delz_i, delz_i_host);
    Kokkos::deep_copy(xgas, xgas_host);

    auto team_policy = ThreadTeamPolicy(1u, Kokkos::AUTO);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
          Kokkos::single(Kokkos::PerTeam(team), [=]() {
            gas_washout(team, plev - 1, xkgm, xliq_ik, xhen_i, tfld_i, delz_i,
                        xgas);
          });
        });

    Kokkos::deep_copy(xgas_host, xgas);
    std::vector<Real> xgas_out(pver);
    for (int k = 0; k < pver; k++)
      xgas_out[k] = xgas_host(k);

    output.set("xgas", xgas_out);
  });
}
