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
    // std::vector<Real> precip(pver, zero);
    DeviceType::view_1d<Real> trp_out_val("Return from Device", 1);
    Kokkos::parallel_for(
        "calc_precip_rescale", 1, KOKKOS_LAMBDA(int i) {
          calc_precip_rescale(cmfdqr, nrain, nevapr, precip);
        });

    Kokkos::deep_copy(precip_host, precip);
    std::vector<Real> precip_out(pver);
    for (int k = 0; k < pver; k++)
      precip_out[k] = precip_host(k);

    output.set("precip", precip_out);
  });
}