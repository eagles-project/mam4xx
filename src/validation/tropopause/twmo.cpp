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

void twmo(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    using View1DHost = typename HostType::view_1d<Real>;
    using ColumnView = haero::ColumnView;
    constexpr int nlev = mam4::nlev;

    const auto pmid1d_in = input.get_array("pmid1d");
    const auto temp1d_in = input.get_array("temp1d");
    const Real plimu = input.get_array("plimu")[0];
    const Real pliml = input.get_array("pliml")[0];
    const Real gam = input.get_array("gam")[0];
    Real trp_in = input.get_array("trp")[0];

    ColumnView pmid1d, temp1d;
    auto pmid1d_host = View1DHost((Real *)pmid1d_in.data(), nlev);
    auto temp1d_host = View1DHost((Real *)temp1d_in.data(), nlev);
    pmid1d = haero::testing::create_column_view(nlev);
    temp1d = haero::testing::create_column_view(nlev);
    Kokkos::deep_copy(pmid1d, pmid1d_host);
    Kokkos::deep_copy(temp1d, temp1d_host);

    DeviceType::view_1d<Real> trp_out_val("Return from Device", 1);
    Kokkos::parallel_for(
        "twmo", 1, KOKKOS_LAMBDA(int i) {
          Real trp_use = trp_in;
          tropopause::twmo(temp1d, pmid1d, plimu, pliml, gam, trp_use);
          trp_out_val[0] = trp_use;
        });

    auto trp_host = Kokkos::create_mirror_view(trp_out_val);
    Kokkos::deep_copy(trp_host, trp_out_val);
    const Real trp_out = trp_host[0];

    output.set("trp", trp_out);
  });
}
