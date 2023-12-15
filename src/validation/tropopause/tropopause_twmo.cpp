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

void tropopause_twmo(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    using View1DHost = typename HostType::view_1d<Real>;
    constexpr int pver = mam4::nlev;

    const auto pmid_in = input.get_array("pmid");
    const auto pint_in = input.get_array("pint");
    const auto temp_in = input.get_array("temp");
    const auto zm_in = input.get_array("zm");
    const auto zi_in = input.get_array("zi");

    ColumnView pmid, pint, temp, zm, zi;

    auto pmid_host = View1DHost((Real *)pmid_in.data(), pver);
    auto pint_host = View1DHost((Real *)pint_in.data(), pver);
    auto temp_host = View1DHost((Real *)temp_in.data(), pver);
    auto zm_host = View1DHost((Real *)zm_in.data(), pver);
    auto zi_host = View1DHost((Real *)zi_in.data(), pver);
    pmid = haero::testing::create_column_view(pver);
    pint = haero::testing::create_column_view(pver);
    temp = haero::testing::create_column_view(pver);
    zm = haero::testing::create_column_view(pver);
    zi = haero::testing::create_column_view(pver);
    Kokkos::deep_copy(pmid, pmid_host);
    Kokkos::deep_copy(pint, pint_host);
    Kokkos::deep_copy(temp, temp_host);
    Kokkos::deep_copy(zm, zm_host);
    Kokkos::deep_copy(zi, zi_host);

    DeviceType::view_1d<int> tropLev("tropLev", 1);
    Kokkos::parallel_for(
        "tropopause_twmo", 1, KOKKOS_LAMBDA(int i) {
          tropopause::tropopause_twmo(pmid, pint, temp, zm, zi, tropLev(0));
        });

    auto tropLev_host = Kokkos::create_mirror_view(tropLev);
    Kokkos::deep_copy(tropLev_host, tropLev);

    // C++ to Fortran indexing
    output.set("tropLev", tropLev_host(0) + 1);
  });
}