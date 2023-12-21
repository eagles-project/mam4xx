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

void find_ktop(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    using View1DHost = typename HostType::view_1d<Real>;
    using ColumnView = haero::ColumnView;
    constexpr int pver = mam4::nlev;

    const Real rlat = input.get_array("rlat")[0];
    const auto press_in = input.get_array("press");

    ColumnView press;
    auto press_host = View1DHost((Real *)press_in.data(), pver);
    press = haero::testing::create_column_view(pver);
    Kokkos::deep_copy(press, press_host);

    DeviceType::view_1d<int> ktop_out("Return from Device", 1);
    Kokkos::parallel_for(
        "find_ktop", 1, KOKKOS_LAMBDA(int i) {
          int ktop = 0;
          find_ktop(rlat, press, ktop);
          ktop_out(0) = ktop;
        });

    output.set("ktop", ktop_out(0) + 1);
  });
}