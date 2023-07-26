// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <iostream>
#include <mam4xx/aero_modes.hpp>
#include <mam4xx/mam4.hpp>
#include <mam4xx/water_uptake.hpp>
#include <skywalker.hpp>
#include <validation.hpp>

using namespace skywalker;
using namespace mam4;

void makoh_quartic(Ensemble *ensemble) {
  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    EKAT_REQUIRE_MSG(input.has_array("p0"), "Required name: p0");
    EKAT_REQUIRE_MSG(input.has_array("p1"), "Required name: p1");
    EKAT_REQUIRE_MSG(input.has_array("p2"), "Required name: p2");
    EKAT_REQUIRE_MSG(input.has_array("p3"), "Required name: p3");

    auto p0 = input.get_array("p0");
    auto p1 = input.get_array("p1");
    auto p2 = input.get_array("p2");
    auto p3 = input.get_array("p3");

    Kokkos::complex<Real> cx[4] = {};

    water_uptake::makoh_quartic(cx, p3.data()[0], p2.data()[0], p1.data()[0],
                                p0.data()[0]);

    std::vector<Real> real_cx(4, 0.0);
    std::vector<Real> imag_cx(4, 0.0);
    for (int i = 0; i < 4; ++i) {
      real_cx[i] = cx[i].real();
      imag_cx[i] = cx[i].imag();
    }

    output.set("real_cx", real_cx);
    output.set("imag_cx", imag_cx);
  });
}
