// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/mam4.hpp>
#include <validation.hpp>

using namespace skywalker;

void find_real_solution(Ensemble *ensemble) {

  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    EKAT_REQUIRE_MSG(input.has("rdry"), "Required name: rdry");
    EKAT_REQUIRE_MSG(input.has_array("real_cx"), "Required name: real_cx");
    EKAT_REQUIRE_MSG(input.has_array("imag_cx"), "Required name: imag_cx");

    auto rdry = input.get("rdry");
    auto real_cx = input.get_array("real_cx");
    auto imag_cx = input.get_array("imag_cx");

    int nsol = 0;
    Real rwet = 0.0;

    Kokkos::complex<Real> cx[4] = {};
    // fill complex array with input values
    for (int i = 0; i < 4; ++i) {
      cx[i].imag() = imag_cx[i];
      cx[i].real() = real_cx[i];
    }

    // Call the function
    mam4::water_uptake::find_real_solution(rdry, cx, rwet, nsol);

    output.set("nsol",
               nsol + 1); // Adding one to account for Fortran vs C indexing
    output.set("rwet", rwet);
  });
}
