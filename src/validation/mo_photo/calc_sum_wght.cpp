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
void calc_sum_wght(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    // validation test from standalone mo_photo.
    const auto dels = input.get_array("dels");
    const auto wrk0 = input.get_array("wrk0")[0];
    const int iz = int(input.get_array("iz")[0]);
    const int is = int(input.get_array("is")[0]);
    const int iv = int(input.get_array("iv")[0]);
    const int ial = int(input.get_array("ial")[0]);
    const auto rsf_tab_1d = input.get_array("rsf_tab");
    const Real rsf_tab[nw][nump][numsza][numcolo3][numalb] = {};

    const Real zero = 0;
    std::vector<Real> psum(nw, zero);

    calc_sum_wght(dels.data(), wrk0, // in
                  iz, is, iv, ial,   // in
                  rsf_tab,           // in
                  psum.data());

    // C++ indexing to fortran indexing
    output.set("psum", psum);
  });
}
