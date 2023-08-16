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
void find_index(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    // validation test from standalone mo_photo.
    const auto var_in = input.get_array("var_in");
    const auto var_len = input.get_array("var_len")[0];
    const auto var_min = input.get_array("var_min")[0];

    int idx_out = 0;
    find_index(var_in.data(), var_len,
               var_min, //  in
               idx_out);
    // C++ indexing to fortran indexing
    output.set("idx_out", idx_out + 1);
  });
}
