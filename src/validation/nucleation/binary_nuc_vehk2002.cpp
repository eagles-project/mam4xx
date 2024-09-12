// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/nucleation.hpp>

// #include <haero/constants.hpp>
// #include <mam4xx/mam4_types.hpp>
#include <skywalker.hpp>
#include <validation.hpp>

using namespace skywalker;
using namespace mam4;
using namespace haero;
void binary_nuc_vehk2002(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    // Ensemble parameters
    // Declare array of strings for input names
    std::string input_arrays[] = {"life",       "temp",     "rh",
                                  "so4vol",     "ratenucl", "rateloge",
                                  "cnum_h2so4", "cnum_tot", "radius_cluster"};

    // Iterate over input_arrays and error if not in input
    for (std::string name : input_arrays) {
      if (!input.has_array(name.c_str())) {
        std::cerr << "Required name for array: " << name << std::endl;
        exit(1);
      }
    }

    const Real life = input.get_array("life")[0];

    // const Real temp = input.get_array("temp")[0];
    // const Real rh = input.get_array("rh")[0];
    // const Real so4vol = input.get_array("so4vol")[0];
    // Real ratenucl = input.get_array("ratenucl")[0];
    // Real rateloge = input.get_array("rateloge")[0];
    // Real cnum_h2so4 = input.get_array("cnum_h2so4")[0];
    // Real cnum_tot = input.get_array("cnum_tot")[0];
    // Real radius_cluster = input.get_array("radius_cluster")[0];

    // nucleation::binary_nuc_vehk2002(temp, rh, so4vol, ratenucl, rateloge,
                                      //  cnum_h2so4, cnum_tot, radius_cluster)

    output.set("ans", life);

    // output.set("ratenucl", ratenucl);
    // output.set("rateloge", rateloge);
    // output.set("cnum_h2so4", cnum_h2so4);
    // output.set("cnum_tot", cnum_tot);
    // output.set("radius_cluster", radius_cluster);
  });
}
