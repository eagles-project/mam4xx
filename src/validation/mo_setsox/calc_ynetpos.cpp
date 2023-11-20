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
void calc_ynetpos(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    // Ensemble parameters
    // Declare array of strings for input names
    std::string input_variables[] = {"dt"};

    std::string input_arrays[] = {"yph",       "fact1_so2", "fact2_so2",
                                  "fact3_so2", "fact4_so2", "Eco2",
                                  "Eh2o",      "Eso4",      "so4_fact"};

    // Iterate over input_variables and error if not in input
    for (std::string name : input_variables) {
      if (!input.has(name.c_str())) {
        std::cerr << "Required name for variable: " << name << std::endl;
        exit(1);
      }
    }
    // Iterate over input_arrays and error if not in input
    for (std::string name : input_arrays) {
      if (!input.has_array(name.c_str())) {
        std::cerr << "Required name for array: " << name << std::endl;
        exit(1);
      }
    }

    // Parse input
    const Real yph = input.get_array("yph")[0];
    const Real fact1_so2 = input.get_array("fact1_so2")[0];
    const Real fact2_so2 = input.get_array("fact2_so2")[0];
    const Real fact3_so2 = input.get_array("fact3_so2")[0];
    const Real fact4_so2 = input.get_array("fact4_so2")[0];
    const Real Eco2 = input.get_array("Eco2")[0];
    const Real Eh2o = input.get_array("Eh2o")[0];
    const Real Eso4 = input.get_array("Eso4")[0];
    const Real so4_fact = input.get_array("so4_fact")[0];

    Real xph;
    Real ynetpos;

    mam4::mo_setsox::calc_ynetpos(yph, fact1_so2, fact2_so2, fact3_so2,
                                  fact4_so2, Eco2, Eh2o, Eso4, so4_fact, xph,
                                  ynetpos);

    output.set("xph", xph);
    output.set("ynetpos", ynetpos);
  });
}
