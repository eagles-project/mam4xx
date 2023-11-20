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
void calc_ph_values(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    // Ensemble parameters
    // Declare array of strings for input names
    std::string input_variables[] = {"dt"};

    std::string input_arrays[] = {"temperature", "patm", "xlwc",  "t_factor",
                                  "xso2",        "xso4", "xhnm",  "so4_fact",
                                  "Ra",          "xkw",  "const0"};

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

    const mam4::mo_setsox::Config setsox_config_;
    const Real co2g = setsox_config_.co2g;
    const int itermax = setsox_config_.itermax;

    // Parse input
    const Real temperature = input.get_array("temperature")[0];
    const Real patm = input.get_array("patm")[0];
    const Real xlwc = input.get_array("xlwc")[0];
    const Real t_factor = input.get_array("t_factor")[0];
    const Real xso2 = input.get_array("xso2")[0];
    const Real xso4 = input.get_array("xso4")[0];
    const Real xhnm = input.get_array("xhnm")[0];
    const Real so4_fact = input.get_array("so4_fact")[0];
    const Real Ra = input.get_array("Ra")[0];
    const Real xkw = input.get_array("xkw")[0];
    const Real const0 = input.get_array("const0")[0];

    bool converged;
    Real xph;

    mam4::mo_setsox::calc_ph_values(temperature, patm, xlwc, t_factor, xso2,
                                    xso4, xhnm, so4_fact, Ra, xkw, const0, co2g,
                                    itermax, converged, xph);

    output.set("converged", converged);
    output.set("xph", xph);
  });
}
