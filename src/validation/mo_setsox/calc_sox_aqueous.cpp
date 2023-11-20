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
void calc_sox_aqueous(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    // Ensemble parameters
    // Declare array of strings for input names
    std::string input_variables[] = {"dt"};

    std::string input_arrays[] = {
        "const0", "dtime", "h2o2g",  "heo3", "heso2", "modal_aerosols",
        "o3g",    "patm",  "rah2o2", "rao3", "so2g",  "t_factor",
        "xh2o2",  "xhnm",  "xlwc",   "xso2", "xso4",  "xso4_init"};

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
    const Real const0 = input.get_array("const0")[0];
    const Real dt = input.get_array("dtime")[0];
    const Real h2o2g = input.get_array("h2o2g")[0];
    const Real heo3 = input.get_array("heo3")[0];
    const Real heso2 = input.get_array("heso2")[0];
    const bool modal_aerosols = input.get_array("modal_aerosols")[0];
    const Real o3g = input.get_array("o3g")[0];
    const Real patm = input.get_array("patm")[0];
    const Real rah2o2 = input.get_array("rah2o2")[0];
    const Real rao3 = input.get_array("rao3")[0];
    const Real so2g = input.get_array("so2g")[0];
    const Real t_factor = input.get_array("t_factor")[0];
    const Real xhnm = input.get_array("xhnm")[0];
    const Real xlwc = input.get_array("xlwc")[0];

    Real xso2 = input.get_array("xso2")[0];
    Real xso4 = input.get_array("xso4")[0];
    Real xso4_init = input.get_array("xso4_init")[0];
    Real xh2o2 = input.get_array("xh2o2")[0];

    Real xdelso4hp;

    mam4::mo_setsox::calc_sox_aqueous(modal_aerosols, rah2o2, h2o2g, so2g, o3g,
                                      rao3, patm, dt, t_factor, xlwc, const0,
                                      xhnm, heo3, heso2, xso2, xso4, xso4_init,
                                      xh2o2, xdelso4hp);

    output.set("xso2", xso2);
    output.set("xso4", xso4);
    output.set("xso4_init", xso4_init);
    output.set("xh2o2", xh2o2);
    output.set("xdelso4hp_ik", xdelso4hp);
  });
}
