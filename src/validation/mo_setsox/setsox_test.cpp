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
void setsox_test(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    // Ensemble parameters
    // Declare array of strings for input names
    std::string input_variables[] = {"dt"};

    std::string input_arrays[] = {
        "ncol", "loffset", "dtime",  "press", "pdel", "tfld", "mbar",
        "lwc",  "cldfrc",  "cldnum", "xhnm",  "qcw",  "qin"};

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
    const Real dt = input.get_array("dtime")[0];
    const int loffset = input.get_array("loffset")[0];
    const Real press = input.get_array("press")[0];
    const Real pdel = input.get_array("pdel")[0];
    const Real tfld = input.get_array("tfld")[0];
    const Real mbar = input.get_array("mbar")[0];
    const Real lwc = input.get_array("lwc")[0];
    const Real cldfrc = input.get_array("cldfrc")[0];
    const Real cldnum = input.get_array("cldnum")[0];
    const Real xhnm = input.get_array("xhnm")[0];

    auto qcw = input.get_array("qcw");
    auto qin = input.get_array("qin");

    const mam4::mo_setsox::Config setsox_config_;

    mam4::mo_setsox::setsox_single_level(loffset, dt, press, pdel, tfld, mbar,
                                         lwc, cldfrc, cldnum, xhnm,
                                         setsox_config_, &qcw[0], &qin[0]);

    output.set("qcw", qcw);
    output.set("qin", qin);
  });
}
