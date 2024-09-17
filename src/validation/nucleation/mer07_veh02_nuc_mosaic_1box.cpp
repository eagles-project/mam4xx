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
void mer07_veh02_nuc_mosaic_1box(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    // Ensemble parameters
    // Declare array of strings for input names
    std::string input_arrays[] = {
        "dtnuc",    "temp_in",        "rh_in",        "press_in",
        "zm_in",    "pblh_in",        "qh2so4_cur",   "qh2so4_avg",
        "qnh3_cur", "h2so4_uptkrate", "mw_so4a_host", "newnuc_method_flagaa",
        "nsize",    "maxd_asize",     "dplom_sect",   "dphim_sect",
        "ldiagaa",  "rgas",           "avogad",       "mw_so4a",
        "mw_nh4a"};

    // Iterate over input_arrays and error if not in input
    for (std::string name : input_arrays) {
      if (!input.has_array(name.c_str())) {
        std::cerr << "Required name for array: " << name << std::endl;
        exit(1);
      }
    }

    // dtnuc
    // temp_in
    // rh_in
    // press_in
    // zm_in
    // pblh_in
    // qh2so4_cur
    // qh2so4_avg
    // qnh3_cur
    // h2so4_uptkrate
    // mw_so4a_host
    // newnuc_method_flagaa
    // nsize
    // maxd_asize
    // dplom_sect
    // dphim_sect
    // ldiagaa
    // rgas
    // avogad
    // mw_so4a
    // mw_nh4a


  });
}
