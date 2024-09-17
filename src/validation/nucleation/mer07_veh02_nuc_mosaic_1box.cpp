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
        "nsize",    "dplom_sect",     "dphim_sect",   "ldiagaa",
        "rgas",     "avogad",         "mw_so4a",      "mw_nh4a"};

    // Iterate over input_arrays and error if not in input
    for (std::string name : input_arrays) {
      if (!input.has_array(name.c_str())) {
        std::cerr << "Required name for array: " << name << std::endl;
        exit(1);
      }
    }

    const int newnuc_method_flagaa_ =
        input.get_array("newnuc_method_flagaa")[0];
    const Real dtnuc_ = input.get_array("dtnuc")[0];
    const Real temp_in_ = input.get_array("temp_in")[0];
    const Real rh_in_ = input.get_array("rh_in")[0];
    const Real press_in_ = input.get_array("press_in")[0];
    const Real zm_in_ = input.get_array("zm_in")[0];
    const Real pblh_in_ = input.get_array("pblh_in")[0];
    const Real qh2so4_cur_ = input.get_array("qh2so4_cur")[0];
    const Real qh2so4_avg_ = input.get_array("qh2so4_avg")[0];
    const Real qnh3_cur_ = input.get_array("qnh3_cur")[0];
    const Real h2so4_uptkrate_ = input.get_array("h2so4_uptkrate")[0];
    const Real mw_so4a_host_ = input.get_array("mw_so4a_host")[0];
    const int nsize_ = input.get_array("nsize")[0];
    const Real dplom_sect_ = input.get_array("dplom_sect")[0];
    const Real dphim_sect_ = input.get_array("dphim_sect")[0];
    const int ldiagaa_ = input.get_array("ldiagaa")[0];
    const Real rgas_ = input.get_array("rgas")[0];
    const Real avogad_ = input.get_array("avogad")[0];
    const Real mw_nh4a_ = input.get_array("mw_nh4a")[0];
    const Real mw_so4a_ = input.get_array("mw_so4a")[0];
    const Real pi = haero::Constants::pi;

    int newnuc_method_flagaa;
    if (newnuc_method_flagaa_ == 11) {
      newnuc_method_flagaa = 1;
    } else if (newnuc_method_flagaa_ == 12) {
      newnuc_method_flagaa = 2;
    } else {
      std::cerr << "Undefined value for parameter: newnuc_method_flagaa"
                << std::endl;
    }

    int isize_nuc;
    Real qnuma_del;
    Real qso4a_del;
    Real qnh4a_del;
    Real qh2so4_del;
    Real qnh3_del;
    Real dens_nh4so4a;
    Real dnclusterdt;

    nucleation::mer07_veh02_wang08_nuc_1box(
        newnuc_method_flagaa, dtnuc_, temp_in_, rh_in_, press_in_, zm_in_,
        pblh_in_, qh2so4_cur_, qh2so4_avg_, qnh3_cur_, h2so4_uptkrate_,
        mw_so4a_host_, nsize_, dplom_sect_, dphim_sect_, ldiagaa_, rgas_,
        avogad_, mw_nh4a_, mw_so4a_, pi, isize_nuc, qnuma_del, qso4a_del,
        qnh4a_del, qh2so4_del, qnh3_del, dens_nh4so4a, dnclusterdt);

    output.set("isize_nuc", isize_nuc);
    output.set("qnuma_del", qnuma_del);
    output.set("qso4a_del", qso4a_del);
    output.set("qnh4a_del", qnh4a_del);
    output.set("qh2so4_del", qh2so4_del);
    output.set("dens_nh4so4a", dens_nh4so4a);
    output.set("dnclusterdt", dnclusterdt);
  });
}
