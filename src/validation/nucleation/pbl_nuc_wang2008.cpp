// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/nucleation.hpp>

#include <haero/constants.hpp>
// #include <mam4xx/mam4_types.hpp>
#include <skywalker.hpp>
#include <validation.hpp>

using namespace skywalker;
using namespace mam4;
using namespace haero;
void pbl_nuc_wang2008(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    // Ensemble parameters
    // Declare array of strings for input names
    std::string input_arrays[] = {"life",     "pbl_nuc_wang2008_actual",
                                  "ratenucl", "rateloge",
                                  "cnum_tot", "cnum_h2so4",
                                  "cnum_nh3", "radius_cluster_nm"};

    // Iterate over input_arrays and error if not in input
    for (std::string name : input_arrays) {
      if (!input.has_array(name.c_str())) {
        std::cerr << "Required name for array: " << name << std::endl;
        exit(1);
      }
    }

    const Real life_ = input.get_array("life")[0];

    const Real so4vol_ = input.get_array("so4vol")[0];
    constexpr int pi_ = haero::pi;
    const int pbl_nuc_wang2008_user_choice_ =
        input.get_array("pbl_nuc_wang2008_user_choice")[0];
    int pbl_nuc_wang2008_actual_ =
        input.get_array("pbl_nuc_wang2008_actual")[0];
    Real ratenucl_ = input.get_array("ratenucl")[0];
    Real rateloge_ = input.get_array("rateloge")[0];
    Real cnum_tot_ = input.get_array("cnum_tot")[0];
    Real cnum_h2so4_ = input.get_array("cnum_h2so4")[0];
    Real cnum_nh3_ = input.get_array("cnum_nh3")[0];
    Real radius_cluster_nm_ = input.get_array("radius_cluster_nm")[0];

    nucleation::pbl_nuc_wang2008(
        so4vol_, pi_, pbl_nuc_wang2008_user_choice_,
        adjust_factor_pbl_ratenucl_, pbl_nuc_wang2008_actual_, ratenucl_,
        rateloge_, cnum_tot_, cnum_h2so4_, cnum_nh3_, radius_cluster_nm_);

    output.set("ans", life);

    output.set("pbl_nuc_wang2008_actual", pbl_nuc_wang2008_actual_out);
    output.set("ratenucl", ratenucl_out);
    output.set("rateloge", rateloge_out);
    output.set("cnum_tot", cnum_tot_out);
    output.set("cnum_h2so4", cnum_h2so4_out);
    output.set("cnum_nh3", cnum_nh3_out);
    output.set("radius_cluster_nm", radius_cluster_nm_out);
  });
}
