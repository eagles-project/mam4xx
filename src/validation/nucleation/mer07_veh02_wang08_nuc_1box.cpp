// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/nucleation.hpp>
#include <validation.hpp>

#include <haero/constants.hpp>
#include <skywalker.hpp>

#include <iostream>

using namespace skywalker;
using namespace mam4;

void mer07_veh02_wang08_nuc_1box(Ensemble *ensemble) {
  constexpr Real pi = Constants::pi;

  // Figure out settings for binary/ternary nucleation and planetary boundary
  // layer treatment
  Settings settings = ensemble->settings();
  int newnuc_method_user_choice =
      std::stoi(settings.get("newnuc_method_user_choice"));
  int pbl_nuc_wang2008_user_choice =
      std::stoi(settings.get("pbl_nuc_wang2008_user_choice"));

  if ((newnuc_method_user_choice != 2) && (newnuc_method_user_choice != 3)) {
    std::stringstream ss;
    ss << "Invalid newnuc_method_user_choice: " << newnuc_method_user_choice
       << std::endl;
    throw skywalker::Exception(ss.str());
  }

  if ((pbl_nuc_wang2008_user_choice < 0) ||
      (pbl_nuc_wang2008_user_choice > 2)) {
    std::stringstream ss;
    ss << "Invalid pbl_nuc_wang2008_user_choice: "
       << pbl_nuc_wang2008_user_choice << std::endl;
    throw skywalker::Exception(ss.str());
  }

  // Other parameters.
  Real adjust_factor_bin_tern_ratenucl = 1.0;
  Real adjust_factor_pbl_ratenucl = 1.0;
  Real ln_nuc_rate_cutoff = -13.82;

  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    // Parse input

    Real temp = input.get("temperature");
    Real relhumnn = input.get("relative_humidity");

    Real nh3ppt = input.get("xi_nh3");
    Real so4vol = input.get("c_h2so4");

    Real zmid = input.get("height");
    Real pblh = input.get("planetary_boundary_layer_height");

    // Call the nucleation function on device.
    Real dnclusterdt;
    Kokkos::parallel_reduce(
        "mer07_veh02_wang08_nuc_1box", 1,
        KOKKOS_LAMBDA(int i, Real &dnclusterdt) {
          int newnuc_method_actual, pbl_nuc_wang2008_actual;
          Real temp_dnclusterdt, rateloge, cnum_h2so4, cnum_nh3, radius_cluster;
          nucleation::mer07_veh02_wang08_nuc_1box(
              newnuc_method_user_choice, newnuc_method_actual,
              pbl_nuc_wang2008_user_choice, pbl_nuc_wang2008_actual,
              ln_nuc_rate_cutoff, adjust_factor_bin_tern_ratenucl,
              adjust_factor_pbl_ratenucl, pi, so4vol, nh3ppt, temp, relhumnn,
              zmid, pblh, temp_dnclusterdt, rateloge, cnum_h2so4, cnum_nh3,
              radius_cluster);
          dnclusterdt = temp_dnclusterdt;
        },
        Kokkos::Max<Real>(dnclusterdt));

    // Process output
    Real J_cm3s = dnclusterdt * 1e-6;
    output.set("nucleation_rate", J_cm3s);
  });
}
