#include <haero/constants.hpp>
#include <haero/mam4/nucleation_impl.hpp>

#include <validation.hpp>

#include <iostream>
#include <skywalker.hpp>

using namespace skywalker;
using namespace haero;
using namespace haero::mam4;

void mer07_veh02_wang08_nuc_1box(Ensemble* ensemble) {
  constexpr Real pi = Constants::pi;

  // Figure out settings for binary/ternary nucleation and planetary boundary
  // layer treatment
  Settings settings = ensemble->settings();
  int newnuc_method_user_choice = std::stoi(settings.get("newnuc_method_user_choice"));
  int pbl_nuc_wang2008_user_choice = std::stoi(settings.get("pbl_nuc_wang2008_user_choice"));

  if ((newnuc_method_user_choice != 2) &&
      (newnuc_method_user_choice != 3)) {
    std::stringstream ss;
    ss << "Invalid newnuc_method_user_choice: " << newnuc_method_user_choice
       << std::endl;
    throw skywalker::Exception(ss.str());
  }

  if ((pbl_nuc_wang2008_user_choice < 0) ||
      (pbl_nuc_wang2008_user_choice > 2)) {
    std::stringstream ss;
    ss << "Invalid pbl_nuc_wang2008_user_choice: " << pbl_nuc_wang2008_user_choice
       << std::endl;
    throw skywalker::Exception(ss.str());
  }

  // Other parameters.
  Real adjust_factor_bin_tern_ratenucl = 1.0;
  Real adjust_factor_pbl_ratenucl = 1.0;
  Real ln_nuc_rate_cutoff = -13.82;

  // Run the ensemble.
  ensemble->process([=](const Input& input, Output& output) {
    // Parse input

    Pack temp     = input.get("temperature");
    Pack relhumnn = input.get("relative_humidity");

    Pack nh3ppt   = input.get("xi_nh3");
    Pack so4vol   = input.get("c_h2so4");

    Pack zmid     = input.get("height");
    Real pblh     = input.get("planetary_boundary_layer_height");

    // Call the nucleation function on device.
    IntPack newnuc_method_actual, pbl_nuc_wang2008_actual;
    Pack dnclusterdt, rateloge, cnum_h2so4, cnum_nh3, radius_cluster;
    Kokkos::parallel_for("mer07_veh02_wang08_nuc_1box", 1,
      [&]KOKKOS_FUNCTION(int i) {
        mer07_veh02_wang08_nuc_1box(
          newnuc_method_user_choice, newnuc_method_actual,
          pbl_nuc_wang2008_user_choice, pbl_nuc_wang2008_actual,
          ln_nuc_rate_cutoff,
          adjust_factor_bin_tern_ratenucl, adjust_factor_pbl_ratenucl,
          pi, so4vol, nh3ppt, temp, relhumnn, zmid, pblh,
          dnclusterdt, rateloge, cnum_h2so4, cnum_nh3, radius_cluster);
      });

    // Process output
    Real J_cm3s = dnclusterdt[0] * 1e-6;
    output.set("nucleation_rate", J_cm3s);
  });
}
