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
void set_subarea_qmass_for_cldbrn_aerosols(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    // Ensemble parameters
    // Declare array of strings for input names
    std::string input_arrays[] = {
        "fcldy",      "jcldy",      "jclea",  "lmassptrcw_amode",
        "loffset",    "maxsubarea", "ncnst",  "nspec_amode",
        "ntot_amode", "qqcwgcm",    "qqcwsub"};

    // Iterate over input_arrays and error if not in input
    for (std::string name : input_arrays) {
      if (!input.has_array(name.c_str())) {
        std::cerr << "Required name for array: " << name << std::endl;
        exit(1);
      }
    }

    using View2D = typename DeviceType::view_2d<Real>;
    using View2DHost = typename HostType::view_2d<Real>;

    using mam4::gas_chemistry::gas_pcnst;
    constexpr int subarea_max = microphysics::maxsubarea();

    const auto ncnst_ = input.get_array("ncnst")[0];
    const int ncnst = ncnst_;
    EKAT_ASSERT(ncnst == gas_pcnst);

    const auto jclea_ = input.get_array("jclea")[0];
    const auto jcldy_ = input.get_array("jcldy")[0];
    const auto fcldy = input.get_array("fcldy")[0];
    const auto qqcwgcm_ = input.get_array("qqcwgcm");

    const int jclea = jclea_;
    const int jcldy = jcldy_;

    Real qqcwgcm[gas_pcnst];
    for (int i = 0; i < gas_pcnst; ++i) {
      qqcwgcm[i] = qqcwgcm_[i];
    }

    View2DHost qqcwsub_h("qqcwsub", gas_pcnst, subarea_max);
    View2D qqcwsub_d("qqcwsub", gas_pcnst, subarea_max);
    Kokkos::deep_copy(qqcwsub_h, 0.0);
    Kokkos::deep_copy(qqcwsub_d, 0.0);

    auto team_policy = ThreadTeamPolicy(1u, Kokkos::AUTO);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
          Real qqcwsub[gas_pcnst][subarea_max] = {{0.0}};
          mam4::microphysics::set_subarea_qmass_for_cldbrn_aerosols(
              jclea, jcldy, fcldy, qqcwgcm, qqcwsub);
          for (int j = 0; j < subarea_max; ++j) {
            for (int i = 0; i < gas_pcnst; ++i) {
              qqcwsub_d(i, j) = qqcwsub[i][j];
            }
          }
        });

    Kokkos::deep_copy(qqcwsub_h, qqcwsub_d);
    std::vector<Real> qqcwsub_out;

    // NOTE: we go j = [1, 3) here due to the weird indexing convention in
    // microphysics::set_subarea_gases_and_aerosols(), which may be changed in
    // the future
    for (int j = 1; j < subarea_max; ++j) {
      for (int i = 0; i < gas_pcnst; ++i) {
        qqcwsub_out.push_back(qqcwsub_h(i, j));
      }
    }

    output.set("qqcwsub", qqcwsub_out);
  });
}
