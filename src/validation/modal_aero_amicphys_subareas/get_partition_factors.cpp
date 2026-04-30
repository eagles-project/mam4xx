// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/mam4.hpp>
#include <validation.hpp>

using namespace skywalker;

void get_partition_factors(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    // Ensemble parameters
    // Declare array of strings for input names
    std::string input_arrays[] = {"fcldy", "fclea", "qgcm_cldbrn",
                                  "qgcm_intrst"};

    // Iterate over input_arrays and error if not in input
    for (std::string name : input_arrays) {
      if (!input.has_array(name.c_str())) {
        std::cerr << "Required name for array: " << name << std::endl;
        exit(1);
      }
    }

    using View1D = typename mam4::DeviceType::view_1d<Real>;
    using View1DHost = typename mam4::HostType::view_1d<Real>;

    const Real qgcm_intrst = input.get_array("qgcm_intrst")[0];
    const Real qgcm_cldbrn = input.get_array("qgcm_cldbrn")[0];
    const Real fcldy = input.get_array("fcldy")[0];
    const Real fclea = input.get_array("fclea")[0];

    View1DHost factor_clea_h("factor_clea", 1);
    View1D factor_clea_d("factor_clea", 1);
    Kokkos::deep_copy(factor_clea_h, 0.0);
    Kokkos::deep_copy(factor_clea_d, 0.0);
    View1DHost factor_cldy_h("factor_cldy", 1);
    View1D factor_cldy_d("factor_cldy", 1);
    Kokkos::deep_copy(factor_cldy_h, 0.0);
    Kokkos::deep_copy(factor_cldy_d, 0.0);

    auto team_policy = mam4::ThreadTeamPolicy(1u, mam4::testing::team_size);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const mam4::ThreadTeam &team) {
          Real factor_clea_in = 0.0;
          Real factor_cldy_in = 0.0;
          mam4::microphysics::get_partition_factors(
              qgcm_intrst, qgcm_cldbrn, fcldy, fclea, factor_clea_in,
              factor_cldy_in);
          factor_clea_d(0) = factor_clea_in;
          factor_cldy_d(0) = factor_cldy_in;
        });

    Kokkos::deep_copy(factor_clea_h, factor_clea_d);
    Kokkos::deep_copy(factor_cldy_h, factor_cldy_d);
    const Real factor_clea_out = factor_clea_h(0);
    const Real factor_cldy_out = factor_cldy_h(0);

    output.set("factor_clea", factor_clea_out);
    output.set("factor_cldy", factor_cldy_out);
  });
}
