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
void compute_qsub_from_gcm_and_qsub_of_other_subarea(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    // Ensemble parameters
    // Declare array of strings for input names
    std::string input_arrays[] = {"lcompute", "f_a",    "f_b",  "qgcm",
                                  "qsub_a",   "qsub_b", "ncnst"};

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

    const auto lcompute_ = input.get_array("lcompute");
    const Real f_a = input.get_array("f_a")[0];
    const Real f_b = input.get_array("f_b")[0];
    const auto qgcm_ = input.get_array("qgcm");
    const auto jclea_ = input.get_array("jclea")[0];
    const auto jcldy_ = input.get_array("jcldy")[0];
    const auto qsub_a_ = input.get_array("qsub_a");
    const auto qsub_b_ = input.get_array("qsub_b");

    bool lcompute[gas_pcnst];
    Real qgcm[gas_pcnst];
    const int jclea = jclea_ - 1;
    const int jcldy = jcldy_ - 1;

    View2DHost qsub_a_h("qsub_a", gas_pcnst, subarea_max);
    Kokkos::deep_copy(qsub_a_h, 0.0);
    View2D qsub_a_d("qsub_a", gas_pcnst, subarea_max);
    View2DHost qsub_b_h("qsub_b", gas_pcnst, subarea_max);
    Kokkos::deep_copy(qsub_b_h, 0.0);
    View2D qsub_b_d("qsub_b", gas_pcnst, subarea_max);

    for (int i = 0; i < gas_pcnst; ++i) {
      lcompute[i] = lcompute_[i];
      qgcm[i] = qgcm_[i];
    }
    // NOTE: the data file only provides qsub_{a,b} for the below j-indices
    // However, compute_qsub_from_gcm_and_qsub_of_other_subarea() requires an
    // array subarea_max x gas_pcnst, so we fill in zeros
    for (int i = 0; i < gas_pcnst; ++i) {
      qsub_b_h(i, jcldy) = qsub_b_[i];
      qsub_a_h(i, jclea) = qsub_a_[i];
    }

    Kokkos::deep_copy(qsub_a_d, qsub_a_h);
    Kokkos::deep_copy(qsub_b_d, qsub_b_h);

    auto team_policy = ThreadTeamPolicy(1u, Kokkos::AUTO);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
          Real qsub_a_in[gas_pcnst][subarea_max];
          Real qsub_b_in[gas_pcnst][subarea_max];
          for (int j = 0; j < subarea_max; ++j) {
            for (int i = 0; i < gas_pcnst; ++i) {
              qsub_a_in[i][j] = qsub_a_d(i, j);
              qsub_b_in[i][j] = qsub_b_d(i, j);
            }
          }
          mam4::microphysics::compute_qsub_from_gcm_and_qsub_of_other_subarea(
              lcompute, f_a, f_b, qgcm, jclea, jcldy, qsub_a_in, qsub_b_in);
          for (int j = 0; j < subarea_max; ++j) {
            for (int i = 0; i < gas_pcnst; ++i) {
              qsub_a_d(i, j) = qsub_a_in[i][j];
              qsub_b_d(i, j) = qsub_b_in[i][j];
            }
          }
        });

    Kokkos::deep_copy(qsub_a_h, qsub_a_d);
    Kokkos::deep_copy(qsub_b_h, qsub_b_d);

    std::vector<Real> qsub_a_out;
    std::vector<Real> qsub_b_out;
    // only take the values of qsub_{a,b} that were changed within the fxn,
    // fill with original values
    for (int i = 0; i < gas_pcnst; ++i) {
      qsub_a_out.push_back(qsub_a_h(i, jclea));
      qsub_b_out.push_back(qsub_b_h(i, jcldy));
    }
    output.set("qsub_a", qsub_a_out);
    output.set("qsub_b", qsub_b_out);
  });
}
