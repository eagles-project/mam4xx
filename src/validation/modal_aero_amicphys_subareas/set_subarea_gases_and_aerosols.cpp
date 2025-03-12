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
void set_subarea_gases_and_aerosols(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    // Ensemble parameters
    // Declare array of strings for input names
    std::string input_arrays[] = {"loffset",   "nsubarea", "jclea",    "jcldy",
                                  "fclea",     "fcldy",    "qgcm1",    "qgcm2",
                                  "qqcwgcm2",  "qgcm3",    "qqcwgcm3", "ncnst",
                                  "maxsubarea"};

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

    const auto nsubarea_ = input.get_array("nsubarea")[0];
    const auto jclea_ = input.get_array("jclea")[0];
    const auto jcldy_ = input.get_array("jcldy")[0];
    const auto fclea = input.get_array("fclea")[0];
    const auto fcldy = input.get_array("fcldy")[0];
    const auto qgcm1_ = input.get_array("qgcm1");
    const auto qgcm2_ = input.get_array("qgcm2");
    const auto qqcwgcm2_ = input.get_array("qqcwgcm2");
    const auto qgcm3_ = input.get_array("qgcm3");
    const auto qqcwgcm3_ = input.get_array("qqcwgcm3");

    const int nsubarea = nsubarea_;
    const int jclea = jclea_;
    const int jcldy = jcldy_;

    EKAT_ASSERT(nsubarea <= subarea_max);

    Real qgcm1[gas_pcnst];
    Real qgcm2[gas_pcnst];
    Real qqcwgcm2[gas_pcnst];
    Real qgcm3[gas_pcnst];
    Real qqcwgcm3[gas_pcnst];
    for (int i = 0; i < gas_pcnst; ++i) {
      qgcm1[i] = qgcm1_[i];
      qgcm2[i] = qgcm2_[i];
      qqcwgcm2[i] = qqcwgcm2_[i];
      qgcm3[i] = qgcm3_[i];
      qqcwgcm3[i] = qqcwgcm3_[i];
    }

    View2DHost qsub1_h("qsub1", gas_pcnst, subarea_max);
    View2D qsub1_d("qsub1", gas_pcnst, subarea_max);
    Kokkos::deep_copy(qsub1_h, 0.0);
    Kokkos::deep_copy(qsub1_d, 0.0);
    View2DHost qsub2_h("qsub2", gas_pcnst, subarea_max);
    View2D qsub2_d("qsub2", gas_pcnst, subarea_max);
    Kokkos::deep_copy(qsub2_h, 0.0);
    Kokkos::deep_copy(qsub2_d, 0.0);
    View2DHost qqcwsub2_h("qqcwsub2", gas_pcnst, subarea_max);
    View2D qqcwsub2_d("qqcwsub2", gas_pcnst, subarea_max);
    Kokkos::deep_copy(qqcwsub2_h, 0.0);
    Kokkos::deep_copy(qqcwsub2_d, 0.0);
    View2DHost qsub3_h("qsub3", gas_pcnst, subarea_max);
    View2D qsub3_d("qsub3", gas_pcnst, subarea_max);
    Kokkos::deep_copy(qsub3_h, 0.0);
    Kokkos::deep_copy(qsub3_d, 0.0);
    View2DHost qqcwsub3_h("qqcwsub3", gas_pcnst, subarea_max);
    View2D qqcwsub3_d("qqcwsub3", gas_pcnst, subarea_max);
    Kokkos::deep_copy(qqcwsub3_h, 0.0);
    Kokkos::deep_copy(qqcwsub3_d, 0.0);

    auto team_policy = ThreadTeamPolicy(1u, Kokkos::AUTO);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
          Real qsub1[gas_pcnst][subarea_max] = {{0.0}};
          Real qsub2[gas_pcnst][subarea_max] = {{0.0}};
          Real qqcwsub2[gas_pcnst][subarea_max] = {{0.0}};
          Real qsub3[gas_pcnst][subarea_max] = {{0.0}};
          Real qqcwsub3[gas_pcnst][subarea_max] = {{0.0}};
          mam4::microphysics::set_subarea_gases_and_aerosols(
              nsubarea, jclea, jcldy, fclea, fcldy, qgcm1, qgcm2, qqcwgcm2,
              qgcm3, qqcwgcm3, qsub1, qsub2, qqcwsub2, qsub3, qqcwsub3);
          for (int j = 0; j < subarea_max; ++j) {
            for (int i = 0; i < gas_pcnst; ++i) {
              qsub1_d(i, j) = qsub1[i][j];
              qsub2_d(i, j) = qsub2[i][j];
              qqcwsub2_d(i, j) = qqcwsub2[i][j];
              qsub3_d(i, j) = qsub3[i][j];
              qqcwsub3_d(i, j) = qqcwsub3[i][j];
            }
          }
        });

    Kokkos::deep_copy(qsub1_h, qsub1_d);
    Kokkos::deep_copy(qsub2_h, qsub2_d);
    Kokkos::deep_copy(qqcwsub2_h, qqcwsub2_d);
    Kokkos::deep_copy(qsub3_h, qsub3_d);
    Kokkos::deep_copy(qqcwsub3_h, qqcwsub3_d);

    std::vector<Real> qsub1_out;
    std::vector<Real> qsub2_out;
    std::vector<Real> qqcwsub2_out;
    std::vector<Real> qsub3_out;
    std::vector<Real> qqcwsub3_out;

    // NOTE: we go j = [1, 3) here due to the weird indexing convention in
    // microphysics::set_subarea_gases_and_aerosols(), which may be changed in
    // the future
    for (int j = 1; j < subarea_max; ++j) {
      for (int i = 0; i < gas_pcnst; ++i) {
        qsub1_out.push_back(qsub1_h(i, j));
        qsub2_out.push_back(qsub2_h(i, j));
        qqcwsub2_out.push_back(qqcwsub2_h(i, j));
        qsub3_out.push_back(qsub3_h(i, j));
        qqcwsub3_out.push_back(qqcwsub3_h(i, j));
      }
    }

    output.set("qsub1", qsub1_out);
    output.set("qsub2", qsub2_out);
    output.set("qqcwsub2", qqcwsub2_out);
    output.set("qsub3", qsub3_out);
    output.set("qqcwsub3", qqcwsub3_out);
  });
}
