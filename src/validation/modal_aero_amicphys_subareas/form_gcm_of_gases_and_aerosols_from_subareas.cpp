// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <catch2/catch.hpp>

#include <mam4xx/mam4.hpp>

#include <mam4xx/aero_config.hpp>
#include <skywalker.hpp>
#include <validation.hpp>

using namespace skywalker;
using namespace mam4;
using namespace haero;
void form_gcm_of_gases_and_aerosols_from_subareas(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    // Ensemble parameters
    // Declare array of strings for input names
    std::string input_arrays[] = {"nsubarea", "ncldy_subarea", "afracsub",
                                  "qsub",     "qqcwsub",       "qqcwgcm_old",
                                  "ncnst",    "maxsubarea"};

    // Iterate over input_arrays and error if not in input
    for (std::string name : input_arrays) {
      if (!input.has_array(name.c_str())) {
        std::cerr << "Required name for array: " << name << std::endl;
        exit(1);
      }
    }

    using View1D = typename DeviceType::view_1d<Real>;
    using View1DHost = typename HostType::view_1d<Real>;

    constexpr int subarea_max = microphysics::maxsubarea();

    using mam4::gas_chemistry::gas_pcnst;

    EKAT_ASSERT(int(input.get_array("ncnst")[0]) == gas_pcnst);

    const auto nsubarea_ = input.get_array("nsubarea")[0];
    const auto ncldy_subarea_ = input.get_array("ncldy_subarea")[0];
    const auto afracsub_ = input.get_array("afracsub");
    const auto qsub_ = input.get_array("qsub");
    const auto qqcwsub_ = input.get_array("qqcwsub");
    const auto qqcwgcm_old_ = input.get_array("qqcwgcm_old");

    const int nsubarea = nsubarea_;
    EKAT_ASSERT(nsubarea <= subarea_max);
    const int ncldy_subarea = ncldy_subarea_;

    Real afracsub[subarea_max] = {0.0};
    Real qsub[gas_pcnst][subarea_max] = {{0.0}};
    Real qqcwsub[gas_pcnst][subarea_max] = {{0.0}};
    Real qqcwgcm_old[gas_pcnst] = {0.0};

    View1DHost qgcm_h("qgcm", gas_pcnst);
    View1D qgcm_d("qgcm", gas_pcnst);
    Kokkos::deep_copy(qgcm_h, 0.0);
    Kokkos::deep_copy(qgcm_d, 0.0);
    View1DHost qqcwgcm_h("qqcwgcm", gas_pcnst);
    View1D qqcwgcm_d("qqcwgcm", gas_pcnst);
    Kokkos::deep_copy(qqcwgcm_h, 0.0);
    Kokkos::deep_copy(qqcwgcm_d, 0.0);

    for (int i = 0; i < gas_pcnst; ++i) {
      qqcwgcm_old[i] = qqcwgcm_old_[i];
    }
    for (int j = 1, n = 0; j < subarea_max; ++j) {
      std::cout << "j = " << j << "\n";
      std::cout << "n = " << n << "\n";
      afracsub[j] = afracsub_[j - 1];
      std::cout << "afracsub[j] = " << afracsub[j] << "\n";
      if (j < nsubarea)
        std::cout << "afracsub_[j] = " << afracsub_[j] << "\n";
      for (int i = 0; i < gas_pcnst; ++i, ++n) {
        qsub[i][j] = qsub_[n];
        qqcwsub[i][j] = qqcwsub_[n];
      }
    }

    auto team_policy = ThreadTeamPolicy(1u, Kokkos::AUTO);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
          Real qgcm[gas_pcnst] = {0.0};
          Real qqcwgcm[gas_pcnst] = {0.0};
          mam4::microphysics::form_gcm_of_gases_and_aerosols_from_subareas(
              nsubarea, ncldy_subarea, afracsub, qsub, qqcwsub, qqcwgcm_old,
              qgcm, qqcwgcm);
          for (int i = 0; i < gas_pcnst; ++i) {
            qgcm_d(i) = qgcm[i];
            qqcwgcm_d(i) = qqcwgcm[i];
          }
        });

    Kokkos::deep_copy(qgcm_h, qgcm_d);
    Kokkos::deep_copy(qqcwgcm_h, qqcwgcm_d);

    std::vector<Real> qgcm_out;
    std::vector<Real> qqcwgcm_out;
    for (int i = 0; i < gas_pcnst; ++i) {
      qgcm_out.push_back(qgcm_h(i));
      qqcwgcm_out.push_back(qqcwgcm_h(i));
    }
    output.set("qgcm", qgcm_out);
    output.set("qqcwgcm", qqcwgcm_out);
  });
}
