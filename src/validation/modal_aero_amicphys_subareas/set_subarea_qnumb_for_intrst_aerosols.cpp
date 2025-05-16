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
void set_subarea_qnumb_for_intrst_aerosols(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    // Ensemble parameters
    // Declare array of strings for input names
    std::string input_arrays[] = {
        "fcldy",      "fclea", "jcldy",      "jclea",        "loffset",
        "maxsubarea", "ncnst", "ntot_amode", "numptr_amode", "numptrcw_amode",
        "qgcm",       "qgcmx", "qqcwgcm",    "qsubx"};

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

    EKAT_ASSERT(int(input.get_array("ncnst")[0]) == gas_pcnst);

    const auto jclea_ = input.get_array("jclea")[0];
    const auto jcldy_ = input.get_array("jcldy")[0];
    const auto fclea_ = input.get_array("fclea")[0];
    const auto fcldy_ = input.get_array("fcldy")[0];
    const auto qgcm_ = input.get_array("qgcm");
    const auto qqcwgcm_ = input.get_array("qqcwgcm");
    const auto qgcmx_ = input.get_array("qgcmx");

    const auto qsubx_ = input.get_array("qsubx");

    const int jclea = jclea_;
    const int jcldy = jcldy_;
    const Real fclea = fclea_;
    const Real fcldy = fcldy_;
    Real qgcm[gas_pcnst];
    Real qqcwgcm[gas_pcnst];
    Real qgcmx[gas_pcnst];

    for (int i = 0; i < gas_pcnst; ++i) {
      qgcm[i] = qgcm_[i];
      qqcwgcm[i] = qqcwgcm_[i];
      qgcmx[i] = qgcmx_[i];
    }

    View2DHost qsubx_h("qsubx", gas_pcnst, subarea_max);
    View2D qsubx_d("qsubx", gas_pcnst, subarea_max);
    Kokkos::deep_copy(qsubx_h, 0.0);
    Kokkos::deep_copy(qsubx_d, 0.0);

    for (int j = 1, k = 0; j < subarea_max; ++j) {
      for (int i = 0; i < gas_pcnst; ++i, ++k) {
        qsubx_h(i, j) = qsubx_[k];
      }
    }
    Kokkos::deep_copy(qsubx_d, qsubx_h);

    auto team_policy = ThreadTeamPolicy(1u, Kokkos::AUTO);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
          Real qsubx[gas_pcnst][subarea_max] = {{0.0}};
          for (int j = 0; j < subarea_max; ++j) {
            for (int i = 0; i < gas_pcnst; ++i) {
              qsubx[i][j] = qsubx_d(i, j);
            }
          }
          mam4::microphysics::set_subarea_qnumb_for_intrst_aerosols(
              jclea, jcldy, fclea, fcldy, qgcm, qqcwgcm, qgcmx, qsubx);
          for (int j = 0; j < subarea_max; ++j) {
            for (int i = 0; i < gas_pcnst; ++i) {
              qsubx_d(i, j) = qsubx[i][j];
            }
          }
        });

    Kokkos::deep_copy(qsubx_h, qsubx_d);
    std::vector<Real> qsubx_out;

    // NOTE: we go j = [1, 3) here due to the weird indexing convention in
    // microphysics::set_subarea_gases_and_aerosols(), which may be changed in
    // the future
    for (int j = 1; j < subarea_max; ++j) {
      for (int i = 0; i < gas_pcnst; ++i) {
        qsubx_out.push_back(qsubx_h(i, j));
      }
    }

    output.set("qsubx", qsubx_out);
  });
}
