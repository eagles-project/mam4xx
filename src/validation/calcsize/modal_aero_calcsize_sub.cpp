// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/mam4.hpp>

#include <mam4xx/calcsize.hpp>
#include <skywalker.hpp>
#include <validation.hpp>

using namespace skywalker;
using namespace mam4;
using namespace haero;

void modal_aero_calcsize_sub(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    constexpr int pcnst = aero_model::pcnst;
    constexpr int pver = ndrop::pver;
    constexpr int ntot_amode = AeroConfig::num_modes();

    using View2D = DeviceType::view_2d<Real>;

    auto state_q_db = input.get_array("state_q");
    auto qqcw_db = input.get_array("qqcw");
    const auto dt = input.get_array("dt")[0];

    View2D state_q("state_q", pver, pcnst);
    View2D ptend("ptend", pver, pcnst);
    View2D dqqcwdt("dqqcwdt", pver, pcnst);
    mam4::validation::convert_1d_vector_to_2d_view_device(state_q_db, state_q);
    View2D qqcw("qqcw", pver, pcnst);
    auto qqcw_host = create_mirror_view(qqcw);

    int count = 0;
    for (int kk = 0; kk < pver; ++kk) {
      for (int i = 0; i < pcnst; ++i) {
        qqcw_host(kk, i) = qqcw_db[count];
        count++;
      }
    }
    Kokkos::deep_copy(qqcw, qqcw_host);
    View2D dgnumdry_m("dgnumdry_m", pver, ntot_amode);
    View2D dgncur_c("dgncur_c", pver, ntot_amode);
    mam4::modal_aer_opt::CalcsizeData cal_data;
    cal_data.initialize();

    auto team_policy = ThreadTeamPolicy(1u, Kokkos::AUTO);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
          // FIXME: top_lev is set to 1 in calcsize ?
          const int top_lev = 0; // 1( in fortran )

          for (int kk = top_lev; kk < pver; ++kk) {
            const auto state_q_k = Kokkos::subview(state_q, kk, Kokkos::ALL());
            auto ptend_k = Kokkos::subview(ptend, kk, Kokkos::ALL());
            const auto qqcw_k = Kokkos::subview(qqcw, kk, Kokkos::ALL());
            const auto dgncur_i =
                Kokkos::subview(dgnumdry_m, kk, Kokkos::ALL());
            // Real dgncur_c[ntot_amode] = {};
            const auto dgncur_c_k =
                Kokkos::subview(dgncur_c, kk, Kokkos::ALL());
            // Real dqqcwdt[pcnst] = {};
            auto dqqcwdt_k = Kokkos::subview(dqqcwdt, kk, Kokkos::ALL());
            modal_aero_calcsize::modal_aero_calcsize_sub(
                state_q_k, // in
                qqcw_k,    // in/out
                dt, cal_data, dgncur_i, dgncur_c_k, ptend_k, dqqcwdt_k);
          } // k
        });

    constexpr Real zero = 0;
    std::vector<Real> dgnumdry_m_out(pver * ntot_amode, zero);
    mam4::validation::convert_2d_view_device_to_1d_vector(dgnumdry_m,
                                                          dgnumdry_m_out);

    Kokkos::deep_copy(qqcw_host, qqcw);
    count = 0;
    for (int kk = 0; kk < pver; ++kk) {
      for (int i = 0; i < pcnst; ++i) {
        qqcw_db[count] = qqcw_host(kk, i);
        count++;
      }
    }

    output.set("qqcw", qqcw_db);
    output.set("dgnumdry_m", dgnumdry_m_out);
  });
}
