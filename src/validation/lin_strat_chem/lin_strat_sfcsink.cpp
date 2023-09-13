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
using namespace lin_strat_chem;
void lin_strat_sfcsink(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    using View1D = typename DeviceType::view_1d<Real>;
    using View2D = typename DeviceType::view_2d<Real>;
    constexpr Real zero = 0;

    constexpr int ncol = 4;
    constexpr int pver = mam4::nlev;
    const Real delta_t = input.get_array("delta_t")[0];
    const auto pdel_db = input.get_array("pdel");

    View2D pdel("o3col", ncol, pver);
    mam4::validation::convert_1d_vector_to_2d_view_device(pdel_db, pdel);

    const auto o3l_vmr_db = input.get_array("o3l_vmr");
    View2D o3l_vmr("o3l_vmr", ncol, pver);
    mam4::validation::convert_1d_vector_to_2d_view_device(o3l_vmr_db, o3l_vmr);
    const Real o3_sfc = input.get_array("o3_sfc")[0];
    const int o3_lbl = int(input.get_array("o3_lbl")[0]);
    const Real o3_tau = input.get_array("o3_tau")[0];

    View1D o3l_sfcsink("o3l_sfcsink", ncol);

    auto team_policy = ThreadTeamPolicy(ncol, 1u);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
          const int icol = team.league_rank();
          auto pdel_icol = Kokkos::subview(pdel, icol, Kokkos::ALL());
          auto o3l_vmr_icol = Kokkos::subview(o3l_vmr, icol, Kokkos::ALL());

          lin_strat_sfcsink(delta_t, pdel_icol, // in
                            o3l_vmr_icol, o3_sfc, o3_lbl, o3_tau,
                            o3l_sfcsink(icol));
        });

    std::vector<Real> o3l_vmr_out(pver * ncol, zero);
    mam4::validation::convert_2d_view_device_to_1d_vector(o3l_vmr, o3l_vmr_out);

    auto o3l_sfcsink_host = Kokkos::create_mirror_view(o3l_sfcsink);
    Kokkos::deep_copy(o3l_sfcsink_host, o3l_sfcsink);
    std::vector<Real> o3l_sfcsink_v(ncol, zero);
    for (int icol = 0; icol < ncol; ++icol) {
      o3l_sfcsink_v[icol] = o3l_sfcsink_host(icol);
    }

    output.set("o3l_vmr", o3l_vmr_out);
    output.set("o3l_sfcsink", o3l_sfcsink_v);
  });
}
