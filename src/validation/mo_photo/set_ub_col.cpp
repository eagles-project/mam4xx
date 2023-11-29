// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/mam4.hpp>

#include <skywalker.hpp>
#include <validation.hpp>

using namespace skywalker;
using namespace mam4;
using namespace haero;
void set_ub_col(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    using View2D = typename DeviceType::view_2d<Real>;
    using View3D = typename DeviceType::view_3d<Real>;

    constexpr int ncol = 4;
    constexpr int pver = mam4::nlev;
    const auto vmr_in = input.get_array("vmr");
    const auto invariants_in = input.get_array("invariants");
    const auto pdel_in = input.get_array("pdel");

    View3D vmr("vmr", ncol, pver, mam4::gas_chemistry::gas_pcnst);
    View3D invariants("invariants", ncol, pver, mam4::gas_chemistry::nfs);
    View2D pdel("pdel", ncol, pver);

    mam4::validation::convert_1d_vector_to_3d_view_device(vmr_in, vmr);
    mam4::validation::convert_1d_vector_to_3d_view_device(invariants_in, invariants);
    mam4::validation::convert_1d_vector_to_2d_view_device(pdel_in, pdel);

    View2D col_delta("col_delta", ncol, pver+1);
    auto team_policy = ThreadTeamPolicy(ncol, 1u);
    Kokkos::parallel_for(
      team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
        constexpr int gas_pcnst = mam4::gas_chemistry::gas_pcnst;
        constexpr int nfs = mam4::gas_chemistry::nfs;
        const int icol = team.league_rank();
        auto vmr_icol = Kokkos::subview(vmr, icol, Kokkos::ALL(), Kokkos::ALL());
        auto invariants_icol = Kokkos::subview(invariants, icol, Kokkos::ALL(), Kokkos::ALL());
        auto pdel_icol = Kokkos::subview(pdel, icol, Kokkos::ALL());
        auto col_delta_icol = Kokkos::subview(col_delta, icol, Kokkos::ALL());
        col_delta_icol(0) = 0.0;
        Kokkos::parallel_for(pver, KOKKOS_LAMBDA(const int k) {
          Real vmr_ik[gas_pcnst];
          for (int l = 0; l < gas_pcnst; ++l) {
            vmr_ik[l] = vmr_icol(k, l);
          }
          Real inv_ik[nfs];
          for (int l = 0; l < nfs; ++l) {
            inv_ik[l] = invariants_icol(k, l);
          }
          mam4::mo_photo::set_ub_col(col_delta_icol(k+1), vmr_ik, inv_ik, pdel_icol(k));
        });
    });

    constexpr Real zero = 0;
    std::vector<Real> col_delta_out(ncol * (pver+1), zero);
    mam4::validation::convert_2d_view_device_to_1d_vector(col_delta, col_delta_out);
    output.set("col_delta", col_delta_out);
  });
}
