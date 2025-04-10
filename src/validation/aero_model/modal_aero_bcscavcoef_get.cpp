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

void modal_aero_bcscavcoef_get(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    const Real zero = 0;
    using View2DHost = typename HostType::view_2d<Real>;
    using View1DHost = typename HostType::view_1d<Real>;
    using View1D = typename DeviceType::view_1d<Real>;
    using View2D = typename DeviceType::view_2d<Real>;

    const int ncol = int(input.get_array("ncol")[0]);

    auto scavimptblvol_vector = input.get_array("scavimptblvol");
    auto scavimptblnum_vector = input.get_array("scavimptblnum");
    View2DHost scavimptblvol_host("scavimptblvol_host",
                                  aero_model::nimptblgrow_total,
                                  AeroConfig::num_modes());
    View2DHost scavimptblnum_host("scavimptblnum_host",
                                  aero_model::nimptblgrow_total,
                                  AeroConfig::num_modes());

    // Note:  scavimptblvol_vector and scavimptblnum_vector were written in
    // row-major order.
    int count = 0;
    for (int imode = 0; imode < AeroConfig::num_modes(); ++imode) {
      for (int i = 0; i < aero_model::nimptblgrow_total; ++i) {
        scavimptblvol_host(i, imode) = scavimptblvol_vector[count];
        scavimptblnum_host(i, imode) = scavimptblnum_vector[count];
        count++;
      }
    }
    // FIXME: If I use values for scavimptblvol and scavimptblnum from
    // mam4::wetdep::init_scavimptbl,
    //  I will need to decrease the error for this test to 1e-5
#if 0
    View2DHost scavimptblvol_host("scavimptblvol_host", aero_model::nimptblgrow_total, AeroConfig::num_modes());
    View2DHost scavimptblnum_host("scavimptblnum_host", aero_model::nimptblgrow_total, AeroConfig::num_modes());
    mam4::wetdep::init_scavimptbl(scavimptblvol_host, scavimptblnum_host);
#endif

    View2D scavimptblnum("scavimptblnum", mam4::aero_model::nimptblgrow_total,
                         mam4::AeroConfig::num_modes());
    View2D scavimptblvol("scavimptblvol", mam4::aero_model::nimptblgrow_total,
                         mam4::AeroConfig::num_modes());

    Kokkos::deep_copy(scavimptblnum, scavimptblnum_host);
    Kokkos::deep_copy(scavimptblvol, scavimptblvol_host);

    auto dgn_awet_vector = input.get_array("dgn_awet");
    View2DHost dgn_awet_host("dgn_awet_host", ncol, AeroConfig::num_modes());
    View2D dgn_awet("dgn_awet", ncol, AeroConfig::num_modes());

    count = 0;
    for (int imode = 0; imode < AeroConfig::num_modes(); ++imode) {
      for (int icol = 0; icol < ncol; ++icol) {
        dgn_awet_host(icol, imode) = dgn_awet_vector[count];
        count++;
      }
    }
    Kokkos::deep_copy(dgn_awet, dgn_awet_host);

    auto dgnum_amode = input.get_array("dgnum_amode");
    const int imode = input.get_array("imode")[0] - 1;
    auto isprx_vector = input.get_array("isprx");
    View1D isprx("isprx", isprx_vector.size());
    auto isprx_host =
        View1DHost((Real *)isprx_vector.data(), isprx_vector.size());
    Kokkos::deep_copy(isprx, isprx_host);

    View1D scavcoefnum("scavcoefnum", ncol);
    View1D scavcoefvol("scavcoefvol", ncol);

    const Real dgnum_amode_imode = dgnum_amode[imode];

    haero::ThreadTeamPolicy team_policy(ncol, Kokkos::AUTO);

    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
          const int icol = team.league_rank();
          if (isprx(icol)) {
            aero_model::modal_aero_bcscavcoef_get(
                imode, dgn_awet(icol, imode), dgnum_amode_imode, scavimptblvol,
                scavimptblnum, scavcoefnum(icol), scavcoefvol(icol));
          }
        });

    std::vector<Real> scavcoefnum_host_v, scavcoefvol_host_v;
    scavcoefnum_host_v = std::vector(ncol, zero);
    scavcoefvol_host_v = std::vector(ncol, zero);

    auto scavcoefnum_host = View1DHost((Real *)scavcoefnum_host_v.data(), ncol);
    auto scavcoefvol_host = View1DHost((Real *)scavcoefvol_host_v.data(), ncol);

    Kokkos::deep_copy(scavcoefnum_host, scavcoefnum);
    Kokkos::deep_copy(scavcoefvol_host, scavcoefvol);

    output.set("scavcoefnum", scavcoefnum_host_v);
    output.set("scavcoefvol", scavcoefvol_host_v);
  });
}
