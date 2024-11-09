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

void find_season_index(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    // Assuming you have a way to get these from your input
    using View2DInt = typename DeviceType::view_2d<int>;
    using View3DInt = typename DeviceType::view_3d<int>;
    using View1DHost = typename HostType::view_1d<Real>;
    using View1D = typename DeviceType::view_1d<Real>;

    auto clat_db = input.get_array("clat");
    auto lat_lai_db = input.get_array("lat_lai");
    auto wk_lai_db = input.get_array("wk_lai");

    const int plon = clat_db.size();
    View1D clat("clat", plon);
    auto clat_host = View1DHost((Real *)clat_db.data(),plon );
    Kokkos::deep_copy(clat, clat_host);

    const int nlat_lai = lat_lai_db.size();
    View1D lat_lai("lat_lai", nlat_lai);
    auto lat_lai_host = View1DHost((Real *)lat_lai_db.data(),nlat_lai );
    Kokkos::deep_copy(lat_lai, lat_lai_host);

    const int npft_lai = 11;

    View3DInt wk_lai("wk_lai",nlat_lai, npft_lai, 12 );
    validation::convert_1d_vector_to_3d_view_int_device(wk_lai_db, wk_lai);

    View2DInt index_season_lai("index_season_lai", plon, 12);
    auto team_policy = ThreadTeamPolicy(plon, Kokkos::AUTO);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
      const int j = team.league_rank();
      const auto index_season_lai_at_j = ekat::subview(index_season_lai,j);
      mo_drydep::find_season_index(clat(j), lat_lai, nlat_lai, wk_lai, index_season_lai_at_j);

    team.team_barrier();
    // c++ to Fortran; only for validation.
    Kokkos::parallel_for(
      Kokkos::TeamVectorRange(team, index_season_lai_at_j.extent(0)), [&](int i) {
              index_season_lai_at_j(i)=index_season_lai_at_j(i)+1;
    });


    });

    std::vector<Real> index_season_lai_out(plon*12, 0.0);
    mam4::validation::convert_2d_view_int_device_to_1d_vector(index_season_lai,
                                                          index_season_lai_out);
    output.set("index_season_lai", index_season_lai_out);
  });
}
