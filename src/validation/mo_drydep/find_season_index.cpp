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
    using KTH = ekat::KokkosTypes<ekat::HostDevice>;
    using View2DIntHost = typename KTH::view_2d<int>;
    using View3DIntHost = typename KTH::view_3d<int>;
    using View1DHost = typename KTH::view_1d<Real>;

    auto clat_db = input.get_array("clat");
    auto lat_lai_db = input.get_array("lat_lai");
    auto wk_lai_db = input.get_array("wk_lai");

    const int plon = clat_db.size();
    View1DHost clat((Real *)clat_db.data(), plon);

    const int nlat_lai = lat_lai_db.size();
    View1DHost lat_lai((Real *)lat_lai_db.data(), nlat_lai);

    const int npft_lai = 11;

    View3DIntHost wk_lai("wk_lai", nlat_lai, npft_lai, 12);

    int count = 0;
    for (int d3 = 0; d3 < wk_lai.extent(2); ++d3) {
      for (int d2 = 0; d2 < wk_lai.extent(1); ++d2) {
        for (int d1 = 0; d1 < wk_lai.extent(0); ++d1) {
          // make sure that we are using int.
          wk_lai(d1, d2, d3) = static_cast<int>(wk_lai_db[count]);
          count++;
        }
      }
    }

    constexpr Real r2d = 180.0 / Constants::pi; // degrees to radians

    View2DIntHost index_season_lai("index_season_lai", plon, 12);

    auto policy = KTH::RangePolicy(0, plon);
    Kokkos::parallel_for(policy, [&](const int &j) {
      const auto index_season_lai_at_j = ekat::subview(index_season_lai, j);
      // convert to radians
      const Real clat_rads = clat(j) * r2d;
      mo_drydep::find_season_index(clat_rads, lat_lai, nlat_lai, wk_lai,
                                   index_season_lai_at_j);
      // c++ to Fortran; only for validation.
      for (int i = 0; i < 12; ++i) {
        index_season_lai_at_j(i) = index_season_lai_at_j(i) + 1;
      }
    });

    std::vector<Real> index_season_lai_out(plon * 12, 0.0);
    count = 0;
    for (int d2 = 0; d2 < 12; ++d2) {
      for (int d1 = 0; d1 < plon; ++d1) {
        // make sure that we are using int.
        index_season_lai_out[count] = index_season_lai(d1, d2);
        count++;
      }
    }

    output.set("index_season_lai", index_season_lai_out);
  });
}
