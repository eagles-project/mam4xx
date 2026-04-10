// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/mam4.hpp>
#include <validation.hpp>

using namespace skywalker;
using namespace mam4::mo_photo;

void calc_sum_wght(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    using View1DHost = typename mam4::HostType::view_1d<Real>;
    using View1D = typename mam4::DeviceType::view_1d<Real>;

    const auto dels_db = input.get_array("dels");
    const Real wrk0 = input.get_array("wrk0")[0];
    const int iz = int(input.get_array("iz")[0]) - 1;
    const int is = int(input.get_array("is_")[0]) - 1;
    const int iv = int(input.get_array("iv")[0]) - 1;
    const int ial = int(input.get_array("ial")[0]) - 1;

    auto shape_rsf_tab = input.get_array("shape_rsf_tab");
    auto synthetic_values = input.get_array("synthetic_values_rsf_tab");

    const int nw = int(shape_rsf_tab[0]);
    const int nump = int(shape_rsf_tab[1]);
    const int numsza = int(shape_rsf_tab[2]);
    const int numcolo3 = int(shape_rsf_tab[3]);
    const int numalb = int(shape_rsf_tab[4]);

    View5D rsf_tab;

    mam4::validation::create_synthetic_rsf_tab(
        rsf_tab, nw, nump, numsza, numcolo3, numalb, synthetic_values.data());

    const auto dels = View1D("dels", 3);
    auto dels_host = View1DHost((Real *)dels_db.data(), 3);
    Kokkos::deep_copy(dels, dels_host);

    const auto psum = View2D("psum", 1, nw);

    auto team_policy = mam4::ThreadTeamPolicy(1u, Kokkos::AUTO);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const mam4::ThreadTeam &team) {
          calc_sum_wght(dels.data(), wrk0, // in
                        iz, is, iv, ial,   // in
                        rsf_tab,           // in
                        nw,                //
                        psum, 0);          // psum(row=0,:)
        });
    const Real zero = 0;
    std::vector<Real> psum_db(nw, zero);
    auto psum_host = Kokkos::create_mirror_view(psum);
    Kokkos::deep_copy(psum_host, psum);
    for (int w = 0; w < nw; ++w)
      psum_db[w] = psum_host(0, w);

    output.set("psum", psum_db);
  });
}
