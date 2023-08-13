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
using namespace mo_photo;
void calc_sum_wght(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    using View1DHost = typename HostType::view_1d<Real>;
    using View1D = typename DeviceType::view_1d<Real>;

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

    const auto psum = View1D("psum", nw);

    auto team_policy = ThreadTeamPolicy(1u, 1u);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
          calc_sum_wght(dels.data(), wrk0, // in
                        iz, is, iv, ial,   // in
                        rsf_tab,           // in
                        nw,                //
                        psum.data());
        });
    const Real zero = 0;
    std::vector<Real> psum_db(nw, zero);
    auto psum_host = View1DHost((Real *)psum_db.data(), nw);
    Kokkos::deep_copy(psum_host, psum);

    output.set("psum", psum_db);
  });
}
