// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/mam4.hpp>

#include <iostream>
#include <skywalker.hpp>
#include <validation.hpp>

using namespace skywalker;
using namespace mam4;
using namespace haero;

void rebin(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    using View1D = typename DeviceType::view_1d<Real>;
    using View1DHost = typename HostType::view_1d<Real>;
    // Example data retrieval and setup
    auto src_x_db = input.get_array("src_x"); // Source bin positions
    auto trg_x_db = input.get_array("trg_x"); // Source data
    auto src_db = input.get_array("src");     // Source data

    const auto nsrc = static_cast<int>(input.get_array("nsrc")[0]);
    const auto ntrg = static_cast<int>(input.get_array("ntrg")[0]);

    View1D src_x("src_x", src_x_db.size());
    auto src_x_host = View1DHost((Real *)src_x_db.data(), src_x_db.size());
    Kokkos::deep_copy(src_x, src_x_host);

    View1D trg_x("trg_x", trg_x_db.size());
    auto trg_x_host = View1DHost((Real *)trg_x_db.data(), trg_x_db.size());
    Kokkos::deep_copy(trg_x, trg_x_host);

    View1D src("src", src_db.size());
    auto src_host = View1DHost((Real *)src_db.data(), src_db.size());
    Kokkos::deep_copy(src, src_host);

    View1D trg("trg", ntrg);

    auto team_policy = ThreadTeamPolicy(1, Kokkos::AUTO);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
          // Call rebin function
          vertical_interpolation::rebin(nsrc, ntrg, src_x, trg_x.data(), src,
                                        trg);
        });

    std::vector<Real> trg_out(ntrg);
    auto trg_host = View1DHost((Real *)trg_out.data(), ntrg);

    Kokkos::deep_copy(trg_host, trg);

    // Set the rebinned data in the output
    output.set("trg", trg_out);
  });
}
