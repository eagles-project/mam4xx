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
void find_index(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    // validation test from standalone mo_photo.
    using View1DHost = typename HostType::view_1d<Real>;
    using View1D = typename DeviceType::view_1d<Real>;
    using ViewInt1D = typename DeviceType::view_1d<int>;

    const auto var_in_db = input.get_array("var_in");
    const auto var_len = input.get_array("var_len")[0];
    const auto var_min = input.get_array("var_min")[0];

    View1DHost var_in_host((Real *)var_in_db.data(), var_in_db.size());
    View1D var_in("var_in", var_in_db.size());
    Kokkos::deep_copy(var_in, var_in_host);

    ViewInt1D idx_out("idx_out", 1);
    auto team_policy = ThreadTeamPolicy(1u, Kokkos::AUTO);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
          find_index(var_in, var_len,
                     var_min, //  in
                     idx_out(0));
        });
    // C++ indexing to fortran indexing
    auto idx_out_host = Kokkos::create_mirror_view(idx_out);
    Kokkos::deep_copy(idx_out_host, idx_out);
    output.set("idx_out", idx_out_host(0) + 1);
  });
}
