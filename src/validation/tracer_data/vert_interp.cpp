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

void vert_interp(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    // Retrieve input arrays from the ensemble
    const auto pin_db = input.get_array("pin");
    const auto pmid_db = input.get_array("pmid");
    const auto datain_db = input.get_array("datain");
    const auto levsiz = static_cast<int>(input.get_array("levsiz")[0]);
    const auto pver = static_cast<int>(input.get_array("pver")[0]);
    const auto ncol = static_cast<int>(input.get_array("ncol")[0]);

    // Define the Kokkos views based on the input data
    using View2D = typename DeviceType::view_2d<Real>;

    View2D pin("pin", ncol, levsiz);
    View2D pmid("pmid", ncol, pver);
    View2D datain("datain", ncol, levsiz);
    View2D dataout("dataout", ncol, pver); // Output array

    // Convert input data from std::vector or similar structure to Kokkos views
    mam4::validation::convert_1d_vector_to_2d_view_device(pin_db, pin);
    mam4::validation::convert_1d_vector_to_2d_view_device(pmid_db, pmid);
    mam4::validation::convert_1d_vector_to_2d_view_device(datain_db, datain);

    auto team_policy = ThreadTeamPolicy(ncol, Kokkos::AUTO);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
          // Perform the vertical interpolation
          const int icol     = team.league_rank();  // column index
          mam4::vertical_interpolation::vert_interp(
              icol, levsiz, pver, pin, pmid, datain, dataout
              );
        });

    // Convert the output data from Kokkos view to a format suitable for the
    // ensemble
    std::vector<Real> dataout_db(pver * ncol);
    mam4::validation::convert_2d_view_device_to_1d_vector(dataout, dataout_db);

    // Set the output data in the ensemble
    output.set("dataout", dataout_db);
  });
}
