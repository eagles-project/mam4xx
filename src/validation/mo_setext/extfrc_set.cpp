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
using namespace mo_setext;
void extfrc_set(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    using View1DHost = typename HostType::view_1d<Real>;

    int forcings_frc_ndx[extfrc_cnt] = {};
    int forcings_nsectors[extfrc_cnt] = {};
    // std::vector<std::vector<Real>> data;

    bool forcings_file_alt_data[extfrc_cnt] = {};
    View1DHost forcings_fields_data_host;
    View1D forcings_fields_data[extfrc_cnt][4];
    for (int i = 1; i <= extfrc_cnt; ++i) {
      forcings_frc_ndx[i - 1] =
          int(input.get_array("forcings" + std::to_string(i) + "_frc_ndx")[0]);
      forcings_nsectors[i - 1] =
          int(input.get_array("forcings" + std::to_string(i) + "_nsectors")[0]);
      const int file_alt_data = int(input.get_array(
          "forcings" + std::to_string(i) + "_file_alt_data")[0]);
      forcings_file_alt_data[i - 1] = file_alt_data;
      for (int isec = 1; isec <= forcings_nsectors[i - 1]; ++isec) {

        auto label = "forcings" + std::to_string(i) + "_fields" +
                     std::to_string(isec) + "_data";
        const auto data1 = input.get_array(label);

        forcings_fields_data_host =
            View1DHost((Real *)data1.data(), data1.size());
        forcings_fields_data[i - 1][isec - 1] =
            View1D("forcings_fields_data", data1.size());

        Kokkos::deep_copy(forcings_fields_data[i - 1][isec - 1],
                          forcings_fields_data_host);
      }
    }

    View2D frcing("frcing", pver, extcnt);
    const int ncol = 1;
    auto team_policy = ThreadTeamPolicy(ncol, 1u);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
          extfrc_set(forcings_frc_ndx, forcings_nsectors,
                     forcings_file_alt_data, forcings_fields_data, frcing);
        });

    std::vector<Real> frcing_out(pver * extcnt, 0.0);
    mam4::validation::convert_2d_view_device_to_1d_vector(frcing, frcing_out);

    output.set("frcing", frcing_out);
  });
}