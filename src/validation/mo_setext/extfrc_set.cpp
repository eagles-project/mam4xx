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
    // using View2DHost = typename HostType::view_2d<Real>;


    forcing forcings[extfrc_cnt]; 

    // int forcings_frc_ndx[extfrc_cnt] = {};
    // int forcings_nsectors[extfrc_cnt] = {};
    // std::vector<std::vector<Real>> data;

    // bool forcings_file_alt_data[extfrc_cnt] = {};
    // View1DHost forcings_fields_data_host;
    // View1D forcings_fields_data[extfrc_cnt][4];
    for (int i = 1; i <= extfrc_cnt; ++i) {

      forcing forcing_mm; 

      forcing_mm.frc_ndx = int(input.get_array("forcings" + std::to_string(i) + "_frc_ndx")[0]);
      forcing_mm.nsectors  = int(input.get_array("forcings" + std::to_string(i) + "_nsectors")[0]);
      forcing_mm.file_alt_data = int(input.get_array("forcings" + std::to_string(i) + "_file_alt_data")[0]);

      forcing_mm.fields_data = View2D("data", forcing_mm.nsectors, pver);

      for (int isec = 1; isec <= forcing_mm.nsectors; ++isec) {

        auto label = "forcings" + std::to_string(i) + "_fields" +
                     std::to_string(isec) + "_data";
        const auto data1 = input.get_array(label);

        View1DHost forcings_fields_data_host =
            View1DHost((Real *)data1.data(), data1.size());   

        const View1D data_isec = Kokkos::subview(forcing_mm.fields_data , isec-1, Kokkos::ALL());    
        Kokkos::deep_copy(data_isec, forcings_fields_data_host);
      } // isec

      forcings[i-1] = forcing_mm;

    }



    View2D frcing("frcing", pver, extcnt);
    const int ncol = 1;
    auto team_policy = ThreadTeamPolicy(ncol, 1u);


    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
          // extfrc_set(forcings_frc_ndx, forcings_nsectors,
          //            forcings_file_alt_data, forcings_fields_data, frcing);

            extfrc_set2(forcings, frcing);
        });
    std::vector<Real> frcing_out(pver * extcnt, 0.0);
    mam4::validation::convert_2d_view_device_to_1d_vector(frcing, frcing_out);

    output.set("frcing", frcing_out);
  });
}