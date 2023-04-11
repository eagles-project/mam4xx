// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/calcsize.hpp>
#include <skywalker.hpp>
#include <validation.hpp>
// #include <mam4xx/aero_config.hpp>

using namespace skywalker;
using namespace mam4;

void compute_dry_volume_k(Ensemble *ensemble) {

  // We don't need any settings for this particular test.
  // Settings settings = ensemble->settings();

  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    // Fetch ensemble parameters

    int nlev = 1;
    mam4::Prognostics progs = validation::create_prognostics(nlev);
    auto q_i = input.get_array("interstitial");
    auto q_c = input.get_array("cldbrn");
    // imode only has one elements
    // fortran to c++ idx
    const auto imode = int(input.get_array("imode")[0]) - 1;
    const auto n_spec = num_species_mode(imode);
    for (int isp = 0; isp < n_spec; ++isp) {
      // using indexing from mam4xx.
      const int isp_mam4xx = validation::e3sm_to_mam4xx_aerosol_idx[imode][isp];

      auto h_prog_aero_i =
          Kokkos::create_mirror_view(progs.q_aero_i[imode][isp_mam4xx]);
      h_prog_aero_i(0) = q_i[isp];
      Kokkos::deep_copy(progs.q_aero_i[imode][isp_mam4xx], h_prog_aero_i);

      auto h_prog_aero_c =
          Kokkos::create_mirror_view(progs.q_aero_c[imode][isp_mam4xx]);
      h_prog_aero_c(0) = q_c[isp];
      Kokkos::deep_copy(progs.q_aero_c[imode][isp_mam4xx], h_prog_aero_c);
    } // end species

    DeviceType::view_1d<Real> dryvol_i("Return dryvol_i", 1);
    DeviceType::view_1d<Real> dryvol_c("Return dryvol_c", 1);

    Kokkos::parallel_for(
        "compute_dry_volume_k", 1, KOKKOS_LAMBDA(int k) {
          Real inv_density[4][7];
          const auto n_spec = num_species_mode(imode);
          for (int ispec = 0; ispec < n_spec; ispec++) {
            const int aero_id = int(mode_aero_species(imode, ispec));
            inv_density[imode][ispec] =
                Real(1.0) / aero_species(aero_id).density;
          } // for(ispec)

          Real dryvol_i_k = 0;
          Real dryvol_c_k = 0;
          calcsize::compute_dry_volume_k(k, imode, inv_density,
                                         progs,      // in
                                         dryvol_i_k, // out
                                         dryvol_c_k);
          dryvol_i(0) = dryvol_i_k;
          dryvol_c(0) = dryvol_c_k;
        });

    auto host_dryvol_i = Kokkos::create_mirror_view(dryvol_i);
    Kokkos::deep_copy(host_dryvol_i, dryvol_i);

    auto host_dryvol_c = Kokkos::create_mirror_view(dryvol_c);
    Kokkos::deep_copy(host_dryvol_c, dryvol_c);

    std::vector<Real> values_i, values_c;
    values_i.push_back(host_dryvol_i(0));
    values_c.push_back(host_dryvol_c(0));

    output.set("dryvol_a", values_i);
    output.set("dryvol_c", values_c);
  });
}
