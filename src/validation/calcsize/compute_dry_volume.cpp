
#include <haero/constants.hpp>
#include <iostream>
#include <mam4xx/calcsize.hpp>
#include <mam4xx/mam4.hpp>
#include <skywalker.hpp>
#include <validation.hpp>

using namespace skywalker;
using namespace mam4;

void compute_dry_volume_k(Ensemble *ensemble) {

  // We don't need any settings for this particular test.
  // Settings settings = ensemble->settings();

  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    // Fetch ensemble parameters

    int nlev = 1;
    mam4::Prognostics progs(nlev);
    auto q_i = input.get_array("interstitial");
    auto q_c = input.get_array("cloud_borne");

    const auto nmodes = AeroConfig::num_modes();
    // FIXME: only works for one cell
    int count = 0;
    for (int imode = 0; imode < nmodes; ++imode) {
      const auto n_spec = num_species_mode[imode];
      for (int isp = 0; isp < n_spec; ++isp) {
        // const auto prog_aero_i = ekat::scalarize(progs.q_aero_i[imode][i]);
        auto h_prog_aero_i =
            Kokkos::create_mirror_view(progs.q_aero_i[imode][isp]);
        h_prog_aero_i(0) = q_i[count];
        Kokkos::deep_copy(h_prog_aero_i, progs.q_aero_i[imode][isp]);

        auto h_prog_aero_c =
            Kokkos::create_mirror_view(progs.q_aero_c[imode][isp]);
        h_prog_aero_c(0) = q_c[count];
        Kokkos::deep_copy(h_prog_aero_c, progs.q_aero_c[imode][isp]);

        count++;
      } // end species
    }   // end modes

    Pack dryvol_i[4];
    Pack dryvol_c[4];

    // Call the cluster growth function on device.
    // FIXME: Will compile in CUDA?
    Kokkos::parallel_for("compute_dry_volume_k", 1, [&] KOKKOS_FUNCTION(int k) {
      for (int imode = 0; imode < nmodes; ++imode) {
        Pack dryvol_i_k = 0;
        Pack dryvol_c_k = 0;
        calcsize::compute_dry_volume_k(k, imode,
                                       progs,      // in
                                       dryvol_i_k, // out
                                       dryvol_c_k);
        dryvol_i[imode] = dryvol_i_k;
        dryvol_c[imode] = dryvol_c_k;
      }
    });

    std::vector<Real> values_i, values_c;

    for (int imode = 0; imode < nmodes; ++imode) {
      values_i.push_back(dryvol_i[imode][0]);
      values_c.push_back(dryvol_c[imode][0]);
    }

    output.set("dryvol_i", values_i);
    output.set("dryvol_c", values_c);
  });
}
