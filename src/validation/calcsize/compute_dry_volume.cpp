#include <haero/constants.hpp>
#include <iostream>
#include <mam4xx/calcsize.hpp>
#include <skywalker.hpp>
#include <validation.hpp>

using namespace skywalker;
using namespace mam4;

void compute_dry_volume_k(Ensemble* ensemble) {

  // We don't need any settings for this particular test.
  // Settings settings = ensemble->settings();

  // Run the ensemble.
  ensemble->process([=](const Input& input, Output& output) {
    // Fetch ensemble parameters

    // Pack qi = input.get("mf_interstitial_aerosols_mode_0");
    // Pack qc = input.get("mf_cloudborne_aerosols_mode_0");

    Prognostics progs; 
    Pack dryvol_i[4];
    Pack dryvol_c[4];

    // Call the cluster growth function on device.
    Kokkos::parallel_for(
        "compute_dry_volume_k", 1, [&] KOKKOS_FUNCTION(int k) {
          for (int imode = 0; imode < 4; ++imode)
          {
            Pack dryvol_i_k = 0;
            Pack dryvol_c_k = 0;
            calcsize::compute_dry_volume_k(k, imode,
            progs, // in
            dryvol_i_k, // out
            dryvol_c_k); 
            dryvol_i[imode] = dryvol_i_k;
            dryvol_c[imode] = dryvol_c_k;
          }

        });

    output.set("dryvol_i", dryvol_i[0][0]);
    output.set("dryvol_c", dryvol_c[0][0]);
    
  });
}
