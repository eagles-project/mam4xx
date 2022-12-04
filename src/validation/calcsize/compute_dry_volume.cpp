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
    mam4::Prognostics progs(nlev);
    auto q_i = input.get_array("interstitial");
    auto q_c = input.get_array("cloud_borne");

    const auto nmodes = AeroConfig::num_modes();
    // FIXME: only works for one cell
    int count = 0;
    for (int imode = 0; imode < nmodes; ++imode) {
      const auto n_spec = num_species_mode(imode);
      for (int isp = 0; isp < n_spec; ++isp) {
        // const auto prog_aero_i = ekat::scalarize(progs.q_aero_i[imode][i]);
        auto h_prog_aero_i =
            Kokkos::create_mirror_view(progs.q_aero_i[imode][isp]);
        h_prog_aero_i(0) = q_i[count];
        Kokkos::deep_copy(progs.q_aero_i[imode][isp], h_prog_aero_i);

        auto h_prog_aero_c =
            Kokkos::create_mirror_view(progs.q_aero_c[imode][isp]);
        h_prog_aero_c(0) = q_c[count];
        Kokkos::deep_copy(progs.q_aero_c[imode][isp], h_prog_aero_c);

        count++;
      } // end species
    }   // end modes

    DeviceType::view_1d<Real> dryvol_i("Return dryvol_i", 4);
    DeviceType::view_1d<Real> dryvol_c("Return dryvol_c", 4);

    // FIXMED: need to update this variable

    // Call the cluster growth function on device.
    // FIXME: Will compile in CUDA?
    Kokkos::parallel_for(
        "compute_dry_volume_k", 1, KOKKOS_LAMBDA(int k) {
          Real inv_density[4][7];
          for (int imode = 0; imode < nmodes; ++imode) {
            const auto n_spec = num_species_mode(imode);
            for (int ispec = 0; ispec < n_spec; ispec++) {
              const int aero_id = int(mode_aero_species(imode, ispec));
              inv_density[imode][ispec] =
                  Real(1.0) / aero_species(aero_id).density;
            } // for(ispec)
          }
          for (int imode = 0; imode < nmodes; ++imode) {
            Real dryvol_i_k = 0;
            Real dryvol_c_k = 0;
            calcsize::compute_dry_volume_k(k, imode, inv_density,
                                           progs,      // in
                                           dryvol_i_k, // out
                                           dryvol_c_k);
            dryvol_i(imode) = dryvol_i_k;
            dryvol_c(imode) = dryvol_c_k;
          }
        });

    auto host_dryvol_i = Kokkos::create_mirror_view(dryvol_i);
    Kokkos::deep_copy(host_dryvol_i, dryvol_i);

    auto host_dryvol_c = Kokkos::create_mirror_view(dryvol_c);
    Kokkos::deep_copy(host_dryvol_c, dryvol_c);

    std::vector<Real> values_i, values_c;
    // FIXME need to copy from device to host
    for (int imode = 0; imode < nmodes; ++imode) {
      values_i.push_back(host_dryvol_i(imode));
      values_c.push_back(host_dryvol_c(imode));
    }

    output.set("dryvol_i", values_i);
    output.set("dryvol_c", values_c);
  });
}
