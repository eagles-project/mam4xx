#include <mam4xx/mam4.hpp>

#include <mam4xx/aero_config.hpp>
#include <mam4xx/calcsize.hpp>
#include <skywalker.hpp>
#include <validation.hpp>

using namespace skywalker;
using namespace mam4;

void aitken_accum_exchange(Ensemble *ensemble) {

  // We don't need any settings for this particular test.
  // Settings settings = ensemble->settings();

  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    Real dt = input.get("dt");

    int nlev = 1;
    Real pblh = 1000;
    Atmosphere atm(nlev, pblh);
    mam4::Prognostics progs(nlev);
    mam4::Diagnostics diags(nlev);
    mam4::Tendencies tends(nlev);

    mam4::AeroConfig mam4_config;
    mam4::CalcSizeProcess process(mam4_config);
    const auto nmodes = mam4_config.num_modes();

    auto q_i = input.get_array("interstitial");
    auto n_i = input.get_array("interstitial_num");
    auto q_c = input.get_array("cloud_borne");
    auto n_c = input.get_array("cloud_borne_num");

    auto s_v2nnom_nmodes = input.get_array("v2nnom_nmodes");

    DeviceType::view_1d<Real> d_v2nnom_nmodes("v2nnom_nmodes", nmodes);
    auto h_v2nnom_nmodes = Kokkos::create_mirror_view(d_v2nnom_nmodes);
    // const Real v2nnom_nmodes = input.get_array("v2nnom_nmodes");

    const Real drv_i_aitsv = input.get("drv_i_aitsv");
    const Real num_i_aitsv = input.get("num_i_aitsv");
    const Real drv_c_aitsv = input.get("drv_c_aitsv");
    const Real num_c_aitsv = input.get("num_c_aitsv");

    const Real drv_i_accsv = input.get("drv_i_accsv");
    const Real num_i_accsv = input.get("num_i_accsv");
    const Real drv_c_accsv = input.get("drv_c_accsv");
    const Real num_c_accsv = input.get("num_c_accsv");

    int count = 0;
    for (int imode = 0; imode < nmodes; ++imode) {

      auto h_prog_n_mode_i = Kokkos::create_mirror_view(progs.n_mode_i[imode]);
      h_prog_n_mode_i(0) = n_i[imode];
      Kokkos::deep_copy(h_prog_n_mode_i, progs.n_mode_i[imode]);

      auto h_prog_n_mode_c = Kokkos::create_mirror_view(progs.n_mode_c[imode]);
      h_prog_n_mode_c(0) = n_c[imode];
      Kokkos::deep_copy(h_prog_n_mode_c, progs.n_mode_c[imode]);

      const auto n_spec = num_species_mode(imode);
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

      h_v2nnom_nmodes(imode) = s_v2nnom_nmodes[imode];

    } // end modes

    Kokkos::deep_copy(h_v2nnom_nmodes, d_v2nnom_nmodes);
    const int aitken_idx = int(ModeIndex::Aitken);
    const int accum_idx = int(ModeIndex::Accumulation);
    static constexpr Real close_to_one = 1.0 + 1.0e-15;
    static constexpr Real seconds_in_a_day = 86400.0;
    const auto adj_tscale = haero::max(seconds_in_a_day, dt);
    const auto adj_tscale_inv = 1.0 / (adj_tscale * close_to_one);

    Kokkos::parallel_for(
        "compute_dry_volume_k", 1, KOKKOS_LAMBDA(int k) {
          Real v2nnom_nmodes[nmodes];
          for (int m = 0; m < nmodes; ++m) {
            v2nnom_nmodes[m] = d_v2nnom_nmodes(m);
          }

          calcsize::aitken_accum_exchange(
              k, aitken_idx, accum_idx, v2nnom_nmodes, adj_tscale_inv, dt,
              progs, drv_i_aitsv, num_i_aitsv, drv_c_aitsv, num_c_aitsv,
              drv_i_accsv, num_i_accsv, drv_c_accsv, num_c_accsv, diags, tends);
        });

    std::vector<Real> tend_aero_i_out;
    std::vector<Real> tend_n_mode_i_out;

    std::vector<Real> tend_aero_c_out;
    std::vector<Real> tend_n_mode_c_out;

    for (int imode = 0; imode < nmodes; ++imode) {

      auto h_tend_num_i = Kokkos::create_mirror_view(tends.n_mode_i[imode]);
      Kokkos::deep_copy(tends.n_mode_i[imode], h_tend_num_i);
      tend_n_mode_i_out.push_back(h_tend_num_i(0));

      auto h_tend_num_c = Kokkos::create_mirror_view(tends.n_mode_c[imode]);
      Kokkos::deep_copy(tends.n_mode_c[imode], h_tend_num_c);
      tend_n_mode_c_out.push_back(h_tend_num_c(0));

      const auto n_spec = num_species_mode(imode);
      for (int isp = 0; isp < n_spec; ++isp) {
        auto h_tend_aero_i =
            Kokkos::create_mirror_view(tends.q_aero_i[imode][isp]);
        Kokkos::deep_copy(tends.q_aero_i[imode][isp], h_tend_aero_i);
        tend_aero_i_out.push_back(h_tend_aero_i(0));

        auto h_tend_aero_c =
            Kokkos::create_mirror_view(tends.q_aero_c[imode][isp]);
        Kokkos::deep_copy(tends.q_aero_c[imode][isp], h_tend_aero_c);
        tend_aero_c_out.push_back(h_tend_aero_c(0));

      } // end species
    }   // end mode

    output.set("interstitial_ptend", tend_aero_i_out);
    output.set("interstitial_ptend_num", tend_n_mode_i_out);

    output.set("cloud_borne_ptend_num", tend_n_mode_c_out);
    output.set("cloud_borne_ptend", tend_aero_c_out);
  });
}
