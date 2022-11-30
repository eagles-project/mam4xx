#include <mam4xx/mam4.hpp>

#include <mam4xx/aero_config.hpp>
#include <mam4xx/calcsize.hpp>
#include <skywalker.hpp>
#include <validation.hpp>

using namespace skywalker;
using namespace mam4;

void aitken_accum_exchange(Ensemble* ensemble) {

  // We don't need any settings for this particular test.
  // Settings settings = ensemble->settings();

  // Run the ensemble.
  ensemble->process([=](const Input& input, Output& output) {
    Real dt = input.get("dt");

    int nlev = 1;
    Real pblh = 1000;
    Atmosphere atm(nlev, pblh);
    mam4::Prognostics progs(nlev);
    mam4::Diagnostics diags(nlev);
    mam4::Tendencies tends(nlev);

    mam4::AeroConfig mam4_config;
    mam4::CalcSizeProcess process(mam4_config);
    const int nmodes = mam4_config.num_modes();
    const int nspec = mam4_config.num_aerosol_ids();

    auto q_i = input.get_array("interstitial");
    auto n_i = input.get_array("interstitial_num");
    auto q_c = input.get_array("cloud_borne");
    auto n_c = input.get_array("cloud_borne_num");


    DeviceType::view_1d<Real> d_v2nnom_nmodes("v2nnom_nmodes", nmodes);
    auto h_v2nnom_nmodes = Kokkos::create_mirror_view(d_v2nnom_nmodes);
    auto s_v2nnom_nmodes = input.get_array("v2nnom_nmodes");

    auto v2nmin_nmodes = input.get_array("v2nmin_nmodes");
    auto v2nmax_nmodes = input.get_array("v2nmax_nmodes");

    auto dgnnom_nmodes = input.get_array("dgnnom_nmodes");
    auto dgnmin_nmodes = input.get_array("dgnmin_nmodes");
    auto dgnmax_nmodes = input.get_array("dgnmax_nmodes");

    auto common_factor_nmodes = input.get_array("common_factor_nmodes");

    const Real num_i_aitsv = input.get("num_i_aitsv");
    const Real num_c_aitsv = input.get("num_c_aitsv");
    const Real num_i_accsv = input.get("num_i_accsv");
    const Real num_c_accsv = input.get("num_c_accsv");
    const Real num_i_k_aitsv = input.get("num_i_k_aitsv");
    const Real num_c_k_aitsv = input.get("num_c_k_aitsv");
    const Real num_i_k_accsv = input.get("num_i_k_accsv");
    const Real num_c_k_accsv = input.get("num_c_k_accsv");

    const Real dryvol_i_aitsv = input.get("dryvol_i_aitsv");
    const Real dryvol_c_aitsv = input.get("dryvol_c_aitsv");
    const Real dryvol_i_accsv = input.get("dryvol_i_accsv");
    const Real dryvol_c_accsv = input.get("dryvol_c_accsv");
    const Real drv_i_aitsv = input.get("drv_i_aitsv");
    const Real drv_c_aitsv = input.get("drv_c_aitsv");
    const Real drv_i_accsv = input.get("drv_i_accsv");
    const Real drv_c_accsv = input.get("drv_c_accsv");

    const Real dgncur_i_k = input.get("dgncur_i_k");
    const Real dgncur_c_k = input.get("dgncur_c_k");

    const Real v2ncur_c_k = input.get("v2ncur_c_k");
    const Real v2ncur_i_k = input.get("v2ncur_i_k");

    const Real drv_i_aitsv = input.get("drv_i_aitsv");
    const Real num_i_aitsv = input.get("num_i_aitsv");
    const Real drv_c_aitsv = input.get("drv_c_aitsv");
    const Real num_c_aitsv = input.get("num_c_aitsv");

    const Real drv_i_accsv = input.get("drv_i_accsv");
    const Real num_i_accsv = input.get("num_i_accsv");
    const Real drv_c_accsv = input.get("drv_c_accsv");
    const Real num_c_accsv = input.get("num_c_accsv");

    // // full array size = nmodes x nspec (4 x 7);
    DeviceType::view_2d<Real> d_inv_density("inv_density", nmodes, nspec);
    Kokkos::deep_copy(d_inv_density, 0.0);
    auto h_inv_density = Kokkos::create_mirror_view(d_inv_density);

    // const auto n_spec = num_species_mode(m);
    // for (int ispec = 0; ispec < n_spec; ispec++) {
    //   const int aero_id = int(mode_aero_species(m, ispec));
    //   h_inv_density[m][ispec] = Real(1.0) / aero_species(aero_id).density;
    // } // for(ispec)

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

        const int aero_id = int(mode_aero_species(imode, isp));
        h_inv_density(imode, isp) = Real(1.0) / aero_species(aero_id).density;

        count++;
      } // end species

      h_v2nnom_nmodes(imode) = s_v2nnom_nmodes[imode];

    } // end modes

    Kokkos::deep_copy(h_v2nnom_nmodes, d_v2nnom_nmodes);
    Kokkos::deep_copy(h_inv_density, d_inv_density);

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

          // calcsize::aitken_accum_exchange(
          //     k, aitken_idx, accum_idx, v2nmax_nmodes, v2nmin_nmodes,
          //     v2nnom_nmodes, dgnmax_nmodes, dgnmin_nmodes, dgnnom_nmodes,
          //     common_factor_nmodes, inv_density, adj_tscale_inv, dt,
          //     prognostics, dryvol_i_aitsv, num_i_k_aitsv, dryvol_c_aitsv,
          //     num_c_k_aitsv, dryvol_i_accsv, num_i_k_accsv, dryvol_c_accsv,
          //     num_c_k_accsv, dgncur_i_k, v2ncur_i_k, dgncur_c_k, v2ncur_c_k,
          //     diagnostics, tendencies);
          // calcsize::aitken_accum_exchange(
          //     k, aitken_idx, accumulation_idx,
          //     no_transfer_acc2ait, n_common_species_ait_accum,
          //     ait_spec_in_acc, acc_spec_in_ait,
          //     v2nmax_nmodes, v2nmin_nmodes,
          //     v2nnom_nmodes, dgnmax_nmodes, dgnmin_nmodes, dgnnom_nmodes,
          //     common_factor_nmodes, inv_density, adj_tscale_inv, dt,
          //     prognostics, dryvol_i_aitsv, num_i_k_aitsv, dryvol_c_aitsv,
          //     num_c_k_aitsv, dryvol_i_accsv, num_i_k_accsv, dryvol_c_accsv,
          //     num_c_k_accsv,
          //     // dgncur_i_k, v2ncur_i_k, dgncur_c_k, v2ncur_c_k,
          //     diagnostics, tendencies);
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
