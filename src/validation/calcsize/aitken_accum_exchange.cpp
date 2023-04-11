// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

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
    Atmosphere atm = validation::create_atmosphere(nlev, pblh);
    mam4::Prognostics progs = validation::create_prognostics(nlev);
    mam4::Diagnostics diags = validation::create_diagnostics(nlev);
    mam4::Tendencies tends = validation::create_tendencies(nlev);

    mam4::AeroConfig mam4_config;
    mam4::CalcSizeProcess process(mam4_config);
    const int nmodes = mam4_config.num_modes();
    const int nspec = mam4_config.num_aerosol_ids();

    const bool no_transfer_acc2ait[7] = {true,  false, true, false,
                                         false, true,  true};
    const int n_common_species_ait_accum = 4;
    const int ait_spec_in_acc[4] = {0, 1, 2, 3};
    const int acc_spec_in_ait[4] = {0, 2, 5, 6};

    const int max_k = input.get("max_k");

    auto q_i = input.get_array("interstitial");
    auto n_i = input.get_array("interstitial_num");
    auto q_c = input.get_array("cloud_borne");
    auto n_c = input.get_array("cloud_borne_num");

    auto in_v2nmin_nmodes = input.get_array("v2nmin_nmodes");
    Real v2nmin_nmodes[nmodes];
    auto in_v2nmax_nmodes = input.get_array("v2nmax_nmodes");
    Real v2nmax_nmodes[nmodes];
    auto in_v2nnom_nmodes = input.get_array("v2nnom_nmodes");
    Real v2nnom_nmodes[nmodes];

    auto in_dgnmin_nmodes = input.get_array("dgnmin_nmodes");
    Real dgnmin_nmodes[nmodes];
    auto in_dgnmax_nmodes = input.get_array("dgnmax_nmodes");
    Real dgnmax_nmodes[nmodes];
    auto in_dgnnom_nmodes = input.get_array("dgnnom_nmodes");
    Real dgnnom_nmodes[nmodes];

    auto in_common_factor_nmodes = input.get_array("common_factor_nmodes");
    Real common_factor_nmodes[nmodes];

    const Real num_i_k_aitsv = input.get("num_i_k_aitsv");
    const Real num_c_k_aitsv = input.get("num_c_k_aitsv");
    const Real num_i_k_accsv = input.get("num_i_k_accsv");
    const Real num_c_k_accsv = input.get("num_c_k_accsv");

    const Real dryvol_i_aitsv = input.get("dryvol_i_aitsv");
    const Real dryvol_c_aitsv = input.get("dryvol_c_aitsv");
    const Real dryvol_i_accsv = input.get("dryvol_i_accsv");
    const Real dryvol_c_accsv = input.get("dryvol_c_accsv");

    Real inv_density[nmodes][nspec];

    int count = 0;
    for (int imode = 0; imode < nmodes; ++imode) {

      auto h_prog_n_mode_i = Kokkos::create_mirror_view(progs.n_mode_i[imode]);
      h_prog_n_mode_i(0) = n_i[imode];
      Kokkos::deep_copy(progs.n_mode_i[imode], h_prog_n_mode_i);

      auto h_prog_n_mode_c = Kokkos::create_mirror_view(progs.n_mode_c[imode]);
      h_prog_n_mode_c(0) = n_c[imode];
      Kokkos::deep_copy(progs.n_mode_c[imode], h_prog_n_mode_c);

      const auto n_spec = num_species_mode(imode);
      for (int isp = 0; isp < n_spec; ++isp) {
        auto h_prog_aero_i =
            Kokkos::create_mirror_view(progs.q_aero_i[imode][isp]);
        h_prog_aero_i(0) = q_i[count];
        Kokkos::deep_copy(progs.q_aero_i[imode][isp], h_prog_aero_i);

        auto h_prog_aero_c =
            Kokkos::create_mirror_view(progs.q_aero_c[imode][isp]);
        h_prog_aero_c(0) = q_c[count];
        Kokkos::deep_copy(progs.q_aero_c[imode][isp], h_prog_aero_c);

        const int aero_id = int(mode_aero_species(imode, isp));
        inv_density[imode][isp] = Real(1.0) / aero_species(aero_id).density;

        count++;
      } // end species

      v2nmin_nmodes[imode] = in_v2nmin_nmodes[imode];
      v2nmax_nmodes[imode] = in_v2nmax_nmodes[imode];
      v2nnom_nmodes[imode] = in_v2nnom_nmodes[imode];

      dgnmin_nmodes[imode] = in_dgnmin_nmodes[imode];
      dgnmax_nmodes[imode] = in_dgnmax_nmodes[imode];
      dgnnom_nmodes[imode] = in_dgnnom_nmodes[imode];

      common_factor_nmodes[imode] = in_common_factor_nmodes[imode];

    } // end modes

    const int aitken_idx = int(ModeIndex::Aitken);
    const int accum_idx = int(ModeIndex::Accumulation);
    static constexpr Real close_to_one = 1.0 + 1.0e-15;
    static constexpr Real seconds_in_a_day = 86400.0;
    const auto adj_tscale = haero::max(seconds_in_a_day, dt);
    const auto adj_tscale_inv = 1.0 / (adj_tscale * close_to_one);

    Kokkos::parallel_for(
        "aitken_accum_exchange_k", max_k, KOKKOS_LAMBDA(const int &k) {
          calcsize::aitken_accum_exchange(
              k, aitken_idx, accum_idx, no_transfer_acc2ait,
              n_common_species_ait_accum, ait_spec_in_acc, acc_spec_in_ait,
              v2nmax_nmodes, v2nmin_nmodes, v2nnom_nmodes, dgnmax_nmodes,
              dgnmin_nmodes, dgnnom_nmodes, common_factor_nmodes, inv_density,
              adj_tscale_inv, dt, progs, dryvol_i_aitsv, num_i_k_aitsv,
              dryvol_c_aitsv, num_c_k_aitsv, dryvol_i_accsv, num_i_k_accsv,
              dryvol_c_accsv, num_c_k_accsv, diags, tends);
        });

    std::vector<Real> tend_aero_i_out;
    std::vector<Real> tend_n_mode_i_out;

    std::vector<Real> tend_aero_c_out;
    std::vector<Real> tend_n_mode_c_out;

    for (int imode = 0; imode < nmodes; ++imode) {

      auto h_tend_num_i = Kokkos::create_mirror_view(tends.n_mode_i[imode]);
      Kokkos::deep_copy(h_tend_num_i, tends.n_mode_i[imode]);
      tend_n_mode_i_out.push_back(h_tend_num_i(0));

      auto h_tend_num_c = Kokkos::create_mirror_view(tends.n_mode_c[imode]);
      Kokkos::deep_copy(h_tend_num_c, tends.n_mode_c[imode]);
      tend_n_mode_c_out.push_back(h_tend_num_c(0));

      const auto n_spec = num_species_mode(imode);
      for (int isp = 0; isp < n_spec; ++isp) {
        auto h_tend_aero_i =
            Kokkos::create_mirror_view(tends.q_aero_i[imode][isp]);
        Kokkos::deep_copy(h_tend_aero_i, tends.q_aero_i[imode][isp]);
        tend_aero_i_out.push_back(h_tend_aero_i(0));

        auto h_tend_aero_c =
            Kokkos::create_mirror_view(tends.q_aero_c[imode][isp]);
        Kokkos::deep_copy(h_tend_aero_c, tends.q_aero_c[imode][isp]);
        tend_aero_c_out.push_back(h_tend_aero_c(0));

      } // end species
    }   // end mode

    output.set("interstitial_ptend", tend_aero_i_out);
    output.set("interstitial_ptend_num", tend_n_mode_i_out);

    output.set("cloud_borne_ptend_num", tend_n_mode_c_out);
    output.set("cloud_borne_ptend", tend_aero_c_out);
  });
}
