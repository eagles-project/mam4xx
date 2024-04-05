// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/mam4.hpp>

#include <mam4xx/calcsize.hpp>
#include <skywalker.hpp>
#include <validation.hpp>

using namespace skywalker;
using namespace mam4;
using namespace haero;

void modal_aero_calcsize_sub_ptend(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    constexpr int pcnst = aero_model::pcnst;
    constexpr int pver = ndrop::pver;
    constexpr int ntot_amode = AeroConfig::num_modes();
    constexpr int nspec_max = ndrop::nspec_max;
    constexpr int maxd_aspectype = ndrop::maxd_aspectype;

    using View2D = DeviceType::view_2d<Real>;

    auto state_q_db = input.get_array("state_q");
    auto qqcw_db = input.get_array("qqcw");
    const auto dt = input.get_array("dt")[0];

    View2D state_q("state_q", pver, pcnst);
    mam4::validation::convert_1d_vector_to_2d_view_device(state_q_db, state_q);
    View2D qqcw("qqcw", pver, pcnst);
    auto qqcw_host = create_mirror_view(qqcw);

    int count = 0;
    for (int kk = 0; kk < pver; ++kk) {
      for (int i = 0; i < pcnst; ++i) {
        qqcw_host(kk, i) = qqcw_db[count];
        count++;
      }
    }
    Kokkos::deep_copy(qqcw, qqcw_host);
    View2D dgnumdry_m("dgnumdry_m", pver, ntot_amode);

    View2D ptend_q("ptend_q", pver, pcnst);
    View2D dqqcwdt("dqqcwdt", pver, pcnst);

    auto team_policy = ThreadTeamPolicy(1u, Kokkos::AUTO);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
          Real inv_density[AeroConfig::num_modes()]
                          [AeroConfig::num_aerosol_ids()] = {};
          Real num2vol_ratio_min[AeroConfig::num_modes()] = {};
          Real num2vol_ratio_max[AeroConfig::num_modes()] = {};
          Real num2vol_ratio_max_nmodes[AeroConfig::num_modes()] = {};
          Real num2vol_ratio_min_nmodes[AeroConfig::num_modes()] = {};
          Real num2vol_ratio_nom_nmodes[AeroConfig::num_modes()] = {};
          Real dgnmin_nmodes[AeroConfig::num_modes()] = {};
          Real dgnmax_nmodes[AeroConfig::num_modes()] = {};
          Real dgnnom_nmodes[AeroConfig::num_modes()] = {};
          Real mean_std_dev_nmodes[AeroConfig::num_modes()] = {};
          // outputs
          bool noxf_acc2ait[AeroConfig::num_aerosol_ids()] = {};
          int n_common_species_ait_accum = {};
          int ait_spec_in_acc[AeroConfig::num_aerosol_ids()] = {};
          int acc_spec_in_ait[AeroConfig::num_aerosol_ids()] = {};
          

          modal_aero_calcsize::init_calcsize(
              inv_density, num2vol_ratio_min, num2vol_ratio_max,
              num2vol_ratio_max_nmodes, num2vol_ratio_min_nmodes,
              num2vol_ratio_nom_nmodes, dgnmin_nmodes, dgnmax_nmodes,
              dgnnom_nmodes, mean_std_dev_nmodes,
              // outputs
              noxf_acc2ait, n_common_species_ait_accum, ait_spec_in_acc,
              acc_spec_in_ait
              );
          int n_common_species_ait_accum_2 = 4; 
          int ait_spec_in_acc_2[4] ={23, 24, 25, 26}; //frm aitken
          int acc_spec_in_ait_2[4] = {15, 17, 20, 21 };// to accum

          const bool do_adjust = true;
          const bool do_aitacc_transfer = true;
          const bool update_mmr = true;

          int nspec_amode[ntot_amode];
          int lspectype_amode[maxd_aspectype][ntot_amode];
          int lmassptr_amode[maxd_aspectype][ntot_amode];
          Real specdens_amode[maxd_aspectype];
          Real spechygro[maxd_aspectype];
          int numptr_amode[ntot_amode];
          int mam_idx[ntot_amode][nspec_max];
          int mam_cnst_idx[ntot_amode][nspec_max];

          ndrop::get_e3sm_parameters(
              nspec_amode, lspectype_amode, lmassptr_amode, numptr_amode,
              specdens_amode, spechygro, mam_idx, mam_cnst_idx);

          // Note: Need to compute inv density using indexing from e3sm
          for (int imode = 0; imode < ntot_amode; ++imode) {
            const int nspec = nspec_amode[imode];
            for (int isp = 0; isp < nspec; ++isp) {
              const int idx = lspectype_amode[isp][imode] - 1;
              inv_density[imode][isp] = 1.0 / specdens_amode[idx];
            } // isp
          }   // imode

          // FIXME: top_lev is set to 1 in calcsize ?
          const int top_lev = 0; // 1( in fortran )

          for (int kk = top_lev; kk < pver; ++kk) {
            printf("---- \n");
            printf("kk %d \n", kk);
            const auto state_q_k = Kokkos::subview(state_q, kk, Kokkos::ALL());
            

            const auto qqcw_k = Kokkos::subview(qqcw, kk, Kokkos::ALL());
            const auto dgncur_i =
                Kokkos::subview(dgnumdry_m, kk, Kokkos::ALL());
            Real dgncur_c[ntot_amode] = {};
            const auto ptend_q_k = Kokkos::subview(ptend_q, kk, Kokkos::ALL());
            const auto dqqcwdt_k = Kokkos::subview(dqqcwdt, kk, Kokkos::ALL());
            modal_aero_calcsize::modal_aero_calcsize_sub(
                state_q_k.data(), // in
                qqcw_k.data(),    // in/out
                dt, do_adjust, do_aitacc_transfer, update_mmr, lmassptr_amode,
                numptr_amode,
                inv_density, // in
                num2vol_ratio_min, num2vol_ratio_max, num2vol_ratio_max_nmodes,
                num2vol_ratio_min_nmodes, num2vol_ratio_nom_nmodes,
                dgnmin_nmodes, dgnmax_nmodes, dgnnom_nmodes,
                mean_std_dev_nmodes,
                // outputs
                noxf_acc2ait, n_common_species_ait_accum_2, ait_spec_in_acc_2,
                acc_spec_in_ait_2, dgncur_i.data(), dgncur_c, ptend_q_k.data(), dqqcwdt_k.data());
          } // k
        });

    constexpr Real zero = 0;
    std::vector<Real> dgnumdry_m_out(pver * ntot_amode, zero);
    mam4::validation::convert_2d_view_device_to_1d_vector(dgnumdry_m,
                                                          dgnumdry_m_out);
    
    std::vector<Real> ptend_q_out(pver * pcnst, zero);
    mam4::validation::convert_2d_view_device_to_1d_vector(ptend_q,
                                                          ptend_q_out);

    Kokkos::deep_copy(qqcw_host, qqcw);
    count = 0;
    for (int kk = 0; kk < pver; ++kk) {
      for (int i = 0; i < pcnst; ++i) {
        qqcw_db[count] = qqcw_host(kk, i);
        count++;
      }
    }

    output.set("qqcw", qqcw_db);
    output.set("dgnumdry_m", dgnumdry_m_out);
    output.set("ptend_q", ptend_q_out);
  });
}
