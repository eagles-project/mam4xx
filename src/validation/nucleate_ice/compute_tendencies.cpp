// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/mam4.hpp>
#include <skywalker.hpp>
#include <validation.hpp>

using namespace skywalker;
using namespace mam4;
using namespace haero;

void compute_tendencies(Ensemble *ensemble) {

  // We don't need any settings for this particular test.
  // Settings settings = ensemble->settings();

  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    // Fetch ensemble parameters

    Real dt = 0;
    Real t = 0;

    int nlev = 1;
    Real pblh = 1000;
    Atmosphere atm = validation::create_atmosphere(nlev, pblh);
    Surface sfc = validation::create_surface();
    mam4::Prognostics progs = validation::create_prognostics(nlev);
    mam4::Diagnostics diags = validation::create_diagnostics(nlev);
    mam4::Tendencies tends = validation::create_tendencies(nlev);

    const Real subgrid = input.get_array("nucleate_ice_subgrid")[0];
    const Real so4_sz_thresh_icenuc =
        input.get_array("so4_sz_thresh_icenuc")[0];

    mam4::AeroConfig mam4_config;
    NucleateIce::Config nucleate_ice_config(subgrid, so4_sz_thresh_icenuc);
    mam4::NucleateIceProcess process(mam4_config, nucleate_ice_config);

    const Real pmid = input.get_array("pmid")[0];        // air pressure
    const Real temp = input.get_array("temperature")[0]; // air temperature
    const Real updraft_vel_ice_nucleation =
        input.get_array("wsubi")[0]; // cloud fraction
    const Real cloud_fraction = input.get_array("ast")[0];

    auto state_q = input.get_array("state_q");
    // qn(pcols,pver)           ! water vapor mixing ratio [kg/kg]
    // qn(:ncol,:pver) = state_q(:ncol,:pver,1)
    const Real vapor_mixing_ratio = state_q[0];

    auto d_temp = validation::create_column_view(nlev);
    auto d_pmid = validation::create_column_view(nlev);
    auto d_cloud_fraction = validation::create_column_view(nlev);
    auto d_updraft_vel_ice_nucleation = validation::create_column_view(nlev);
    auto d_vapor_mixing_ratio = validation::create_column_view(nlev);
    Kokkos::deep_copy(d_temp, temp);
    Kokkos::deep_copy(d_pmid, pmid);
    Kokkos::deep_copy(d_cloud_fraction, cloud_fraction);
    Kokkos::deep_copy(d_updraft_vel_ice_nucleation, updraft_vel_ice_nucleation);
    Kokkos::deep_copy(d_vapor_mixing_ratio, vapor_mixing_ratio);
    atm.temperature = d_temp;
    atm.pressure = d_pmid;
    atm.cloud_fraction = d_cloud_fraction;
    atm.updraft_vel_ice_nucleation = d_updraft_vel_ice_nucleation;
    atm.vapor_mixing_ratio = d_vapor_mixing_ratio;

    auto numptr_amode = input.get_array("numptr_amode");
    auto modeptr_aitken = int(input.get_array("modeptr_aitken")[0] - 1);
    auto modeptr_coarse = int(input.get_array("modeptr_coarse")[0] - 1);

    // mode number mixing ratios
    // const int
    auto num_aitken = state_q[int(numptr_amode[modeptr_aitken]) - 1];
    auto num_coarse = state_q[int(numptr_amode[modeptr_coarse]) - 1];

    // we only copy values of mass m.r that are use in this process.
    // Other values will be equal to zero
    const int aitken_idx = int(ModeIndex::Aitken);
    const int coarse_idx = int(ModeIndex::Coarse);
    Kokkos::deep_copy(progs.n_mode_i[aitken_idx], num_aitken);
    Kokkos::deep_copy(progs.n_mode_i[coarse_idx], num_coarse);

    auto lptr_dust_a_amode = input.get_array("lptr_dust_a_amode");
    auto lptr_nacl_a_amode = input.get_array("lptr_nacl_a_amode");
    auto lptr_so4_a_amode = input.get_array("lptr_so4_a_amode");
    auto lptr_mom_a_amode = input.get_array("lptr_mom_a_amode");
    auto lptr_bc_a_amode = input.get_array("lptr_bc_a_amode");
    auto lptr_pom_a_amode = input.get_array("lptr_pom_a_amode");
    auto lptr_soa_a_amode = input.get_array("lptr_soa_a_amode");

    // mode specie mass m.r.
    // -1 because of indexing in C++
    auto coarse_dust = state_q[int(lptr_dust_a_amode[modeptr_coarse]) - 1];
    auto coarse_nacl = state_q[int(lptr_nacl_a_amode[modeptr_coarse]) - 1];
    auto coarse_so4 = state_q[int(lptr_so4_a_amode[modeptr_coarse]) - 1];
    auto coarse_mom = state_q[int(lptr_mom_a_amode[modeptr_coarse]) - 1];
    auto coarse_bc = state_q[int(lptr_bc_a_amode[modeptr_coarse]) - 1];
    auto coarse_pom = state_q[int(lptr_pom_a_amode[modeptr_coarse]) - 1];
    auto coarse_soa = state_q[int(lptr_soa_a_amode[modeptr_coarse]) - 1];

    const int dst_idx = aerosol_index_for_mode(ModeIndex::Coarse, AeroId::DST);
    const int nacl_idx =
        aerosol_index_for_mode(ModeIndex::Coarse, AeroId::NaCl);
    const int so4_idx = aerosol_index_for_mode(ModeIndex::Coarse, AeroId::SO4);
    const int mom_idx = aerosol_index_for_mode(ModeIndex::Coarse, AeroId::MOM);
    const int bc_idx = aerosol_index_for_mode(ModeIndex::Coarse, AeroId::BC);
    const int pom_idx = aerosol_index_for_mode(ModeIndex::Coarse, AeroId::POM);
    const int soa_idx = aerosol_index_for_mode(ModeIndex::Coarse, AeroId::SOA);

    // we only copy values of mass m.r that are use in this process.
    // Other values will be equal to zero
    Kokkos::deep_copy(progs.q_aero_i[coarse_idx][dst_idx], coarse_dust);
    Kokkos::deep_copy(progs.q_aero_i[coarse_idx][nacl_idx], coarse_nacl);
    Kokkos::deep_copy(progs.q_aero_i[coarse_idx][so4_idx], coarse_so4);
    Kokkos::deep_copy(progs.q_aero_i[coarse_idx][mom_idx], coarse_mom);
    Kokkos::deep_copy(progs.q_aero_i[coarse_idx][bc_idx], coarse_bc);
    Kokkos::deep_copy(progs.q_aero_i[coarse_idx][pom_idx], coarse_pom);
    Kokkos::deep_copy(progs.q_aero_i[coarse_idx][soa_idx], coarse_soa);

    auto dgnum = input.get_array("dgnum");
    Kokkos::deep_copy(diags.dry_geometric_mean_diameter_i[aitken_idx],
                      dgnum[modeptr_aitken]);

    auto team_policy = ThreadTeamPolicy(1u, Kokkos::AUTO);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
          process.compute_tendencies(team, t, dt, atm, sfc, progs, diags,
                                     tends);
        });

    auto h_icenuc_num_hetfrz =
        Kokkos::create_mirror_view(diags.icenuc_num_hetfrz);
    Kokkos::deep_copy(h_icenuc_num_hetfrz, diags.icenuc_num_hetfrz);

    auto h_icenuc_num_immfrz =
        Kokkos::create_mirror_view(diags.icenuc_num_immfrz);
    Kokkos::deep_copy(h_icenuc_num_immfrz, diags.icenuc_num_immfrz);

    auto h_icenuc_num_depnuc =
        Kokkos::create_mirror_view(diags.icenuc_num_depnuc);
    Kokkos::deep_copy(h_icenuc_num_depnuc, diags.icenuc_num_depnuc);

    auto h_icenuc_num_meydep =
        Kokkos::create_mirror_view(diags.icenuc_num_meydep);
    Kokkos::deep_copy(h_icenuc_num_meydep, diags.icenuc_num_meydep);

    auto h_num_act_aerosol_ice_nucle =
        Kokkos::create_mirror_view(diags.num_act_aerosol_ice_nucle);
    Kokkos::deep_copy(h_num_act_aerosol_ice_nucle,
                      diags.num_act_aerosol_ice_nucle);
    auto h_num_act_aerosol_ice_nucle_hom =
        Kokkos::create_mirror_view(diags.num_act_aerosol_ice_nucle_hom);
    Kokkos::deep_copy(h_num_act_aerosol_ice_nucle_hom,
                      diags.num_act_aerosol_ice_nucle_hom);

    output.set("naai", std::vector<Real>(1, h_num_act_aerosol_ice_nucle[0]));
    output.set("naai_hom",
               std::vector<Real>(1, h_num_act_aerosol_ice_nucle_hom[0]));

    output.set("nihf", std::vector<Real>(1, h_icenuc_num_hetfrz[0]));
    output.set("niimm", std::vector<Real>(1, h_icenuc_num_immfrz[0]));
    output.set("nidep", std::vector<Real>(1, h_icenuc_num_depnuc[0]));
    output.set("nimey", std::vector<Real>(1, h_icenuc_num_meydep[0]));
  });
}
