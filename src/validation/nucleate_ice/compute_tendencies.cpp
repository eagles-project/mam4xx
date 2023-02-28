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
    Atmosphere atm(nlev, pblh);
    mam4::Prognostics progs(nlev);
    mam4::Diagnostics diags(nlev);
    mam4::Tendencies tends(nlev);

    mam4::AeroConfig mam4_config;
    mam4::CalcSizeProcess process(mam4_config);
    const auto nmodes = mam4_config.num_modes();
    const Real pmid = input.get("pressure"); // air pressure
    const Real temp = input.get("temperature"); // air temperature
    const Real updraft_vel_ice_nucleation = input.get("updraft_vel_ice_nucleation"); // cloud fraction
    const Real cloud_fraction = input.get("cloud_fraction");

    auto q_i = input.get_array("interstitial");
    auto n_i = input.get_array("interstitial_num");

    Kokkos::deep_copy(atm.temperature, temp);
    Kokkos::deep_copy(atm.pressure, pmid);
    Kokkos::deep_copy(atm.cloud_fraction, cloud_fraction);
    Kokkos::deep_copy(atm.updraft_vel_ice_nucleation, updraft_vel_ice_nucleation);


    const Real dry_diameter_aitken =  input.get("dry_diameter_aitken");
    const int aitken_idx = int(ModeIndex::Aitken); 
    Kokkos::deep_copy(diags.dry_geometric_mean_diameter_i[aitken_idx], dry_diameter_aitken);

    int count = 0;
    for (int imode = 0; imode < nmodes; ++imode) {

      auto h_prog_n_mode_i = Kokkos::create_mirror_view(progs.n_mode_i[imode]);
      h_prog_n_mode_i(0) = n_i[imode];
      Kokkos::deep_copy(progs.n_mode_i[imode], h_prog_n_mode_i);

      const auto n_spec = num_species_mode(imode);
      for (int isp = 0; isp < n_spec; ++isp) {
        // correcting index for inputs.
        const int isp_mam4xx =
            validation::e3sm_to_mam4xx_aerosol_idx[imode][isp];
        auto h_prog_aero_i =
            Kokkos::create_mirror_view(progs.q_aero_i[imode][isp_mam4xx]);
        h_prog_aero_i(0) = q_i[count];
        Kokkos::deep_copy(progs.q_aero_i[imode][isp_mam4xx], h_prog_aero_i);
        count++;
      } // end species
    }   // end modes

    auto team_policy = ThreadTeamPolicy(1u, Kokkos::AUTO);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
          process.compute_tendencies(team, t, dt, atm, progs, diags, tends);
    });

  
    auto h_icenuc_num_hetfrz= Kokkos::create_mirror_view(diags.icenuc_num_hetfrz);
    Kokkos::deep_copy(h_icenuc_num_hetfrz, diags.icenuc_num_hetfrz);

    auto h_icenuc_num_immfrz = Kokkos::create_mirror_view(diags.icenuc_num_immfrz);
    Kokkos::deep_copy(h_icenuc_num_immfrz,diags.icenuc_num_immfrz);

    auto h_icenuc_num_depnuc = Kokkos::create_mirror_view(diags.icenuc_num_depnuc);
    Kokkos::deep_copy(h_icenuc_num_depnuc,diags.icenuc_num_depnuc);

    auto h_icenuc_num_meydep = Kokkos::create_mirror_view(diags.icenuc_num_meydep);
    Kokkos::deep_copy(h_icenuc_num_meydep,diags.icenuc_num_meydep);

    output.set("icenuc_num_hetfrz", h_icenuc_num_hetfrz[0]);
    output.set("icenuc_num_immfrz", h_icenuc_num_immfrz[0]);
    output.set("icenuc_num_depnuc", h_icenuc_num_depnuc[0]);
    output.set("icenuc_num_meydep", h_icenuc_num_meydep[0]);


  });
}
