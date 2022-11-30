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

    Real dt = input.get("dt");
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

    auto q_i = input.get_array("interstitial");
    auto n_i = input.get_array("interstitial_num");
    auto q_c = input.get_array("cloud_borne");
    auto n_c = input.get_array("cloud_borne_num");

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
    }   // end modes

    auto team_policy = ThreadTeamPolicy(1u, Kokkos::AUTO);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
          process.compute_tendencies(team, t, dt, atm, progs, diags, tends);
        });

    std::vector<Real> tend_aero_i_out;
    std::vector<Real> tend_n_mode_i_out;

    std::vector<Real> tend_aero_c_out;
    std::vector<Real> tend_n_mode_c_out;

    std::vector<Real> diags_dgncur_i;
    std::vector<Real> diags_dgncur_c;

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

      // diameter interstitial
      auto h_dgncur_i = Kokkos::create_mirror_view(diags.dgncur_i[imode]);
      Kokkos::deep_copy(diags.dgncur_i[imode], h_dgncur_i);
      diags_dgncur_i.push_back(h_dgncur_i(0));

      // diameter cloud_borne
      auto h_dgncur_c = Kokkos::create_mirror_view(diags.dgncur_c[imode]);
      Kokkos::deep_copy(diags.dgncur_c[imode], h_dgncur_c);
      diags_dgncur_c.push_back(h_dgncur_c(0));


    }   // end mode

    output.set("interstitial_ptend", tend_aero_i_out);
    output.set("interstitial_ptend_num", tend_n_mode_i_out);

    output.set("cloud_borne_ptend_num", tend_n_mode_c_out);
    output.set("cloud_borne_ptend", tend_aero_c_out);

    // output.set("cloud_borne_diameter", diags_dgncur_c);
    // output.set("interstitial_diameter", diags_dgncur_i);
    output.set("diameter", diags_dgncur_i);

    // add more outputs (diagnostics)
  });
}
