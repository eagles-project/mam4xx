// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/mam4.hpp>

#include <mam4xx/aero_config.hpp>
#include <mam4xx/mam4.hpp>
#include <skywalker.hpp>
#include <validation.hpp>
using namespace skywalker;
using namespace mam4;
using namespace haero;
using namespace haero::testing;

void aero_model_wetdep(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    using View1DHost = typename HostType::view_1d<Real>;
    using View2D = DeviceType::view_2d<Real>;

    mam4::Prognostics progs = validation::create_prognostics(nlev);
    mam4::Tendencies tends = validation::create_tendencies(nlev);
    int nlev = mam4::nlev;
    Real pblh = 1000;
    const Real dt = 1;
    //
    View2D state_q("state_q", nlev, aero_model::pcnst);
    const auto state_q_db = input.get_array("state_q");
    validation::convert_1d_vector_to_2d_view_device(state_q_db, state_q);
    auto qqcw_db = input.get_array("qqcw"); // 2d

    View2D qqcw("qqcw", nlev, aero_model::pcnst);
    auto qqcw_host = Kokkos::create_mirror_view(qqcw);
    int count = 0;
    for (int kk = 0; kk < nlev; ++kk) {
      for (int i = 0; i < aero_model::pcnst; ++i) {
        qqcw_host(kk, i) = qqcw_db[count];
        count++;
      }
    }
    Kokkos::deep_copy(qqcw, qqcw_host);

    ColumnView temperature =
        validation::get_input_in_columnview(input, "temperature");
    ColumnView pressure = validation::get_input_in_columnview(input, "pmid");
    ColumnView hydrostatic_dp =
        validation::get_input_in_columnview(input, "pdel");

    auto liquid_mixing_ratio = create_column_view(nlev); //

    auto ice_mixing_ratio = create_column_view(nlev); //
    // We need deep_copy because of executation error due to different layout
    // q[1] = atm.liquid_mixing_ratio(klev);              // qc
    Kokkos::deep_copy(liquid_mixing_ratio,
                      Kokkos::subview(state_q, Kokkos::ALL(), 1));
    // q[2] = atm.ice_mixing_ratio(klev);                 // qi
    Kokkos::deep_copy(ice_mixing_ratio,
                      Kokkos::subview(state_q, Kokkos::ALL(), 2));

    const auto cldn_db = input.get_array("cldn");
    auto vapor_mixing_ratio = create_column_view(nlev);
    auto cloud_liquid_number_mixing_ratio = create_column_view(nlev);
    auto cloud_ice_number_mixing_ratio = create_column_view(nlev);
    auto height = create_column_view(nlev);
    auto interface_pressure = create_column_view(nlev + 1);
    auto cloud_fraction = create_column_view(nlev);
    auto updraft_vel_ice_nucleation = create_column_view(nlev);

    auto atm = Atmosphere(nlev, temperature, pressure, vapor_mixing_ratio,
                          liquid_mixing_ratio, cloud_liquid_number_mixing_ratio,
                          ice_mixing_ratio, cloud_ice_number_mixing_ratio,
                          height, hydrostatic_dp, interface_pressure,
                          cloud_fraction, updraft_vel_ice_nucleation, pblh);
    // inputs
    ColumnView cldt = validation::get_input_in_columnview(input, "cldn");
    // Note that itim and itim_old are used separately for the cld variables
    // although they are the same, as also indicated by the discussion on the
    // Confluence page
    ColumnView cldn_prev_step = cldt; // d

    ColumnView rprdsh = validation::get_input_in_columnview(input, "rprdsh");
    ColumnView rprddp = validation::get_input_in_columnview(input, "rprddp");
    ColumnView evapcdp = validation::get_input_in_columnview(input, "evapcdp");
    ColumnView evapcsh = validation::get_input_in_columnview(input, "evapcsh");

    ColumnView dp_frac =
        validation::get_input_in_columnview(input, "p_dp_frac");
    ColumnView sh_frac =
        validation::get_input_in_columnview(input, "p_sh_frac");
    ColumnView icwmrdp =
        validation::get_input_in_columnview(input, "p_icwmrdp");
    ColumnView icwmrsh =
        validation::get_input_in_columnview(input, "p_icwmrsh");

    ColumnView evapr =
        validation::get_input_in_columnview(input, "inputs_evapr"); //

    // outputs
    ColumnView dlf =
        validation::get_input_in_columnview(input, "inputs_evapr"); //
    wetdep::View1D aerdepwetcw("aerdepwetcw", aero_model::pcnst);
    wetdep::View1D aerdepwetis("aerdepwetis", aero_model::pcnst);
    const int num_modes = AeroConfig::num_modes();

    wetdep::View2D wet_geometric_mean_diameter_i(
        "wet_geometric_mean_diameter_i", num_modes, nlev);
    const auto dgnumwet_db = input.get_array("dgnumwet");
    mam4::validation::convert_1d_vector_to_2d_view_device(
        dgnumwet_db, wet_geometric_mean_diameter_i);

    wetdep::View2D dry_geometric_mean_diameter_i(
        "dry_geometric_mean_diameter_i", num_modes, nlev);
    const auto dgncur_a_db = input.get_array("dgncur_a");
    mam4::validation::convert_1d_vector_to_2d_view_device(
        dgncur_a_db, dry_geometric_mean_diameter_i);

    wetdep::View2D qaerwat("qaerwat", num_modes, nlev);
    const auto qaerwat_db = input.get_array("qaerwat");
    mam4::validation::convert_1d_vector_to_2d_view_device(qaerwat_db, qaerwat);

    wetdep::View2D wetdens("wetdens", num_modes, nlev);
    const auto wetdens_db = input.get_array("wetdens");
    mam4::validation::convert_1d_vector_to_2d_view_device(wetdens_db, wetdens);

    wetdep::View2D ptend_q("ptend_q", nlev, aero_model::pcnst);

    // work arrays
    const int work_len = wetdep::get_aero_model_wetdep_work_len();
    wetdep::View1D work("work", work_len);

    auto team_policy = ThreadTeamPolicy(1u, Kokkos::AUTO);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
          auto progs_in = progs;
          auto tends_in = tends;

          // we need to inject validation values to progs.
          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team, nlev), [&](int kk) {
                // copy data from prog to stateq
                const auto state_q_kk = ekat::subview(state_q, kk);
                const auto qqcw_kk = ekat::subview(qqcw, kk);
                utils::inject_qqcw_to_prognostics(qqcw_kk.data(), progs_in, kk);
                utils::inject_stateq_to_prognostics(state_q_kk.data(), progs_in,
                                                    kk);
              });
          team.team_barrier();

          wetdep::aero_model_wetdep(
              team, atm, progs_in, tends_in, dt,
              // inputs
              cldt, cldn_prev_step, rprdsh, rprddp, evapcdp, evapcsh, dp_frac,
              sh_frac, icwmrdp, icwmrsh, evapr,
              // outputs
              dlf, wet_geometric_mean_diameter_i, dry_geometric_mean_diameter_i,
              qaerwat, wetdens,
              // output
              aerdepwetis, aerdepwetcw, work);

          team.team_barrier();
          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team, 0, nlev), [&](int kk) {
                const auto ptend_q_kk = ekat::subview(ptend_q, kk);
                utils::extract_ptend_from_tendencies(tends_in,
                                                     ptend_q_kk.data(), kk);
              });
        });

    // std::vector<Real> dlf_output(nlev, 0);
    // auto dlf_host = View1DHost((Real *)dlf_output.data(), nlev);
    // Kokkos::deep_copy(dlf, dlf_host);
    // output.set("dlf", dlf_output);

    std::vector<Real> output_pcnst(aero_model::pcnst, 0);
    auto aerdepwetcw_host =
        View1DHost((Real *)output_pcnst.data(), aero_model::pcnst);
    Kokkos::deep_copy(aerdepwetcw, aerdepwetcw_host);
    output.set("aerdepwetcw", output_pcnst);

    std::vector<Real> aerdepwetis_output(aero_model::pcnst, 0);
    auto aerdepwetis_host =
        View1DHost((Real *)output_pcnst.data(), aero_model::pcnst);
    Kokkos::deep_copy(aerdepwetis, aerdepwetis_host);
    output.set("aerdepwetis", output_pcnst);

    std::vector<Real> output_modes(nlev * num_modes, 0);
    mam4::validation::convert_transpose_2d_view_device_to_1d_vector(
        wet_geometric_mean_diameter_i, output_modes);
    output.set("dgnumwet", output_modes);

    mam4::validation::convert_transpose_2d_view_device_to_1d_vector(
        dry_geometric_mean_diameter_i, output_modes);
    output.set("dgncur_a", output_modes);

    mam4::validation::convert_transpose_2d_view_device_to_1d_vector(qaerwat,
                                                          output_modes);
    output.set("qaerwat", output_modes);

    mam4::validation::convert_transpose_2d_view_device_to_1d_vector(wetdens,
                                                          output_modes);
    output.set("wetdens", output_modes);

    std::vector<Real> output_ptend(nlev * aero_model::pcnst, 0);
    mam4::validation::convert_2d_view_device_to_1d_vector(ptend_q,
                                                          output_ptend);
    output.set("ptend_lq", output_ptend);
  });
}
