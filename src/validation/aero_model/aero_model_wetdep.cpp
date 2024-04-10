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
    mam4::Prognostics progs = validation::create_prognostics(nlev);
    mam4::Tendencies tends = validation::create_tendencies(nlev);
    int nlev = mam4::nlev;
    Real pblh = 1000;
    // Atmosphere atm = validation::create_atmosphere(nlev, pblh);
    // FIXME get dt yaml
    const Real dt = 1;
    //

    auto temperature = create_column_view(nlev);
    auto pressure = create_column_view(nlev);
    auto vapor_mixing_ratio = create_column_view(nlev);
    auto liquid_mixing_ratio = create_column_view(nlev);
    auto cloud_liquid_number_mixing_ratio = create_column_view(nlev);
    auto ice_mixing_ratio = create_column_view(nlev);
    auto cloud_ice_number_mixing_ratio = create_column_view(nlev);
    auto height = create_column_view(nlev);
    auto hydrostatic_dp = create_column_view(nlev);
    auto interface_pressure = create_column_view(nlev + 1);
    auto cloud_fraction = create_column_view(nlev);
    auto updraft_vel_ice_nucleation = create_column_view(nlev);
    // FIXME: update these values.
    Kokkos::deep_copy(temperature, 300);
    Kokkos::deep_copy(pressure, 1e5);
    Kokkos::deep_copy(hydrostatic_dp, 1e5);

    auto atm = Atmosphere(nlev, temperature, pressure, vapor_mixing_ratio,
                          liquid_mixing_ratio, cloud_liquid_number_mixing_ratio,
                          ice_mixing_ratio, cloud_ice_number_mixing_ratio,
                          height, hydrostatic_dp, interface_pressure,
                          cloud_fraction, updraft_vel_ice_nucleation, pblh);

    // inputs
    ColumnView cldn_prev_step = create_column_view(nlev);
    ColumnView cldt = create_column_view(nlev);
    ColumnView rprdsh = create_column_view(nlev);
    ColumnView rprddp = create_column_view(nlev);
    ColumnView evapcdp = create_column_view(nlev);
    ColumnView evapcsh = create_column_view(nlev);
    ColumnView dp_frac = create_column_view(nlev);
    ColumnView sh_frac = create_column_view(nlev);
    ColumnView icwmrdp = create_column_view(nlev);
    ColumnView icwmrsh = create_column_view(nlev);
    ColumnView evapr = create_column_view(nlev);

    // outputs
    ColumnView dlf = create_column_view(nlev);
    wetdep::View1D aerdepwetcw("aerdepwetcw", aero_model::pcnst);
    wetdep::View1D aerdepwetis("aerdepwetis", aero_model::pcnst);
    const int num_modes = AeroConfig::num_modes();

    wetdep::View2D wet_geometric_mean_diameter_i("wet_geometric_mean_diameter_i",num_modes, nlev);
    Kokkos::deep_copy(wet_geometric_mean_diameter_i, 1.0);

    wetdep::View2D dry_geometric_mean_diameter_i("dry_geometric_mean_diameter_i",num_modes, nlev);
    Kokkos::deep_copy(dry_geometric_mean_diameter_i, 1.0);

    wetdep::View2D qaerwat("qaerwat",num_modes, nlev);
    Kokkos::deep_copy(qaerwat, 1.0);

    wetdep::View2D wetdens("wetdens",num_modes, nlev);
    Kokkos::deep_copy(wetdens, 1.0);

    wetdep::View2D ptend_q("ptend_q",nlev, aero_model::pcnst);


    const int work_len = wetdep::get_aero_model_wetdep_work_len();
    wetdep::View1D work("work", work_len);

    Kokkos::View<Real * [aero_model::maxd_aspectype + 2][aero_model::pcnst]>
        qqcw_sav("qqcw_sav", nlev);

    auto team_policy = ThreadTeamPolicy(1u, Kokkos::AUTO);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
          auto progs_in = progs;
          auto tends_in = tends;
          wetdep::aero_model_wetdep(team, atm, progs_in, tends_in, dt,
                                    // inputs
                                    cldt, 
                                    cldn_prev_step, rprdsh, rprddp, evapcdp,
                                    evapcsh, dp_frac, sh_frac,
                                    icwmrdp, icwmrsh, evapr,
                                    dlf,
                                    wet_geometric_mean_diameter_i,
                                    dry_geometric_mean_diameter_i, 
                                    qaerwat, 
                                    wetdens, 
                                    // output
                                    aerdepwetis, aerdepwetcw,
                                    // FIXME
                                    qqcw_sav, work);
    
    team.team_barrier();
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 0, nlev), [&](int kk) {
    const auto ptend_q_kk = ekat::subview(ptend_q, kk);
    utils::extract_ptend_from_tendencies(tends_in, ptend_q_kk.data(), kk);
    });

    });

    std::vector<Real> dlf_output(nlev, 0);
    auto dlf_host = View1DHost((Real *)dlf_output.data(), nlev);
    Kokkos::deep_copy(dlf, dlf_host);
    output.set("dlf", dlf_output);

    std::vector<Real> output_pcnst(aero_model::pcnst, 0);
    auto aerdepwetcw_host = View1DHost((Real *)output_pcnst.data(), aero_model::pcnst);
    Kokkos::deep_copy(aerdepwetcw, aerdepwetcw_host);
    output.set("aerdepwetcw", output_pcnst);

    std::vector<Real> aerdepwetis_output(aero_model::pcnst, 0);
    auto aerdepwetis_host = View1DHost((Real *)output_pcnst.data(), aero_model::pcnst);
    Kokkos::deep_copy(aerdepwetis, aerdepwetis_host);
    output.set("aerdepwetis", output_pcnst);

    std::vector<Real> output_modes(nlev * num_modes, 0);
    mam4::validation::convert_2d_view_device_to_1d_vector(wet_geometric_mean_diameter_i,
     output_modes);
    output.set("dgnumwet", output_modes);

    mam4::validation::convert_2d_view_device_to_1d_vector(dry_geometric_mean_diameter_i,
     output_modes);
    output.set("dgnumdry", output_modes);

    mam4::validation::convert_2d_view_device_to_1d_vector(qaerwat,
     output_modes);
    output.set("qaerwat", output_modes);

    mam4::validation::convert_2d_view_device_to_1d_vector(wetdens,
     output_modes);
    output.set("wetdens", output_modes);

    std::vector<Real> output_ptend(nlev * aero_model::pcnst, 0);
    mam4::validation::convert_2d_view_device_to_1d_vector(ptend_q,
     output_ptend);
    output.set("ptend_q", output_ptend);




  });
}
