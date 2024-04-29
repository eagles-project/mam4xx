// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/mam4.hpp>

#include <ekat/kokkos/ekat_subview_utils.hpp>
#include <haero/math.hpp>
#include <mam4xx/aero_config.hpp>
#include <skywalker.hpp>
#include <validation.hpp>

using namespace skywalker;
using namespace mam4;
using namespace haero;
using namespace validation;

void data_transfer_state_q_qqwc_to_prog(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    constexpr int pver = mam4::nlev;
    constexpr int pcnst = aero_model::pcnst;
    using View2D = DeviceType::view_2d<Real>;

    const auto state_q_db = input.get_array("state_q");
    auto qqcw_db = input.get_array("qqcw"); // 2d
    Real pblh = 1000;

    // using skywalker to get state_q and qqcw
    View2D state_q("state_q", pver, pcnst);
    mam4::validation::convert_1d_vector_to_2d_view_device(state_q_db, state_q);

    View2D qqcw("qqcw", pver, pcnst);
    auto qqcw_host = Kokkos::create_mirror_view(qqcw);

    int count = 0;
    for (int kk = 0; kk < pver; ++kk) {
      for (int i = 0; i < pcnst; ++i) {
        qqcw_host(kk, i) = qqcw_db[count];
        count++;
      }
    }
    Kokkos::deep_copy(qqcw, qqcw_host);

    // we do not need temperature, pressure, and hydrostatic_dp in this test
    ColumnView temperature = create_column_view(nlev);
    ColumnView pressure = create_column_view(nlev);
    ColumnView hydrostatic_dp = create_column_view(nlev);

    auto vapor_mixing_ratio = create_column_view(nlev);
    auto liquid_mixing_ratio = create_column_view(nlev); //
    auto ice_mixing_ratio = create_column_view(nlev);    //
    auto cloud_liquid_number_mixing_ratio = create_column_view(nlev);
    auto cloud_ice_number_mixing_ratio = create_column_view(nlev);
    // Some variables from state_q are part of atm.
    // We need deep_copy because of executation error due to different layout
    // q[0] = atm.vapor_mixing_ratio(klev);               // qv
    Kokkos::deep_copy(vapor_mixing_ratio,
                      Kokkos::subview(state_q, Kokkos::ALL(), 0));
    // q[1] = atm.liquid_mixing_ratio(klev);              // qc
    Kokkos::deep_copy(liquid_mixing_ratio,
                      Kokkos::subview(state_q, Kokkos::ALL(), 1));
    // q[2] = atm.ice_mixing_ratio(klev);                 // qi
    Kokkos::deep_copy(ice_mixing_ratio,
                      Kokkos::subview(state_q, Kokkos::ALL(), 2));
    // q[3] = atm.cloud_liquid_number_mixing_ratio(klev); //  nc
    Kokkos::deep_copy(cloud_liquid_number_mixing_ratio,
                      Kokkos::subview(state_q, Kokkos::ALL(), 3));
    // q[4] = atm.cloud_ice_number_mixing_ratio(klev);    // ni
    Kokkos::deep_copy(cloud_ice_number_mixing_ratio,
                      Kokkos::subview(state_q, Kokkos::ALL(), 4));

    auto height = create_column_view(nlev);
    auto interface_pressure = create_column_view(nlev + 1);
    auto cloud_fraction = create_column_view(nlev);
    auto updraft_vel_ice_nucleation = create_column_view(nlev);

    auto atm = Atmosphere(nlev, temperature, pressure, vapor_mixing_ratio,
                          liquid_mixing_ratio, cloud_liquid_number_mixing_ratio,
                          ice_mixing_ratio, cloud_ice_number_mixing_ratio,
                          height, hydrostatic_dp, interface_pressure,
                          cloud_fraction, updraft_vel_ice_nucleation, pblh);

    View2D state_q_output("state_q_output", pver, pcnst);
    View2D qqcw_output("qqcw_output", pver, pcnst);

    View2D diff_state_q("diff_state_q", pver, pcnst);
    View2D diff_qqcw("diff_qqcw", pver, pcnst);

    using range_type = Kokkos::pair<int, int>;
    // inject_stateq_to_prognostics does not set values for index lower than
    // utils::aero_start_ind().
    const auto &qqcw_output_non = Kokkos::subview(
        qqcw_output, Kokkos::ALL, range_type(0, utils::aero_start_ind()));
    Kokkos::deep_copy(qqcw_output_non, -9999.900390625);

    // inject_stateq_to_prognostics is not emplying values from index 5 to
    // utils::gasses_start_ind() Hence, set outout view with these values.
    const auto &state_q_output_non = Kokkos::subview(
        state_q_output, Kokkos::ALL, range_type(5, utils::gasses_start_ind()));
    const auto &state_non = Kokkos::subview(
        state_q, Kokkos::ALL, range_type(5, utils::gasses_start_ind()));

    Kokkos::deep_copy(state_q_output_non, state_non);

    mam4::Prognostics progs = validation::create_prognostics(nlev);

    auto team_policy = ThreadTeamPolicy(1u, Kokkos::AUTO);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
          // 1. We inject values of staque_q in prog.
          //  inject_qqcw_to_prognostics and inject_stateq_to_prognostics are
          //  mostly use for testing.
          auto progs_in = progs;
          // we need to inject validation values to progs.
          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team, nlev), [&](int kk) {
                // copy data from prog to stateq
                const auto &state_q_kk = ekat::subview(state_q, kk);
                const auto qqcw_kk = ekat::subview(qqcw, kk);
                utils::inject_qqcw_to_prognostics(qqcw_kk.data(), progs_in, kk);
                utils::inject_stateq_to_prognostics(state_q_kk.data(), progs_in,
                                                    kk);
              });
          team.team_barrier();
          // 2. Let's extact state_q and qqcw from prog.
          // Currently, many mam4xx function are using
          // extract_stateq_from_prognostics and extract_qqcw_from_prognostics,
          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team, pver), [&](int kk) {
                const auto state_q_output_kk =
                    Kokkos::subview(state_q_output, kk, Kokkos::ALL());
                const auto qqcw_output_k =
                    Kokkos::subview(qqcw_output, kk, Kokkos::ALL());
                utils::extract_stateq_from_prognostics(
                    progs, atm, state_q_output_kk.data(), kk);
                utils::extract_qqcw_from_prognostics(progs,
                                                     qqcw_output_k.data(), kk);
              });

          team.team_barrier();
          // 3. Let's compute the difference between the original state_q (or
          // qqcw ) with the one obtain after extracting the data from prog.
          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team, pver), [&](int kk) {
                Kokkos::parallel_for(
                    Kokkos::ThreadVectorRange(team, pcnst), [&](int isp) {
                      diff_state_q(isp, kk) =
                          state_q(isp, kk) - state_q_output(isp, kk);
                      diff_qqcw(isp, kk) = qqcw(isp, kk) - qqcw_output(isp, kk);
                    });
              });
        });

    constexpr int zero = 0.0;
    std::vector<Real> diff_state_q_out(pver * pcnst, zero);
    mam4::validation::convert_2d_view_device_to_1d_vector(diff_state_q,
                                                          diff_state_q_out);
    output.set("diff_state_q", diff_state_q_out);

    std::vector<Real> diff_diff_qqcw_out(pver * pcnst, zero);
    mam4::validation::convert_2d_view_device_to_1d_vector(diff_qqcw,
                                                          diff_diff_qqcw_out);
    output.set("diff_qqcw", diff_diff_qqcw_out);
  });
}