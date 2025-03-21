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

void aero_model_calcsize_water_uptake_dr(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    constexpr int pcnst = aero_model::pcnst;
    constexpr int pver = ndrop::pver;
    constexpr int ntot_amode = AeroConfig::num_modes();

    using View2D = DeviceType::view_2d<Real>;
    constexpr Real zero = 0.0;

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
    if (input.has_array("dgncur_a")) {
      auto dgncur_a_db = input.get_array("dgncur_a");
      mam4::validation::convert_1d_vector_to_2d_view_device(dgncur_a_db,
                                                            dgnumdry_m);
    }

    View2D ptend_q("ptend_q", pver, pcnst);
    View2D dqqcwdt("dqqcwdt", pver, pcnst);

    wetdep::View2D qaerwat("qaerwat", pver, ntot_amode);
    const auto qaerwat_db = input.get_array("qaerwat");
    mam4::validation::convert_1d_vector_to_2d_view_device(qaerwat_db, qaerwat);

    wetdep::View2D wetdens("wetdens", pver, ntot_amode);
    const auto wetdens_db = input.get_array("wetdens");
    mam4::validation::convert_1d_vector_to_2d_view_device(wetdens_db, wetdens);

    wetdep::View2D dgnumwet("dgnumwet", pver, ntot_amode);
    const auto dgnumwet_db = input.get_array("dgnumwet");
    mam4::validation::convert_1d_vector_to_2d_view_device(dgnumwet_db,
                                                          dgnumwet);

    ColumnView temperature =
        validation::get_input_in_columnview(input, "temperature");
    ColumnView pmid = validation::get_input_in_columnview(input, "pmid");

    ColumnView cldn = validation::get_input_in_columnview(input, "cldn");

    mam4::modal_aero_calcsize::CalcsizeData cal_data;
    cal_data.initialize();
    const bool update_mmr = true;
    cal_data.set_update_mmr(update_mmr);

    auto team_policy = ThreadTeamPolicy(1u, Kokkos::AUTO);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
          // FIXME: top_lev is set to 1 in calcsize ?
          const int top_lev = 0; // 1( in fortran )

          Kokkos::parallel_for(
              Kokkos::TeamVectorRange(team, top_lev, pver), [&](int kk) {
                const auto state_q_k =
                    Kokkos::subview(state_q, kk, Kokkos::ALL());
                const auto qqcw_k = Kokkos::subview(qqcw, kk, Kokkos::ALL());
                const auto dgncur_i =
                    Kokkos::subview(dgnumdry_m, kk, Kokkos::ALL());
                Real dgncur_c[ntot_amode] = {};
                const auto ptend_q_k =
                    Kokkos::subview(ptend_q, kk, Kokkos::ALL());
                const auto dqqcwdt_k =
                    Kokkos::subview(dqqcwdt, kk, Kokkos::ALL());
                modal_aero_calcsize::modal_aero_calcsize_sub(
                    state_q_k.data(), // in
                    qqcw_k.data(),    // in/out
                    dt, cal_data,
                    // outputs
                    dgncur_i.data(), dgncur_c, ptend_q_k.data(),
                    dqqcwdt_k.data());

                const auto dgnumwet_kk =
                    Kokkos::subview(dgnumwet, kk, Kokkos::ALL());

                const auto qaerwat_kk =
                    Kokkos::subview(qaerwat, kk, Kokkos::ALL());

                const auto wetdens_kk =
                    Kokkos::subview(wetdens, kk, Kokkos::ALL());

                mam4::water_uptake::modal_aero_water_uptake_dr(
                    cal_data.nspec_amode, cal_data.specdens_amode,
                    cal_data.spechygro, cal_data.lspectype_amode,
                    state_q_k.data(), temperature(kk), pmid(kk), cldn(kk),
                    dgncur_i.data(), dgnumwet_kk.data(), qaerwat_kk.data(),
                    wetdens_kk.data());

                if (update_mmr) {
                  // Note: it only needs to update aerosol variables.
                  for (int i = utils::aero_start_ind(); i < pcnst; ++i) {
                    qqcw(kk, i) =
                        haero::max(zero, qqcw(kk, i) + dqqcwdt(kk, i) * dt);
                  }
                }
              }); // k
        });

    std::vector<Real> dgnumdry_m_out(pver * ntot_amode, zero);
    mam4::validation::convert_2d_view_device_to_1d_vector(dgnumdry_m,
                                                          dgnumdry_m_out);
    output.set("dgnumdry_m", dgnumdry_m_out);

    std::vector<Real> dgnumwet_out(pver * ntot_amode, zero);
    mam4::validation::convert_2d_view_device_to_1d_vector(dgnumwet,
                                                          dgnumwet_out);
    output.set("dgnumwet", dgnumwet_out);

    std::vector<Real> wetdens_out(pver * ntot_amode, zero);
    mam4::validation::convert_2d_view_device_to_1d_vector(wetdens, wetdens_out);
    output.set("wetdens", wetdens_out);

    std::vector<Real> qaerwat_out(pver * ntot_amode, zero);
    mam4::validation::convert_2d_view_device_to_1d_vector(qaerwat, qaerwat_out);
    output.set("qaerwat", qaerwat_out);

    std::vector<Real> ptend_q_out(pver * pcnst, zero);
    mam4::validation::convert_2d_view_device_to_1d_vector(ptend_q, ptend_q_out);

    std::vector<Real> qqcw_out(pver * pcnst, zero);
    mam4::validation::convert_transpose_2d_view_device_to_1d_vector(qqcw,
                                                                    qqcw_out);
    output.set("qqcw", qqcw_out);
    output.set("ptend_q", ptend_q_out);

    std::vector<Real> state_q_out(pver * 40, zero);
    mam4::validation::convert_2d_view_device_to_1d_vector(state_q, state_q_out);
    output.set("state_q", state_q_out);
  });
}
