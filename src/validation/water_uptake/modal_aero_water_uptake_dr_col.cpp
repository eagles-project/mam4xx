// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <iostream>
#include <mam4xx/aero_modes.hpp>
#include <mam4xx/mam4.hpp>
#include <mam4xx/water_uptake.hpp>
#include <skywalker.hpp>
#include <validation.hpp>

using namespace skywalker;
using namespace mam4;
using namespace haero;

void modal_aero_water_uptake_dr_col(Ensemble *ensemble) {

  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    EKAT_REQUIRE_MSG(input.has_array("state_q"), "Required name: state_q");
    EKAT_REQUIRE_MSG(input.has_array("temperature"),
                     "Required name: temperature");
    EKAT_REQUIRE_MSG(input.has_array("pmid"), "Required name: pmid");
    EKAT_REQUIRE_MSG(input.has_array("cldn"), "Required name: cldn");
    EKAT_REQUIRE_MSG(input.has_array("dgncur_a"), "Required name: dgncur_a");
    EKAT_REQUIRE_MSG(input.has_array("dgncur_awet"),
                     "Required name: dgncur_awet");
    EKAT_REQUIRE_MSG(input.has_array("qaerwat"), "Required name: qaerwat");

    auto temperature_db = input.get_array("temperature"); // done
    auto pmid_db = input.get_array("pmid");               // done
    auto cldn_db = input.get_array("cldn");               // done
    auto state_q_db = input.get_array("state_q");         // done
    auto dgncur_a_db = input.get_array("dgncur_a");
    auto dgncur_awet_db = input.get_array("dgncur_awet");
    auto qaerwat_db = input.get_array("qaerwat");

    using View2D = typename DeviceType::view_2d<Real>;
    using View1DHost = typename HostType::view_1d<Real>;

    constexpr int pcnst = aero_model::pcnst;
    constexpr int nlev = mam4::nlev;
    constexpr int ntot_amode = AeroConfig::num_modes();

    View2D state_q("state_q", nlev, pcnst);
    mam4::validation::convert_1d_vector_to_2d_view_device(state_q_db, state_q);

    ColumnView temperature;
    temperature = haero::testing::create_column_view(nlev);
    auto temperature_host = View1DHost((Real *)temperature_db.data(), nlev);

    ColumnView pmid;
    pmid = haero::testing::create_column_view(nlev);
    auto pmid_host = View1DHost((Real *)pmid_db.data(), nlev);

    ColumnView cldn;
    cldn = haero::testing::create_column_view(nlev);
    auto cldn_host = View1DHost((Real *)cldn_db.data(), nlev);

    Kokkos::deep_copy(temperature, temperature_host);
    Kokkos::deep_copy(pmid, pmid_host);
    Kokkos::deep_copy(cldn, cldn_host);

    View2D dgncur_a("dgncur_a_db", nlev, ntot_amode);
    View2D dgncur_awet("dgncur_awet", nlev, ntot_amode);
    View2D qaerwat("qaerwat", nlev, ntot_amode);
    mam4::validation::convert_1d_vector_to_2d_view_device(dgncur_a_db,
                                                          dgncur_a);

    mam4::validation::convert_1d_vector_to_2d_view_device(dgncur_awet_db,
                                                          dgncur_awet);

    mam4::validation::convert_1d_vector_to_2d_view_device(qaerwat_db, qaerwat);

    {

      const int top_lev = 1;
      auto team_policy = ThreadTeamPolicy(1, 1u);
      Kokkos::parallel_for(
          team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
            for (int kk = top_lev; kk < nlev; ++kk) {

              int nspec_amode[AeroConfig::num_modes()];
              int lspectype_amode[water_uptake::maxd_aspectype]
                                 [AeroConfig::num_modes()];
              Real specdens_amode[water_uptake::maxd_aspectype];
              Real spechygro[water_uptake::maxd_aspectype];

              water_uptake::get_e3sm_parameters(nspec_amode, lspectype_amode,
                                                specdens_amode, spechygro);

              const auto state_q_kk =
                  Kokkos::subview(state_q, kk, Kokkos::ALL());
              const auto dgnumdry_m_kk =
                  Kokkos::subview(dgncur_a, kk, Kokkos::ALL());
              const auto dgnumwet_m_kk =
                  Kokkos::subview(dgncur_awet, kk, Kokkos::ALL());
              const auto qaerwat_m_kk =
                  Kokkos::subview(qaerwat, kk, Kokkos::ALL());
              mam4::water_uptake::modal_aero_water_uptake_dr(
                  nspec_amode, specdens_amode, spechygro, lspectype_amode,
                  state_q_kk.data(), temperature(kk), pmid(kk), cldn(kk),
                  dgnumdry_m_kk.data(), dgnumwet_m_kk.data(),
                  qaerwat_m_kk.data());

            } // kk
          });
    } // modal_aero_wateruptake_dr

    constexpr Real zero = 0;
    std::vector<Real> dgncur_awet_out(nlev * ntot_amode, zero);
    mam4::validation::convert_2d_view_device_to_1d_vector(dgncur_awet,
                                                          dgncur_awet_out);

    std::vector<Real> qaerwat_out(nlev * ntot_amode, zero);
    mam4::validation::convert_2d_view_device_to_1d_vector(qaerwat, qaerwat_out);

    output.set("dgncur_awet", dgncur_awet_out);
    output.set("qaerwat", qaerwat_out);
  });
}
