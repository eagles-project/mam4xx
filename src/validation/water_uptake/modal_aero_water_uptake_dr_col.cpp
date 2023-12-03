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

    constexpr int nvars = ndrop::nvars;
    constexpr int pver = ndrop::pver;
    constexpr int ntot_amode = AeroConfig::num_modes();

    View2D state_q("state_q", pver, nvars);
    mam4::validation::convert_1d_vector_to_2d_view_device(state_q_db, state_q);

    ColumnView temperature;
    temperature = haero::testing::create_column_view(pver);
    auto temperature_host = View1DHost((Real *)temperature_db.data(), pver);

    ColumnView pmid;
    pmid = haero::testing::create_column_view(pver);
    auto pmid_host = View1DHost((Real *)pmid_db.data(), pver);

    ColumnView cldn;
    cldn = haero::testing::create_column_view(pver);
    auto cldn_host = View1DHost((Real *)cldn_db.data(), pver);

    Kokkos::deep_copy(temperature, temperature_host);
    Kokkos::deep_copy(pmid, pmid_host);
    Kokkos::deep_copy(cldn, cldn_host);

    View2D dgncur_a("dgncur_a_db", pver, ntot_amode);
    View2D dgncur_awet("dgncur_awet", pver, ntot_amode);
    View2D qaerwat("qaerwat", pver, ntot_amode);
    mam4::validation::convert_1d_vector_to_2d_view_device(dgncur_a_db,
                                                          dgncur_a);

    mam4::validation::convert_1d_vector_to_2d_view_device(dgncur_awet_db,
                                                          dgncur_awet);

    mam4::validation::convert_1d_vector_to_2d_view_device(qaerwat_db, qaerwat);

    int nspec_amode[AeroConfig::num_modes()];
    int lspectype_amode[water_uptake::maxd_aspectype][AeroConfig::num_modes()];
    Real specdens_amode[water_uptake::maxd_aspectype];
    Real spechygro[water_uptake::maxd_aspectype];

    water_uptake::get_e3sm_parameters(nspec_amode, lspectype_amode,
                                      specdens_amode, spechygro);

    modal_aer_opt::modal_aero_wateruptake_dr(
        state_q, temperature, pmid, cldn, dgncur_a, dgncur_awet, qaerwat,
        // const int list_idx_in,
        nspec_amode, specdens_amode, spechygro, lspectype_amode);

    constexpr Real zero = 0;
    std::vector<Real> dgncur_awet_out(pver * ntot_amode, zero);
    mam4::validation::convert_2d_view_device_to_1d_vector(dgncur_awet,
                                                          dgncur_awet_out);

    std::vector<Real> qaerwat_out(pver * ntot_amode, zero);
    mam4::validation::convert_2d_view_device_to_1d_vector(qaerwat, qaerwat_out);

    output.set("dgncur_awet", dgncur_awet_out);
    output.set("qaerwat", qaerwat_out);
  });
}
