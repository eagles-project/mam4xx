// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/mam4.hpp>

#include <mam4xx/aero_config.hpp>
#include <skywalker.hpp>
#include <validation.hpp>

using namespace skywalker;
using namespace mam4;
using namespace haero;
using namespace modal_aer_opt;

void volcanic_cmip_sw(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    using View1DHost = typename HostType::view_1d<Real>;

    constexpr Real zero = 0;
    const auto zi_db = input.get_array("zi");
    ColumnView zi;
    zi = haero::testing::create_column_view(pver);
    auto zi_host = View1DHost((Real *)zi_db.data(), pver);
    Kokkos::deep_copy(zi, zi_host);
    printf("zi size %lu \n", zi.size());

    // Fortran to C++ indexing
    const int ilev_tropp = int(input.get_array("trop_level")[0]) - 1;

    // Note: We do not need to convert units for ext_cmip6_sw_inv_m
    const auto ext_cmip6_sw_db = input.get_array("ext_cmip6_sw_inv_m");
    // We need to reshape ext_cmip6_sw
    View2D ext_cmip6_sw("ext_cmip6_sw", nswbands, pver);
    auto ext_cmip6_sw_host = Kokkos::create_mirror_view(ext_cmip6_sw);
    int count = 0;
    for (int d1 = 0; d1 < nswbands; ++d1) {
      for (int d2 = 0; d2 < pver; ++d2) {
        ext_cmip6_sw_host(d1, d2) = ext_cmip6_sw_db[count];
        count++;
      }
    }
    Kokkos::deep_copy(ext_cmip6_sw, ext_cmip6_sw_host);
    const auto ssa_cmip6_sw_db = input.get_array("ssa_cmip6_sw");

    View2D ssa_cmip6_sw("ssa_cmip6_sw", pver, nswbands);
    mam4::validation::convert_1d_vector_to_2d_view_device(ssa_cmip6_sw_db,
                                                          ssa_cmip6_sw);

    const auto af_cmip6_sw_db = input.get_array("af_cmip6_sw");
    View2D af_cmip6_sw("af_cmip6_sw", pver, nswbands);
    mam4::validation::convert_1d_vector_to_2d_view_device(af_cmip6_sw_db,
                                                          af_cmip6_sw);

    View2D tau, tau_w, tau_w_g, tau_w_f;

    tau =
        View2D("tau", nswbands, pver + 1); // layer extinction optical depth [1]
    tau_w =
        View2D("tau_w", nswbands, pver + 1); // layer single-scatter albedo [1]
    tau_w_g = View2D("tau_w_g", nswbands, pver + 1); // asymmetry factor [1]
    tau_w_f =
        View2D("tau_w_f", nswbands, pver + 1); // forward scattered fraction [1]

    const auto tau_db = input.get_array("tau");
    const auto tau_w_db = input.get_array("tau_w");
    const auto tau_w_g_db = input.get_array("tau_w_g");
    const auto tau_w_f_db = input.get_array("tau_w_f");

    mam4::validation::convert_1d_vector_to_transpose_2d_view_device(tau_db,
                                                                    tau);
    mam4::validation::convert_1d_vector_to_transpose_2d_view_device(tau_w_db,
                                                                    tau_w);
    mam4::validation::convert_1d_vector_to_transpose_2d_view_device(tau_w_g_db,
                                                                    tau_w_g);
    mam4::validation::convert_1d_vector_to_transpose_2d_view_device(tau_w_f_db,
                                                                    tau_w_f);

    auto team_policy = ThreadTeamPolicy(1u, Kokkos::AUTO);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
          aer_rad_props::volcanic_cmip_sw(team, zi, ilev_tropp, ext_cmip6_sw,
                                          ssa_cmip6_sw, af_cmip6_sw, tau, tau_w,
                                          tau_w_g, tau_w_f);
        });

    const int pver_po = pver + 1;

    std::vector<Real> tau_out(pver_po * nswbands, zero);
    mam4::validation::convert_transpose_2d_view_device_to_1d_vector(tau,
                                                                    tau_out);
    output.set("tau", tau_out);

    std::vector<Real> tau_w_out(pver_po * nswbands, zero);
    mam4::validation::convert_transpose_2d_view_device_to_1d_vector(tau_w,
                                                                    tau_w_out);
    output.set("tau_w", tau_w_out);

    std::vector<Real> tau_w_g_out(pver_po * nswbands, zero);
    mam4::validation::convert_transpose_2d_view_device_to_1d_vector(
        tau_w_g, tau_w_g_out);
    output.set("tau_w_g", tau_w_g_out);

    std::vector<Real> tau_w_f_out(pver_po * nswbands, zero);
    mam4::validation::convert_transpose_2d_view_device_to_1d_vector(
        tau_w_f, tau_w_f_out);
    output.set("tau_w_f", tau_w_f_out);
  });
}
