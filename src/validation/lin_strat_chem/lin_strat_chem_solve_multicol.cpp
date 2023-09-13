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
using namespace lin_strat_chem;
void lin_strat_chem_solve_multicol(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    const auto o3col_db = input.get_array("o3col");
    const auto temperature_db = input.get_array("temp");
    const auto pmid_db = input.get_array("pmid");
    using View1D = typename DeviceType::view_1d<Real>;
    using View2D = typename DeviceType::view_2d<Real>;
    using View1DHost = typename HostType::view_1d<Real>;

    constexpr int ncol = 4;
    constexpr int pver = mam4::nlev;

    View2D o3col("o3col", ncol, pver);
    View2D temperature("temperature", ncol, pver);
    View2D pmid("pmid", ncol, pver);
    mam4::validation::convert_1d_vector_to_2d_view_device(o3col_db, o3col);
    mam4::validation::convert_1d_vector_to_2d_view_device(temperature_db,
                                                          temperature);
    mam4::validation::convert_1d_vector_to_2d_view_device(pmid_db, pmid);

    const auto linoz_o3_clim_db = input.get_array("linoz_o3_clim");
    const auto linoz_t_clim_db = input.get_array("linoz_t_clim");
    const auto linoz_o3col_clim_db = input.get_array("linoz_o3col_clim");
    const auto linoz_PmL_clim_db = input.get_array("linoz_PmL_clim");
    const auto linoz_dPmL_dO3_db = input.get_array("linoz_dPmL_dO3");
    const auto linoz_dPmL_dO3col_db = input.get_array("linoz_dPmL_dO3col");
    const auto linoz_cariolle_psc_db = input.get_array("linoz_cariolle_psc");
    const auto linoz_dPmL_dT_db = input.get_array("linoz_dPmL_dT");
    const auto o3_vmr_db = input.get_array("o3_vmr");

    View2D linoz_o3_clim("linoz_o3_clim", ncol, pver);
    View2D linoz_t_clim("linoz_t_clim", ncol, pver);
    View2D linoz_o3col_clim("linoz_o3col_clim", ncol, pver);
    View2D linoz_PmL_clim("inoz_PmL_clim", ncol, pver);
    View2D linoz_dPmL_dO3("linoz_dPmL_dO3", ncol, pver);
    View2D linoz_dPmL_dO3col("linoz_dPmL_dO3col", ncol, pver);
    View2D linoz_cariolle_psc("linoz_cariolle_psc", ncol, pver);
    View2D linoz_dPmL_dT("linoz_dPmL_dT", ncol, pver);
    View2D o3_vmr("o3_vmr", ncol, pver);

    mam4::validation::convert_1d_vector_to_2d_view_device(linoz_o3_clim_db,
                                                          linoz_o3_clim);
    mam4::validation::convert_1d_vector_to_2d_view_device(linoz_t_clim_db,
                                                          linoz_t_clim);
    mam4::validation::convert_1d_vector_to_2d_view_device(linoz_o3col_clim_db,
                                                          linoz_o3col_clim);
    mam4::validation::convert_1d_vector_to_2d_view_device(linoz_PmL_clim_db,
                                                          linoz_PmL_clim);
    mam4::validation::convert_1d_vector_to_2d_view_device(linoz_dPmL_dO3_db,
                                                          linoz_dPmL_dO3);
    mam4::validation::convert_1d_vector_to_2d_view_device(linoz_dPmL_dO3col_db,
                                                          linoz_dPmL_dO3col);
    mam4::validation::convert_1d_vector_to_2d_view_device(linoz_cariolle_psc_db,
                                                          linoz_cariolle_psc);
    mam4::validation::convert_1d_vector_to_2d_view_device(linoz_dPmL_dT_db,
                                                          linoz_dPmL_dT);
    mam4::validation::convert_1d_vector_to_2d_view_device(o3_vmr_db, o3_vmr);

    const auto sza_db = input.get_array("sza");
    auto sza_host = View1DHost((Real *)sza_db.data(), ncol);
    const auto sza = View1D("sza", ncol);
    Kokkos::deep_copy(sza, sza_host);

    const Real delta_t = input.get_array("delta_t")[0];
    const auto rlats_db = input.get_array("rlats");
    const Real chlorine_loading = input.get_array("chlorine_loading")[0];
    const Real psc_T = input.get_array("psc_T")[0];

    View2D do3_linoz("do3_linoz", ncol, pver);
    View2D do3_linoz_psc("do3_linoz_psc", ncol, pver);
    View2D ss_o3("ss_o3", ncol, pver);
    View2D o3col_du_diag("o3col_du_diag", ncol, pver);
    View2D o3clim_linoz_diag("o3clim_linoz_diag", ncol, pver);
    View2D sza_degrees("sza_degrees", ncol, pver);

    const auto ltrop_db = input.get_array("ltrop");

    int ltrop[ncol] = {};
    Real rlats[ncol] = {};

    for (int i = 0; i < ncol; ++i) {
      ltrop[i] = int(ltrop_db[i]);
      rlats[i] = rlats_db[i];
    }

    auto team_policy = ThreadTeamPolicy(ncol, 1u);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
          const int icol = team.league_rank();
          auto o3col_icol = Kokkos::subview(o3col, icol, Kokkos::ALL());
          auto temperature_icol =
              Kokkos::subview(temperature, icol, Kokkos::ALL());
          auto pmid_icol = Kokkos::subview(pmid, icol, Kokkos::ALL());
          auto linoz_o3_clim_icol =
              Kokkos::subview(linoz_o3_clim, icol, Kokkos::ALL());
          auto linoz_t_clim_icol =
              Kokkos::subview(linoz_t_clim, icol, Kokkos::ALL());
          auto linoz_o3col_clim_icol =
              Kokkos::subview(linoz_o3col_clim, icol, Kokkos::ALL());
          auto linoz_PmL_clim_icol =
              Kokkos::subview(linoz_PmL_clim, icol, Kokkos::ALL());
          auto linoz_dPmL_dO3_icol =
              Kokkos::subview(linoz_dPmL_dO3, icol, Kokkos::ALL());
          auto linoz_dPmL_dO3col_icol =
              Kokkos::subview(linoz_dPmL_dO3col, icol, Kokkos::ALL());
          auto linoz_dPmL_dT_icol =
              Kokkos::subview(linoz_dPmL_dT, icol, Kokkos::ALL());
          auto o3_vmr_icol = Kokkos::subview(o3_vmr, icol, Kokkos::ALL());
          auto do3_linoz_icol = Kokkos::subview(do3_linoz, icol, Kokkos::ALL());
          auto do3_linoz_psc_icol =
              Kokkos::subview(do3_linoz_psc, icol, Kokkos::ALL());
          auto ss_o3_icol = Kokkos::subview(ss_o3, icol, Kokkos::ALL());
          auto o3col_du_diag_icol =
              Kokkos::subview(o3col_du_diag, icol, Kokkos::ALL());
          auto o3clim_linoz_diag_icol =
              Kokkos::subview(o3clim_linoz_diag, icol, Kokkos::ALL());
          auto sza_degrees_icol =
              Kokkos::subview(sza_degrees, icol, Kokkos::ALL());
          auto linoz_cariolle_psc_icol =
              Kokkos::subview(linoz_cariolle_psc, icol, Kokkos::ALL());

          lin_strat_chem_solve(
              team, o3col_icol, temperature_icol, sza(icol), pmid_icol, delta_t,
              rlats[icol],
              // ltrop, & !in
              linoz_o3_clim_icol, linoz_t_clim_icol, linoz_o3col_clim_icol,
              linoz_PmL_clim_icol, linoz_dPmL_dO3_icol,
              linoz_dPmL_dT_icol, // in
              linoz_dPmL_dO3col_icol,
              linoz_cariolle_psc_icol, // in
              ltrop[icol], chlorine_loading,
              psc_T, // PSC ozone loss T (K) threshold
              o3_vmr_icol,
              // diagnostic variables outputs
              do3_linoz_icol, do3_linoz_psc_icol, ss_o3_icol,
              o3col_du_diag_icol, o3clim_linoz_diag_icol, sza_degrees_icol);
        });

    constexpr Real zero = 0;
    std::vector<Real> o3_vmr_out(pver * ncol, zero);
    mam4::validation::convert_2d_view_device_to_1d_vector(o3_vmr, o3_vmr_out);
    std::vector<Real> do3_linoz_out(pver * ncol, zero);
    mam4::validation::convert_2d_view_device_to_1d_vector(do3_linoz,
                                                          do3_linoz_out);
    std::vector<Real> do3_linoz_psc_out(pver * ncol, zero);
    mam4::validation::convert_2d_view_device_to_1d_vector(do3_linoz_psc,
                                                          do3_linoz_psc_out);
    std::vector<Real> ss_o3_out(pver * ncol, zero);
    mam4::validation::convert_2d_view_device_to_1d_vector(ss_o3, ss_o3_out);
    std::vector<Real> o3col_du_diag_out(pver * ncol, zero);
    mam4::validation::convert_2d_view_device_to_1d_vector(o3col_du_diag,
                                                          o3col_du_diag_out);
    std::vector<Real> o3clim_linoz_diag_out(pver * ncol, zero);
    mam4::validation::convert_2d_view_device_to_1d_vector(
        o3clim_linoz_diag, o3clim_linoz_diag_out);
    output.set("o3_vmr", o3_vmr_out);
    output.set("do3_linoz", do3_linoz_out);
    output.set("do3_linoz_psc", do3_linoz_psc_out);
    output.set("ss_o3", ss_o3_out);
    output.set("o3col_du_diag", o3col_du_diag_out);
    output.set("o3clim_linoz_diag", o3clim_linoz_diag_out);
    // output.set("sza_degrees", sza_degrees);
  });
}
