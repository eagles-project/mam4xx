// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include "testing.hpp"
#include <mam4xx/aero_config.hpp>
#include <mam4xx/mam4.hpp>

#include <catch2/catch.hpp>
#include <ekat/logging/ekat_logger.hpp>
#include <ekat/mpi/ekat_comm.hpp>

#include <map>
#include <sstream>

using namespace mam4;

TEST_CASE("aero_config", "") {
  ekat::Comm comm;

  ekat::logger::Logger<> logger("nucleation unit tests",
                                ekat::logger::LogLevel::debug, comm);

  SECTION("check init") {
    using std::isnan;

    const int nlev = 72;
    Prognostics progs = testing::create_prognostics(nlev);
    Diagnostics diags = testing::create_diagnostics(nlev);
    Tendencies tends = testing::create_tendencies(nlev);

    typedef typename ColumnView::HostMirror HostColumnView;

    HostColumnView h_progs_num_aer[4];
    HostColumnView h_progs_q_aer_i[4][7];
    HostColumnView h_progs_q_aer_c[4][7];
    HostColumnView h_progs_q_gas[13];
    HostColumnView h_progs_uptkaer[13][4];

    HostColumnView h_tends_num_aer[4];
    HostColumnView h_tends_q_aer_i[4][7];
    HostColumnView h_tends_q_aer_c[4][7];
    HostColumnView h_tends_q_gas[13];
    HostColumnView h_tends_uptkaer[13][4];

    HostColumnView h_diags_wet_diam_i[4];

    HostColumnView h_diags_dry_diam_i[4];
    HostColumnView h_diags_dry_diam_c[4];
    HostColumnView h_diags_dry_diam_total[4];

    logger.info("creating host mirror views");

    for (int m = 0; m < AeroConfig::num_modes(); ++m) {
      logger.debug("mode m = {}", m);
      h_progs_num_aer[m] = Kokkos::create_mirror_view(progs.n_mode_i[m]);
      h_tends_num_aer[m] = Kokkos::create_mirror_view(tends.n_mode_i[m]);
      h_diags_dry_diam_i[m] =
          Kokkos::create_mirror_view(diags.dry_geometric_mean_diameter_i[m]);
      h_diags_dry_diam_c[m] =
          Kokkos::create_mirror_view(diags.dry_geometric_mean_diameter_c[m]);
      h_diags_dry_diam_total[m] = Kokkos::create_mirror_view(
          diags.dry_geometric_mean_diameter_total[m]);

      h_diags_wet_diam_i[m] =
          Kokkos::create_mirror_view(diags.wet_geometric_mean_diameter_i[m]);
      for (int s = 0; s < AeroConfig::num_aerosol_ids(); ++s) {
        logger.debug("[mode, species] = [{}, {}]", m, s);
        h_progs_q_aer_i[m][s] =
            Kokkos::create_mirror_view(progs.q_aero_i[m][s]);
        h_tends_q_aer_i[m][s] =
            Kokkos::create_mirror_view(tends.q_aero_i[m][s]);
        h_progs_q_aer_c[m][s] =
            Kokkos::create_mirror_view(progs.q_aero_c[m][s]);
        h_tends_q_aer_c[m][s] =
            Kokkos::create_mirror_view(tends.q_aero_c[m][s]);
      }
      for (int g = 0; g < AeroConfig::num_gas_ids(); ++g) {
        logger.debug("[mode, gas] = [{}, {}]", m, g);
        h_progs_uptkaer[g][m] = Kokkos::create_mirror_view(progs.uptkaer[g][m]);
        h_tends_uptkaer[g][m] = Kokkos::create_mirror_view(tends.uptkaer[g][m]);
      }
    }
    for (int g = 0; g < AeroConfig::num_gas_ids(); ++g) {
      h_progs_q_gas[g] = Kokkos::create_mirror_view(progs.q_gas[g]);
      h_tends_q_gas[g] = Kokkos::create_mirror_view(tends.q_gas[g]);
    }

    logger.info("deep copying views");
    for (int m = 0; m < AeroConfig::num_modes(); ++m) {
      Kokkos::deep_copy(h_progs_num_aer[m], progs.n_mode_i[m]);
      Kokkos::deep_copy(h_tends_num_aer[m], tends.n_mode_i[m]);

      Kokkos::deep_copy(h_diags_dry_diam_i[m],
                        diags.dry_geometric_mean_diameter_i[m]);
      Kokkos::deep_copy(h_diags_dry_diam_c[m],
                        diags.dry_geometric_mean_diameter_c[m]);
      Kokkos::deep_copy(h_diags_dry_diam_total[m],
                        diags.dry_geometric_mean_diameter_total[m]);

      Kokkos::deep_copy(h_diags_wet_diam_i[m],
                        diags.wet_geometric_mean_diameter_i[m]);
      for (int s = 0; s < AeroConfig::num_aerosol_ids(); ++s) {
        Kokkos::deep_copy(h_progs_q_aer_i[m][s], progs.q_aero_i[m][s]);
        Kokkos::deep_copy(h_tends_q_aer_i[m][s], tends.q_aero_i[m][s]);
        Kokkos::deep_copy(h_progs_q_aer_c[m][s], progs.q_aero_c[m][s]);
        Kokkos::deep_copy(h_tends_q_aer_c[m][s], tends.q_aero_c[m][s]);
      }
      for (int g = 0; g < AeroConfig::num_gas_ids(); ++g) {
        Kokkos::deep_copy(h_progs_uptkaer[g][m], progs.uptkaer[g][m]);
        Kokkos::deep_copy(h_tends_uptkaer[g][m], tends.uptkaer[g][m]);
      }
    }
    for (int g = 0; g < AeroConfig::num_gas_ids(); ++g) {
      Kokkos::deep_copy(h_progs_q_gas[g], progs.q_gas[g]);
      Kokkos::deep_copy(h_tends_q_gas[g], tends.q_gas[g]);
    }

    logger.info("checking that all views are initialized to zero.");
    for (int m = 0; m < AeroConfig::num_modes(); ++m) {
      for (int k = 0; k < nlev; ++k) {
        REQUIRE(h_progs_num_aer[m](k) == 0);
        REQUIRE(h_tends_num_aer[m](k) == 0);
        REQUIRE(h_diags_dry_diam_i[m](k) == 0);
        REQUIRE(h_diags_dry_diam_c[m](k) == 0);
        REQUIRE(h_diags_dry_diam_total[m](k) == 0);
        REQUIRE(h_diags_wet_diam_i[m](k) == 0);
        for (int s = 0; s < AeroConfig::num_aerosol_ids(); ++s) {
          REQUIRE(h_progs_q_aer_i[m][s](k) == 0);
          REQUIRE(h_tends_q_aer_i[m][s](k) == 0);
          REQUIRE(h_progs_q_aer_c[m][s](k) == 0);
          REQUIRE(h_tends_q_aer_c[m][s](k) == 0);
        }
        for (int g = 0; g < AeroConfig::num_gas_ids(); ++g) {
          REQUIRE(h_progs_uptkaer[g][m](k) == 0);
          REQUIRE(h_tends_uptkaer[g][m](k) == 0);
          if (m == 0) {
            REQUIRE(h_progs_q_gas[g](k) == 0);
            REQUIRE(h_tends_q_gas[g](k) == 0);
          }
        }
      }
    }
  }
}
