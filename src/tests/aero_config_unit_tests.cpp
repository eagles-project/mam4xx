#include "aero_config.hpp"
#include "ekat/logging/ekat_logger.hpp"
#include "ekat/mpi/ekat_comm.hpp"
#include "mam4.hpp"
#include <catch2/catch.hpp>
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
    Prognostics progs(nlev);
    Diagnostics diags(nlev);
    Tendencies tends(nlev);

    typedef typename ColumnView::HostMirror HostColumnView;

    HostColumnView h_progs_num_aer[4];
    HostColumnView h_progs_q_aer[4][7];
    HostColumnView h_progs_q_gas[13];
    HostColumnView h_progs_uptkaer[13][4];

    HostColumnView h_tends_num_aer[4];
    HostColumnView h_tends_q_aer[4][7];
    HostColumnView h_tends_q_gas[13];
    HostColumnView h_tends_uptkaer[13][4];

    HostColumnView h_diags_wet_diam[4];
    HostColumnView h_diags_dry_diam[4];

    logger.info("creating host mirror views");

    for (int m = 0; m < 4; ++m) {
      logger.debug("mode m = {}", m);
      h_progs_num_aer[m] = Kokkos::create_mirror_view(progs.n_mode[m]);
      h_tends_num_aer[m] = Kokkos::create_mirror_view(tends.n_mode[m]);
      h_diags_dry_diam[m] =
          Kokkos::create_mirror_view(diags.dry_geometric_mean_diameter[m]);
      h_diags_wet_diam[m] =
          Kokkos::create_mirror_view(diags.wet_geometric_mean_diameter[m]);
      for (int s = 0; s < 7; ++s) {
        logger.debug("[mode, species] = [{}, {}]", m, s);
        h_progs_q_aer[m][s] = Kokkos::create_mirror_view(progs.q_aero[m][s]);
        h_tends_q_aer[m][s] = Kokkos::create_mirror_view(tends.q_aero[m][s]);
      }
      for (int g = 0; g < 13; ++g) {
        logger.debug("[mode, gas] = [{}, {}]", m, g);
        h_progs_uptkaer[g][m] = Kokkos::create_mirror_view(progs.uptkaer[g][m]);
        h_tends_uptkaer[g][m] = Kokkos::create_mirror_view(tends.uptkaer[g][m]);
      }
    }
    for (int g = 0; g < 13; ++g) {
      h_progs_q_gas[g] = Kokkos::create_mirror_view(progs.q_gas[g]);
      h_tends_q_gas[g] = Kokkos::create_mirror_view(tends.q_gas[g]);
    }

    logger.info("deep copying views");
    for (int m = 0; m < 4; ++m) {
      Kokkos::deep_copy(h_progs_num_aer[m], progs.n_mode[m]);
      Kokkos::deep_copy(h_tends_num_aer[m], tends.n_mode[m]);
      Kokkos::deep_copy(h_diags_dry_diam[m],
                        diags.dry_geometric_mean_diameter[m]);
      Kokkos::deep_copy(h_diags_wet_diam[m],
                        diags.wet_geometric_mean_diameter[m]);
      for (int s = 0; s < 7; ++s) {
        Kokkos::deep_copy(h_progs_q_aer[m][s], progs.q_aero[m][s]);
        Kokkos::deep_copy(h_tends_q_aer[m][s], tends.q_aero[m][s]);
      }
      for (int g = 0; g < 13; ++g) {
        Kokkos::deep_copy(h_progs_uptkaer[g][m], progs.uptkaer[g][m]);
        Kokkos::deep_copy(h_tends_uptkaer[g][m], tends.uptkaer[g][m]);
      }
    }
    for (int g = 0; g < 13; ++g) {
      Kokkos::deep_copy(h_progs_q_gas[g], progs.q_gas[g]);
      Kokkos::deep_copy(h_tends_q_gas[g], tends.q_gas[g]);
    }

    logger.info("checking that all views are initialized to zero.");
    for (int m = 0; m < 4; ++m) {
      for (int k = 0; k < nlev; ++k) {
        const int p = PackInfo::pack_idx(k);
        const int v = PackInfo::vec_idx(k);
        REQUIRE(h_progs_num_aer[m](p)[v] == 0);
        REQUIRE(h_tends_num_aer[m](p)[v] == 0);
        REQUIRE(h_diags_dry_diam[m](p)[v] == 0);
        REQUIRE(h_diags_wet_diam[m](p)[v] == 0);
        for (int s = 0; s < 7; ++s) {
          REQUIRE(h_progs_q_aer[m][s](p)[v] == 0);
          REQUIRE(h_tends_q_aer[m][s](p)[v] == 0);
        }
        for (int g = 0; g < 13; ++g) {
          REQUIRE(h_progs_uptkaer[g][m](p)[v] == 0);
          REQUIRE(h_tends_uptkaer[g][m](p)[v] == 0);
          if (m == 0) {
            REQUIRE(h_progs_q_gas[g](p)[v] == 0);
            REQUIRE(h_tends_q_gas[g](p)[v] == 0);
          }
        }
      }
    }
  }
}
