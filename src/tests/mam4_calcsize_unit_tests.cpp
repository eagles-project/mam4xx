// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include "testing.hpp"
#include <mam4xx/mam4.hpp>

#include <catch2/catch.hpp>
#include <ekat/ekat_pack_kokkos.hpp>
#include <ekat/logging/ekat_logger.hpp>
#include <ekat/mpi/ekat_comm.hpp>

// if you need something from the data/ directory
// std::string data_file = MAM4_TEST_DATA_DIR;
// #include <mam4_test_config.hpp>

using namespace haero;

TEST_CASE("test_constructor", "mam4_calcsize_process") {
  mam4::AeroConfig mam4_config;
  mam4::CalcSizeProcess process(mam4_config);
  REQUIRE(process.name() == "MAM4 calcsize");
  REQUIRE(process.aero_config() == mam4_config);
}

TEST_CASE("test_compute_tendencies", "mam4_calcsize_process") {
  ekat::Comm comm;

  ekat::logger::Logger<> logger("calcsize unit tests",
                                ekat::logger::LogLevel::debug, comm);

  int nlev = 1;
  Real pblh = 1000;
  Atmosphere atm = mam4::testing::create_atmosphere(nlev, pblh);
  mam4::Prognostics progs = mam4::testing::create_prognostics(nlev);
  mam4::Diagnostics diags = mam4::testing::create_diagnostics(nlev);
  mam4::Tendencies tends = mam4::testing::create_tendencies(nlev);

  mam4::AeroConfig mam4_config;
  mam4::CalcSizeProcess process(mam4_config);

  const auto nmodes = mam4::AeroConfig::num_modes();

  Kokkos::Array<Real, 21> interstitial = {
      0.1218350564E-08, 0.3560443333E-08, 0.4203338951E-08, 0.3723412167E-09,
      0.2330196615E-09, 0.1435909119E-10, 0.2704344376E-11, 0.2116400132E-11,
      0.1326256343E-10, 0.1741610336E-16, 0.1280539377E-16, 0.1045693148E-08,
      0.5358722850E-10, 0.2926142847E-10, 0.3986256848E-11, 0.3751267639E-10,
      0.4679337373E-10, 0.9456518445E-13, 0.2272248527E-08, 0.2351792086E-09,
      0.2733926271E-16};

  Kokkos::Array<Real, nmodes> interstitial_num = {
      0.8098354597E+09, 0.4425427527E+08, 0.1400840545E+06, 0.1382391601E+10};

  std::ostringstream ss;
  int count = 0;
  for (int imode = 0; imode < nmodes; ++imode) {
    auto h_prog_n_mode_i = Kokkos::create_mirror_view(progs.n_mode_i[imode]);

    for (int k = 0; k < nlev; ++k) {
      // set all cell to same value.
      h_prog_n_mode_i(k) = interstitial_num[imode];
    }
    Kokkos::deep_copy(progs.n_mode_i[imode], h_prog_n_mode_i);

    ss << "progs.n_mode_i (mode No " << imode << ") [in]: [ ";
    for (int k = 0; k < nlev; ++k) {
      ss << h_prog_n_mode_i(k) << " ";
    }
    ss << "]";
    logger.debug(ss.str());
    ss.str("");

    for (int k = 0; k < nlev; ++k) {
      CHECK(!isnan(h_prog_n_mode_i(k)));
    }

    const auto n_spec = mam4::num_species_mode(imode);
    for (int isp = 0; isp < n_spec; ++isp) {
      auto h_prog_aero_i =
          Kokkos::create_mirror_view(progs.q_aero_i[imode][isp]);
      for (int k = 0; k < nlev; ++k) {
        // set all cell to same value.
        h_prog_aero_i(k) = interstitial[count];
      }
      Kokkos::deep_copy(progs.q_aero_i[imode][isp], h_prog_aero_i);
      count++;

      ss << "progs.q_aero_i (mode No " << imode << ", species No " << isp
         << " ) [in]: [ ";
      for (int k = 0; k < nlev; ++k) {
        ss << h_prog_aero_i(k) << " ";
      }
      ss << "]";
      logger.debug(ss.str());
      ss.str("");

      for (int k = 0; k < nlev; ++k) {
        CHECK(!isnan(h_prog_aero_i(k)));
      }

    } // end species
  }   // end modes

  const int ncol = 1;
  // Single-column dispatch.
  auto team_policy = ThreadTeamPolicy(ncol, Kokkos::AUTO);
  Real t = 0.0, dt = 30.0;
  Kokkos::parallel_for(
      team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
        process.compute_tendencies(team, t, dt, atm, progs, diags, tends);
      });

  for (int imode = 0; imode < nmodes; ++imode) {
    auto h_tends_n_mode_i = Kokkos::create_mirror_view(tends.n_mode_i[imode]);
    Kokkos::deep_copy(h_tends_n_mode_i, tends.n_mode_i[imode]);

    ss << "tends.n_mode_i (mode No " << imode << ") [out]: [ ";
    for (int k = 0; k < nlev; ++k) {
      ss << h_tends_n_mode_i(k) << " ";
    }
    ss << "]";
    logger.debug(ss.str());
    ss.str("");

    for (int k = 0; k < nlev; ++k) {
      CHECK(!isnan(h_tends_n_mode_i(k)));
    }

    const auto n_spec = mam4::num_species_mode(imode);
    for (int isp = 0; isp < n_spec; ++isp) {
      // const auto prog_aero_i = ekat::scalarize(tends.q_aero_i[imode][i]);
      auto h_tends_aero_i =
          Kokkos::create_mirror_view(tends.q_aero_i[imode][isp]);
      Kokkos::deep_copy(h_tends_aero_i, tends.q_aero_i[imode][isp]);

      ss << "tends.q_aero_i (mode No " << imode << ", species No " << isp
         << " ) [out]: [ ";
      for (int k = 0; k < nlev; ++k) {
        ss << h_tends_aero_i(k) << " ";
      }
      ss << "]";
      logger.debug(ss.str());
      ss.str("");

      for (int k = 0; k < nlev; ++k) {
        CHECK(!isnan(h_tends_aero_i(k)));
      }

    } // end species
  }   // end modes
}
