// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include "testing.hpp"
#include <mam4xx/mam4.hpp>

#include <ekat/ekat_type_traits.hpp>
#include <ekat/logging/ekat_logger.hpp>
#include <ekat/mpi/ekat_comm.hpp>

#include <catch2/catch.hpp>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>

using namespace haero;
using namespace mam4;

TEST_CASE("test_constructor", "mam4_aging_process") {
  mam4::AeroConfig mam4_config;
  mam4::AgingProcess process(mam4_config);
  REQUIRE(process.name() == "MAM4 aging");
  REQUIRE(process.aero_config() == mam4_config);
}

TEST_CASE("test_aging_pairs", "mam4_aging_pairs") {
  // mam4 aging assumes that max_agepair is 1
  mam4::AeroConfig mam4_config;
  REQUIRE(mam4_config.max_agepair() == 1);
}

TEST_CASE("test_compute_tendencies", "mam4_aging_process") {
  ekat::Comm comm;

  ekat::logger::Logger<> logger("aging unit tests",
                                ekat::logger::LogLevel::debug, comm);
  int nlev = 72;
  Real pblh = 1000;
  Atmosphere atm = mam4::testing::create_atmosphere(nlev, pblh);
  Surface sfc = mam4::testing::create_surface();
  mam4::Prognostics progs = mam4::testing::create_prognostics(nlev);
  mam4::Diagnostics diags = mam4::testing::create_diagnostics(nlev);
  mam4::Tendencies tends = mam4::testing::create_tendencies(nlev);

  {
    Kokkos::Array<Real, 21> interstitial = {
        0.1218350564E-08, 0.3560443333E-08, 0.4203338951E-08, 0.3723412167E-09,
        0.2330196615E-09, 0.1435909119E-10, 0.2704344376E-11, 0.2116400132E-11,
        0.1326256343E-10, 0.1741610336E-16, 0.1280539377E-16, 0.1045693148E-08,
        0.5358722850E-10, 0.2926142847E-10, 0.3986256848E-11, 0.3751267639E-10,
        0.4679337373E-10, 0.9456518445E-13, 0.2272248527E-08, 0.2351792086E-09,
        0.2733926271E-16};
    const auto nmodes = mam4::AeroConfig::num_modes();
    for (int imode = 0, count = 0; imode < nmodes; ++imode) {
      const auto n_spec = mam4::num_species_mode(imode);
      for (int isp = 0; isp < n_spec; ++isp, ++count) {
        auto h_prog_aero_i =
            Kokkos::create_mirror_view(progs.q_aero_i[imode][isp]);
        for (int k = 0; k < nlev; ++k)
          h_prog_aero_i(k) = interstitial[count];
        Kokkos::deep_copy(progs.q_aero_i[imode][isp], h_prog_aero_i);
      }
    }
  }
  mam4::AeroConfig mam4_config;
  mam4::AgingProcess process(mam4_config);

  const auto prog_qgas0 = progs.q_gas[0];
  const auto tend_qgas0 = tends.q_gas[0];
  auto h_prog_qgas0 = Kokkos::create_mirror_view(prog_qgas0);
  auto h_tend_qgas0 = Kokkos::create_mirror_view(tend_qgas0);
  Kokkos::deep_copy(h_prog_qgas0, prog_qgas0);
  Kokkos::deep_copy(h_tend_qgas0, tend_qgas0);

  std::ostringstream ss;
  ss << "prog_qgas0 [in]: [ ";
  for (int k = 0; k < nlev; ++k) {
    ss << h_prog_qgas0(k) << " ";
  }
  ss << "]";
  logger.debug(ss.str());
  ss.str("");
  ss << "tend_qgas0 [in]: [ ";
  for (int k = 0; k < nlev; ++k) {
    ss << h_tend_qgas0(k) << " ";
  }
  ss << "]";
  logger.debug(ss.str());
  ss.str("");

  for (int k = 0; k < nlev; ++k) {
    CHECK(!isnan(h_prog_qgas0(k)));
    CHECK(!isnan(h_tend_qgas0(k)));
  }

  // Single-column dispatch.
  auto team_policy = ThreadTeamPolicy(1u, Kokkos::AUTO);
  Real t = 0.0, dt = 30.0;
  Kokkos::parallel_for(
      team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
        process.compute_tendencies(team, t, dt, atm, sfc, progs, diags, tends);
      });
  Kokkos::deep_copy(h_prog_qgas0, prog_qgas0);
  Kokkos::deep_copy(h_tend_qgas0, tend_qgas0);

  ss << "prog_qgas0 [out]: [ ";
  for (int k = 0; k < nlev; ++k) {
    ss << h_prog_qgas0(k) << " ";
  }
  ss << "]";
  logger.debug(ss.str());
  ss.str("");
  ss << "tend_qgas0 [out]: [ ";
  for (int k = 0; k < nlev; ++k) {
    ss << h_tend_qgas0(k) << " ";
  }
  ss << "]";
  logger.debug(ss.str());
  ss.str("");

  for (int k = 0; k < nlev; ++k) {
    CHECK(!isnan(h_prog_qgas0(k)));
    CHECK(!isnan(h_tend_qgas0(k)));
  }
}

TEST_CASE("test_cond_coag_mass_to_accum", "mam4_aging_process") {
  const int nsrc = static_cast<int>(ModeIndex::PrimaryCarbon);
  const int ndest = static_cast<int>(ModeIndex::Accumulation);

  std::vector<Real> qaer_cur(AeroConfig::num_modes(), 0.0);
  std::vector<Real> qaer_del_cond(AeroConfig::num_modes(), 0.0);
  std::vector<Real> qaer_del_coag(AeroConfig::num_modes(), 0.0);

  aging::transfer_cond_coag_mass_to_accum(
      nsrc, ndest, qaer_cur.data(), qaer_del_cond.data(), qaer_del_coag.data());

  for (int m = 0; m < AeroConfig::num_modes(); ++m) {
    REQUIRE(qaer_cur[m] == 0.0);
    REQUIRE(qaer_del_cond[m] == 0.0);
    REQUIRE(qaer_del_coag[m] == 0.0);
  }

  qaer_cur[nsrc] = 1.0;
  qaer_del_cond[nsrc] = 1.0;
  qaer_del_coag[nsrc] = 1.0;
  aging::transfer_cond_coag_mass_to_accum(
      nsrc, ndest, qaer_cur.data(), qaer_del_cond.data(), qaer_del_coag.data());

  Real sum_for_conservation = 0.0;
  for (int imode = 0; imode < AeroConfig::num_modes(); ++imode) {
    if (imode == nsrc) {
      REQUIRE(qaer_cur[imode] == 0.0);
      REQUIRE(qaer_del_cond[imode] == 0.0);
      REQUIRE(qaer_del_coag[imode] == 0.0);
    } else if (imode == ndest) {
      REQUIRE(qaer_cur[imode] == 1.0);
      REQUIRE(qaer_del_cond[imode] == 1.0);
      REQUIRE(qaer_del_coag[imode] == 1.0);
    } else {
      REQUIRE(qaer_cur[imode] == 0.0);
      REQUIRE(qaer_del_coag[imode] == 0.0);
      REQUIRE(qaer_del_cond[imode] == 0.0);
    }
    sum_for_conservation +=
        qaer_cur[imode] + qaer_del_cond[imode] + qaer_del_coag[imode];
  }

  // Check for conservation
  REQUIRE(sum_for_conservation == 3.0);
}

TEST_CASE("transfer_aged_pcarbon_to_accum", "mam4_aging_process") {
  const int nsrc = static_cast<int>(ModeIndex::PrimaryCarbon);
  const int ndest = static_cast<int>(ModeIndex::Accumulation);

  Real xferfrac_pcage = 0.5;
  Real frac_cond = 0.25;
  Real frac_coag = 0.75;

  std::vector<Real> qaer_cur(AeroConfig::num_modes(), 0.0);
  std::vector<Real> qaer_del_cond(AeroConfig::num_modes(), 0.0);
  std::vector<Real> qaer_del_coag(AeroConfig::num_modes(), 0.0);

  qaer_cur[nsrc] = 0.0;

  aging::transfer_aged_pcarbon_to_accum(
      nsrc, ndest, xferfrac_pcage, frac_cond, frac_coag, qaer_cur.data(),
      qaer_del_cond.data(), qaer_del_coag.data());

  for (int imode = 0; imode < AeroConfig::num_modes(); ++imode) {
    REQUIRE(qaer_cur[imode] == 0.0);
    REQUIRE(qaer_del_cond[imode] == 0.0);
    REQUIRE(qaer_del_coag[imode] == 0.0);
  }

  qaer_cur[nsrc] = 1.0;
  aging::transfer_aged_pcarbon_to_accum(
      nsrc, ndest, xferfrac_pcage, frac_cond, frac_coag, qaer_cur.data(),
      qaer_del_cond.data(), qaer_del_coag.data());

  Real sum_for_conservation = 0.0;
  for (int imode = 0; imode < AeroConfig::num_modes(); ++imode) {
    if (imode == nsrc) {
      REQUIRE(qaer_cur[imode] == 0.5);
    } else if (imode == ndest) {
      REQUIRE(qaer_cur[imode] == 0.5);
      REQUIRE(qaer_del_cond[imode] == 0.125);
      REQUIRE(qaer_del_coag[imode] == 0.375);
    } else {
      REQUIRE(qaer_cur[imode] == 0.0);
      REQUIRE(qaer_del_coag[imode] == 0.0);
      REQUIRE(qaer_del_cond[imode] == 0.0);
    }
    sum_for_conservation +=
        qaer_cur[imode] + qaer_del_cond[imode] + qaer_del_coag[imode];
  }

  // Check for conservation
  REQUIRE(sum_for_conservation == 1.0);
}

TEST_CASE("mam4_pcarbon_aging_1subarea", "mam4_aging_process") {

  Real dgn_a[AeroConfig::num_modes()] = {};
  Real qnum_cur[AeroConfig::num_modes()] = {};
  Real qnum_del_cond[AeroConfig::num_modes()] = {};
  Real qnum_del_coag[AeroConfig::num_modes()] = {};
  Real qaer_cur[AeroConfig::num_aerosol_ids()][AeroConfig::num_modes()] = {};
  Real qaer_del_cond[AeroConfig::num_aerosol_ids()][AeroConfig::num_modes()] =
      {};
  Real qaer_del_coag[AeroConfig::num_aerosol_ids()][AeroConfig::num_modes()] =
      {};
  Real qaer_del_coag_in[AeroConfig::num_aerosol_ids()]
                       [AeroConfig::max_agepair()] = {};

  // Fill all arrays with zeros
  for (int imode = 0; imode < AeroConfig::num_modes(); ++imode) {
    dgn_a[imode] = 0.0;
    qnum_cur[imode] = 0.0;
    qnum_del_cond[imode] = 0.0;
    qnum_del_coag[imode] = 0.0;
  }

  for (int ispec = 0; ispec < AeroConfig::num_aerosol_ids(); ++ispec) {
    for (int imode = 0; imode < AeroConfig::num_modes(); ++imode) {
      qaer_cur[ispec][imode] = 0.0;
      qaer_del_cond[ispec][imode] = 0.0;
      qaer_del_coag[ispec][imode] = 0.0;
    }
  }

  for (int ispec = 0; ispec < AeroConfig::num_aerosol_ids(); ++ispec) {
    for (int imode = 0; imode < AeroConfig::max_agepair(); ++imode) {
      qaer_del_coag_in[ispec][imode] = 0.0;
    }
  }
  const Real n_so4_monolayers_pcage = 8;
  aging::mam_pcarbon_aging_1subarea(
      n_so4_monolayers_pcage, dgn_a, qnum_cur, qnum_del_cond, qnum_del_coag,
      qaer_cur, qaer_del_cond, qaer_del_coag, qaer_del_coag_in);

  // Passing in zeros for everything should give zeros back
  for (int imode = 0; imode < AeroConfig::num_modes(); ++imode) {
    REQUIRE(dgn_a[imode] == 0.0);
    REQUIRE(qnum_cur[imode] == 0.0);
    REQUIRE(qnum_del_cond[imode] == 0.0);
    REQUIRE(qnum_del_coag[imode] == 0.0);
  }

  for (int ispec = 0; ispec < AeroConfig::num_aerosol_ids(); ++ispec) {
    for (int imode = 0; imode < AeroConfig::num_modes(); ++imode) {
      REQUIRE(qaer_cur[ispec][imode] == 0.0);
      REQUIRE(qaer_del_cond[ispec][imode] == 0.0);
      REQUIRE(qaer_del_coag[ispec][imode] == 0.0);
    }
  }

  for (int ispec = 0; ispec < AeroConfig::num_aerosol_ids(); ++ispec) {
    for (int imode = 0; imode < AeroConfig::max_agepair(); ++imode) {
      REQUIRE(qaer_del_coag_in[ispec][imode] == 0.0);
    }
  }
}
