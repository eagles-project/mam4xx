// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include "testing.hpp"

#include <mam4xx/mam4.hpp>

#include <ekat_comm.hpp>
#include <ekat_logger.hpp>
#include <ekat_type_traits.hpp>

#include <catch2/catch.hpp>

using mam4::Real;

TEST_CASE("test_aging_pairs", "mam4_aging_pairs") {
  // mam4 aging assumes that max_agepair is 1
  mam4::AeroConfig mam4_config(
      mam4::aero_species_on_device(mam4::default_aero_species()));
  REQUIRE(mam4_config.max_agepair() == 1);
}

TEST_CASE("test_cond_coag_mass_to_accum", "mam4_aging_process") {
  const int nsrc = static_cast<int>(mam4::ModeIndex::PrimaryCarbon);
  const int ndest = static_cast<int>(mam4::ModeIndex::Accumulation);

  std::vector<Real> qaer_cur(mam4::AeroConfig::num_modes(), 0.0);
  std::vector<Real> qaer_del_cond(mam4::AeroConfig::num_modes(), 0.0);
  std::vector<Real> qaer_del_coag(mam4::AeroConfig::num_modes(), 0.0);

  mam4::aging::transfer_cond_coag_mass_to_accum(
      nsrc, ndest, qaer_cur.data(), qaer_del_cond.data(), qaer_del_coag.data());

  for (int m = 0; m < mam4::AeroConfig::num_modes(); ++m) {
    REQUIRE(qaer_cur[m] == 0.0);
    REQUIRE(qaer_del_cond[m] == 0.0);
    REQUIRE(qaer_del_coag[m] == 0.0);
  }

  qaer_cur[nsrc] = 1.0;
  qaer_del_cond[nsrc] = 1.0;
  qaer_del_coag[nsrc] = 1.0;
  mam4::aging::transfer_cond_coag_mass_to_accum(
      nsrc, ndest, qaer_cur.data(), qaer_del_cond.data(), qaer_del_coag.data());

  Real sum_for_conservation = 0.0;
  for (int imode = 0; imode < mam4::AeroConfig::num_modes(); ++imode) {
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
  const int nsrc = static_cast<int>(mam4::ModeIndex::PrimaryCarbon);
  const int ndest = static_cast<int>(mam4::ModeIndex::Accumulation);

  Real xferfrac_pcage = 0.5;
  Real frac_cond = 0.25;
  Real frac_coag = 0.75;

  std::vector<Real> qaer_cur(mam4::AeroConfig::num_modes(), 0.0);
  std::vector<Real> qaer_del_cond(mam4::AeroConfig::num_modes(), 0.0);
  std::vector<Real> qaer_del_coag(mam4::AeroConfig::num_modes(), 0.0);

  qaer_cur[nsrc] = 0.0;

  mam4::aging::transfer_aged_pcarbon_to_accum(
      nsrc, ndest, xferfrac_pcage, frac_cond, frac_coag, qaer_cur.data(),
      qaer_del_cond.data(), qaer_del_coag.data());

  for (int imode = 0; imode < mam4::AeroConfig::num_modes(); ++imode) {
    REQUIRE(qaer_cur[imode] == 0.0);
    REQUIRE(qaer_del_cond[imode] == 0.0);
    REQUIRE(qaer_del_coag[imode] == 0.0);
  }

  qaer_cur[nsrc] = 1.0;
  mam4::aging::transfer_aged_pcarbon_to_accum(
      nsrc, ndest, xferfrac_pcage, frac_cond, frac_coag, qaer_cur.data(),
      qaer_del_cond.data(), qaer_del_coag.data());

  Real sum_for_conservation = 0.0;
  for (int imode = 0; imode < mam4::AeroConfig::num_modes(); ++imode) {
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

  auto aero_species =
      mam4::aero_species_on_device(mam4::default_aero_species());

  auto test_results = mam4::testing::create_on_device_test_results();
  Kokkos::parallel_for(
      "mam4_pcarbon_aging_1subarea", 1, KOKKOS_LAMBDA(const int i) {
        Real dgn_a[mam4::AeroConfig::num_modes()] = {};
        Real qnum_cur[mam4::AeroConfig::num_modes()] = {};
        Real qnum_del_cond[mam4::AeroConfig::num_modes()] = {};
        Real qnum_del_coag[mam4::AeroConfig::num_modes()] = {};
        Real qaer_cur[mam4::AeroConfig::num_aerosol_ids()]
                     [mam4::AeroConfig::num_modes()] = {};
        Real qaer_del_cond[mam4::AeroConfig::num_aerosol_ids()]
                          [mam4::AeroConfig::num_modes()] = {};
        Real qaer_del_coag[mam4::AeroConfig::num_aerosol_ids()]
                          [mam4::AeroConfig::num_modes()] = {};
        Real qaer_del_coag_in[mam4::AeroConfig::num_aerosol_ids()]
                             [mam4::AeroConfig::max_agepair()] = {};

        const unsigned n_so4_monolayers_pcage = 8;

        mam4::aging::mam_pcarbon_aging_1subarea(
            aero_species, n_so4_monolayers_pcage, dgn_a, qnum_cur,
            qnum_del_cond, qnum_del_coag, qaer_cur, qaer_del_cond,
            qaer_del_coag, qaer_del_coag_in);

        // Passing in zeros for everything should give zeros back
        for (int imode = 0; imode < mam4::AeroConfig::num_modes(); ++imode) {
          REQUIRE_ON_DEVICE(test_results, dgn_a[imode] == 0.0);
          REQUIRE_ON_DEVICE(test_results, qnum_cur[imode] == 0.0);
          REQUIRE_ON_DEVICE(test_results, qnum_del_cond[imode] == 0.0);
          REQUIRE_ON_DEVICE(test_results, qnum_del_coag[imode] == 0.0);
        }

        for (int ispec = 0; ispec < mam4::AeroConfig::num_aerosol_ids();
             ++ispec) {
          for (int imode = 0; imode < mam4::AeroConfig::num_modes(); ++imode) {
            REQUIRE_ON_DEVICE(test_results, qaer_cur[ispec][imode] == 0.0);
            REQUIRE_ON_DEVICE(test_results, qaer_del_cond[ispec][imode] == 0.0);
            REQUIRE_ON_DEVICE(test_results, qaer_del_coag[ispec][imode] == 0.0);
          }
        }

        for (int ispec = 0; ispec < mam4::AeroConfig::num_aerosol_ids();
             ++ispec) {
          for (int imode = 0; imode < mam4::AeroConfig::max_agepair();
               ++imode) {
            REQUIRE_ON_DEVICE(test_results,
                              qaer_del_coag_in[ispec][imode] == 0.0);
          }
        }
      });
  REQUIRE(mam4::testing::on_device_tests_passed(test_results));
}
