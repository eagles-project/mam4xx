// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include "testing.hpp"
#include <mam4xx/mam4.hpp>
#include "mam4xx/aero_modes.hpp"

#include <haero/constants.hpp>

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

TEST_CASE("test_constructor", "mam4_wet_deposition_process") {
  ekat::Comm comm;
  ekat::logger::Logger<> logger("wet deposition constructor test",
                                ekat::logger::LogLevel::debug, comm);
  mam4::AeroConfig mam4_config;
  mam4::WetDepositionProcess::ProcessConfig process_config;
  mam4::WetDepositionProcess process(mam4_config, process_config);
  REQUIRE(process.name() == "MAM4 Wet Deposition");
  REQUIRE(process.aero_config() == mam4_config);
}

TEST_CASE("test_local_precip_production", "mam4_wet_deposition_process") {
  ekat::Comm comm;
  ekat::logger::Logger<> logger("wet deposition local precip production test",
                                ekat::logger::LogLevel::debug, comm);
  int nlev = 72;
  Real pblh = 1000;
  Atmosphere atm = mam4::testing::create_atmosphere(nlev, pblh);

  const int pver = atm.num_levels();

  ColumnView pdel = mam4::testing::create_column_view(pver);
  ColumnView source_term = mam4::testing::create_column_view(pver);
  ColumnView sink_term = mam4::testing::create_column_view(pver);
  ColumnView lprec = mam4::testing::create_column_view(pver);

  // Need to use Kokkos to initialize values
  Kokkos::parallel_for(
    "intialize_values_local_precip", 1, KOKKOS_LAMBDA(const int) {
      for (int i = 0; i < pver; i++) {
        pdel(i) = 1.0;
        source_term(i) = 2.0;
        sink_term(i) = 1.5;
        lprec(i) = 0.0;
      }
    });

  Kokkos::parallel_for(
    "test_local_precip_production", 1, KOKKOS_LAMBDA(const int) {
      Real *pdel_device = pdel.data();
      Real *source_term_device = source_term.data();
      Real *sink_term_device = sink_term.data();
      Real *lprec_device = lprec.data();
      mam4::wetdep::local_precip_production(pdel_device, source_term_device, 
                                            sink_term_device, lprec_device,
                                            atm);
    });

  auto pdel_view = Kokkos::create_mirror_view(pdel);
  Kokkos::deep_copy(pdel_view, pdel);
  auto source_term_view = Kokkos::create_mirror_view(source_term);
  Kokkos::deep_copy(source_term_view, source_term);
  auto sink_term_view = Kokkos::create_mirror_view(sink_term);
  Kokkos::deep_copy(sink_term_view, sink_term);
  auto lprec_view = Kokkos::create_mirror_view(lprec);
  Kokkos::deep_copy(lprec_view, lprec);

  // TODO - generate/use real validation data
  for (int i = 0; i < pver; i++) {
    REQUIRE(pdel_view(i) == 1.0);
    REQUIRE(source_term_view(i) == 2.0);
    REQUIRE(sink_term_view(i) == 1.5);
    REQUIRE(lprec_view(i) ==
            pdel_view(i)/ Constants::gravity * (source_term_view(i) - sink_term_view(i)));
  }
}

TEST_CASE("test_calculate_cloudy_volume", "mam4_wet_deposition_process") {
  ekat::Comm comm;
  ekat::logger::Logger<> logger("wet deposition calculate cloudy volume test",
                                ekat::logger::LogLevel::debug, comm);
  int nlev = 72;
  Real pblh = 1000;
  Atmosphere atm = mam4::testing::create_atmosphere(nlev, pblh);

  const int pver = atm.num_levels();

  ColumnView cld = mam4::testing::create_column_view(pver);
  ColumnView lprec = mam4::testing::create_column_view(pver);
  ColumnView cldv = mam4::testing::create_column_view(pver);
  ColumnView sumppr_all = mam4::testing::create_column_view(pver);

  // Need to use Kokkos to initialize values
  Kokkos::parallel_for(
    "intialize_values_local_precip", 1, KOKKOS_LAMBDA(const int) {
      for (int i = 0; i < pver; i++) {
        cld(i) = 1.0;
        lprec(i) = 2.0;
        cldv(i) = 1.5;
        sumppr_all(i) = 0.0;
      }
    }); 

  // Pass true to flag
  Kokkos::parallel_for(
    "test_calculate_cloudy_volume_true", 1, KOKKOS_LAMBDA(const int) {
      Real *cld_device = cld.data();
      Real *lprec_device = lprec.data();
      Real *cldv_device = cldv.data();
      Real *sumppr_all_device = sumppr_all.data();
      mam4::wetdep::calculate_cloudy_volume(cld_device, lprec_device, true, cldv_device,
                                            sumppr_all_device, atm);
  });

  auto cld_view = Kokkos::create_mirror_view(cld);
  Kokkos::deep_copy(cld_view, cld);
  auto lprec_view = Kokkos::create_mirror_view(lprec);
  Kokkos::deep_copy(lprec_view, lprec);
  auto cldv_view = Kokkos::create_mirror_view(cldv);
  Kokkos::deep_copy(cldv_view, cldv);
  auto sumppr_all_view = Kokkos::create_mirror_view(sumppr_all);
  Kokkos::deep_copy(sumppr_all_view, sumppr_all);
  
  // TODO - generate/use real validation data
  for (int i = 0; i < pver; i++) {
    REQUIRE(cld_view(i) == 1.0);
    REQUIRE(lprec_view(i) == 2.0);
    // REQUIRE(cldv_view(i) == 1.5);
    // REQUIRE(sumppr_all_view(i) == 0.0);
  }

  // Need to use Kokkos to initialize values
  Kokkos::parallel_for(
    "intialize_values_local_precip", 1, KOKKOS_LAMBDA(const int) {
      for (int i = 0; i < pver; i++) {
        cld(i) = 1.0;
        lprec(i) = 2.0;
        cldv(i) = 1.5;
        sumppr_all(i) = 0.0;
      }
    }); 

  // Pass false to flag
  Kokkos::parallel_for(
    "test_calculate_cloudy_volume_false", 1, KOKKOS_LAMBDA(const int) {
      Real *cld_device = cld.data();
      Real *lprec_device = lprec.data();
      Real *cldv_device = cldv.data();
      Real *sumppr_all_device = sumppr_all.data();
      mam4::wetdep::calculate_cloudy_volume(cld_device, lprec_device, false, 
                                            cldv_device, sumppr_all_device, atm);
    });

  // Only need to copy as view is already created
  Kokkos::deep_copy(cld_view, cld);
  Kokkos::deep_copy(lprec_view, lprec);
  Kokkos::deep_copy(cldv_view, cldv);
  Kokkos::deep_copy(sumppr_all_view, sumppr_all);

  // TODO - generate/use real validation data
  for (int i = 0; i < pver; i++) {
    REQUIRE(cld_view(i) == 1.0);
    REQUIRE(lprec_view(i) == 2.0);
    // REQUIRE(cldv_view(i) == 1.5);
    // REQUIRE(sumppr_all_view(i) == 0.0);
  }
}

TEST_CASE("test_rain_mix_ratio", "mam4_wet_deposition_process") {
  ekat::Comm comm;
  ekat::logger::Logger<> logger("rain mixing ratio test",
                                ekat::logger::LogLevel::debug, comm);
  int nlev = 72;
  Real pblh = 1000;
  Atmosphere atm = mam4::testing::create_atmosphere(nlev, pblh);

  const int pver = atm.num_levels();
  
  ColumnView temperature = mam4::testing::create_column_view(pver);
  ColumnView pmid = mam4::testing::create_column_view(pver);
  ColumnView sumppr = mam4::testing::create_column_view(pver);
  ColumnView rain = mam4::testing::create_column_view(pver);

  // Need to use Kokkos to initialize values
  Kokkos::parallel_for(
    "intialize_values_local_precip", 1, KOKKOS_LAMBDA(const int) {
      for (int i = 0; i < pver; i++) {
        temperature(i) = 1.0;
        pmid(i) = 2.0;
        sumppr(i) = 1.5;
        rain(i) = 0.0;
      }
    }); 

  Kokkos::parallel_for(
    "rain_mix_ratio_test", 1, KOKKOS_LAMBDA(const int) {
      Real *temperature_device = temperature.data();
      Real *pmid_device = pmid.data();
      Real *sumppr_device = sumppr.data();
      Real *rain_device = rain.data();
      mam4::wetdep::rain_mix_ratio(temperature_device, pmid_device, sumppr_device,
                                   rain_device, atm);
    });

  auto temperature_view = Kokkos::create_mirror_view(temperature);
  Kokkos::deep_copy(temperature_view, temperature);
  auto pmid_view = Kokkos::create_mirror_view(pmid);
  Kokkos::deep_copy(pmid_view, pmid);
  auto sumppr_view = Kokkos::create_mirror_view(sumppr);
  Kokkos::deep_copy(sumppr_view, sumppr);
  auto rain_view = Kokkos::create_mirror_view(rain);
  Kokkos::deep_copy(rain_view, rain);

  // TODO - generate/use real validation data
  for (int i = 0; i < pver; i++) {
    REQUIRE(temperature_view(i) == 1.0);
    REQUIRE(pmid_view(i) == 2.0);
    REQUIRE(sumppr_view(i) == 1.5);
    // REQUIRE(rain_view(i) == 0.0);
  }
}
