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
#include <set>

// if you need something from the data/ directory
// std::string data_file = MAM4_TEST_DATA_DIR;
// #include <mam4_test_config.hpp>

using namespace haero;

TEST_CASE("test_constructor", "mam4_convproc_process") {
  mam4::AeroConfig mam4_config;
  mam4::ConvProcProcess process(mam4_config);
  REQUIRE(process.name() == "MAM4 convproc");
  REQUIRE(process.aero_config() == mam4_config);
}
TEST_CASE("update_conu_from_act_frac", "mam4_convproc_process") {
  const int la = 3;
  const int lc = 5;
  const Real act_frac = 0.5;
  const Real dt_u_inv = 0.25;
  ColumnView conu_dev = testing::create_column_view(mam4::ConvProc::pcnst_extd);
  {
    auto host_view = Kokkos::create_mirror_view(conu_dev);
    for (int i = 0; i < mam4::ConvProc::pcnst_extd; ++i)
      host_view[i] = i;
    Kokkos::deep_copy(conu_dev, host_view);
  }
  ColumnView dconudt_dev =
      testing::create_column_view(mam4::ConvProc::pcnst_extd);
  {
    auto host_view = Kokkos::create_mirror_view(dconudt_dev);
    for (int i = 0; i < mam4::ConvProc::pcnst_extd; ++i)
      host_view[i] = 2 * i;
    Kokkos::deep_copy(dconudt_dev, host_view);
  }
  Kokkos::parallel_for(
      1, KOKKOS_LAMBDA(const int) {
        Real conu[mam4::ConvProc::pcnst_extd];
        for (int i = 0; i < mam4::ConvProc::pcnst_extd; ++i)
          conu[i] = conu_dev[i];
        Real dconudt[mam4::ConvProc::pcnst_extd];
        for (int i = 0; i < mam4::ConvProc::pcnst_extd; ++i)
          dconudt[i] = dconudt_dev[i];
        mam4::convproc::update_conu_from_act_frac(conu, dconudt, la, lc,
                                                  act_frac, dt_u_inv);
        for (int i = 0; i < mam4::ConvProc::pcnst_extd; ++i)
          conu_dev[i] = conu[i];
        for (int i = 0; i < mam4::ConvProc::pcnst_extd; ++i)
          dconudt_dev[i] = dconudt[i];
      });
  Real conu[mam4::ConvProc::pcnst_extd];
  {
    auto host_view = Kokkos::create_mirror_view(conu_dev);
    Kokkos::deep_copy(host_view, conu_dev);
    for (int i = 0; i < mam4::ConvProc::pcnst_extd; ++i)
      conu[i] = host_view[i];
  }
  Real dconudt[mam4::ConvProc::pcnst_extd];
  {
    auto host_view = Kokkos::create_mirror_view(dconudt_dev);
    Kokkos::deep_copy(host_view, dconudt_dev);
    for (int i = 0; i < mam4::ConvProc::pcnst_extd; ++i)
      dconudt[i] = host_view[i];
  }
  REQUIRE(conu[la] == 1.5);
  REQUIRE(conu[lc] == 6.5);
  REQUIRE(dconudt[la] == -3.0 / 8.0);
  REQUIRE(dconudt[lc] == 3.0 / 8.0);
}
TEST_CASE("set_cloudborne_vars", "mam4_convproc_process") {
  const int gas_pcnst = mam4::ConvProc::gas_pcnst;
  const int num_modes = mam4::ConvProc::num_modes;
  const int pcnst_extd = mam4::ConvProc::pcnst_extd;
  const int maxd_aspectype = mam4::ConvProc::maxd_aspectype;
  ColumnView aqfrac_dev = testing::create_column_view(pcnst_extd);
  ColumnView doconvproc_extd_dev = testing::create_column_view(pcnst_extd);
  Kokkos::parallel_for(
      1, KOKKOS_LAMBDA(const int) {
        Real aqfrac[pcnst_extd];
        bool doconvproc_extd[pcnst_extd];
        {
          bool doconvproc[gas_pcnst];
          for (int i = 0; i < gas_pcnst; ++i)
            // Set every other values to true as a test.
            doconvproc[i] = i % 2;
          mam4::convproc::set_cloudborne_vars(doconvproc, aqfrac,
                                              doconvproc_extd);
        }
        for (int i = 0; i < pcnst_extd; ++i)
          aqfrac_dev[i] = aqfrac[i];
        for (int i = 0; i < pcnst_extd; ++i)
          doconvproc_extd_dev[i] = doconvproc_extd[i];
      });
  Real aqfrac[pcnst_extd];
  {
    auto host_view = Kokkos::create_mirror_view(aqfrac_dev);
    Kokkos::deep_copy(host_view, aqfrac_dev);
    for (int i = 0; i < pcnst_extd; ++i)
      aqfrac[i] = host_view[i];
  }
  bool doconvproc_extd[pcnst_extd];
  {
    auto host_view = Kokkos::create_mirror_view(doconvproc_extd_dev);
    Kokkos::deep_copy(host_view, doconvproc_extd_dev);
    for (int i = 0; i < pcnst_extd; ++i)
      doconvproc_extd[i] = host_view[i];
  }
  std::set<int> check;
  for (int j = 0; j < num_modes; ++j) {
    const int k = mam4::ConvProc::numptr_amode(j);
    // k%2 because that is what is set in doconvproc above.
    if (0 <= k && k % 2)
      check.insert(gas_pcnst + k);
    for (int i = 0; i < maxd_aspectype; ++i) {
      const int k = mam4::ConvProc::lmassptr_amode(i, j);
      // k%2 because that is what is set in doconvproc above.
      if (0 <= k && k % 2)
        check.insert(gas_pcnst + k);
    }
  }
  for (int i = 0; i < pcnst_extd; ++i) {
    if (check.count(i))
      REQUIRE(aqfrac[i] == 1.0);
    else
      REQUIRE(aqfrac[i] == 0.0);
  }
  for (int i = 0; i < pcnst_extd; ++i) {
    if (i < gas_pcnst) {
      // Fist values are set as in doconvproc:
      REQUIRE(doconvproc_extd[i] == i % 2);
    } else {
      // Extended values are set according to
      // mam4::ConvProc::lmassptr_amode:
      if (check.count(i))
        REQUIRE(doconvproc_extd[i] == true);
      else
        REQUIRE(doconvproc_extd[i] == false);
    }
  }
}
TEST_CASE("assign_dotend", "mam4_convproc_process") {
  const int gas_pcnst = mam4::ConvProc::gas_pcnst;
  ColumnView dotend_dev = testing::create_column_view(gas_pcnst);
  Kokkos::parallel_for(
      1, KOKKOS_LAMBDA(const int) {
        bool dotend[gas_pcnst];
        {
          const int species_class[gas_pcnst] = {
              0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2,
              2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
          const bool convproc_do_aer = true;
          const bool convproc_do_gas = false;
          mam4::convproc::assign_dotend(species_class, convproc_do_aer,
                                        convproc_do_gas, dotend);
        }
        for (int i = 0; i < gas_pcnst; ++i)
          dotend_dev[i] = dotend[i];
      });
  bool dotend[gas_pcnst];
  {
    auto host_view = Kokkos::create_mirror_view(dotend_dev);
    Kokkos::deep_copy(host_view, dotend_dev);
    for (int i = 0; i < gas_pcnst; ++i)
      dotend[i] = host_view[i];
  }
  for (int i = 0; i < gas_pcnst; ++i) {
    if (i < 15) {
      // First values are set to species_class != 2
      REQUIRE(dotend[i] == false);
    } else {
      // Rest of values are set to species_class == 2
      REQUIRE(dotend[i] == true);
    }
  }
  Kokkos::parallel_for(
      1, KOKKOS_LAMBDA(const int) {
        bool dotend[gas_pcnst];
        {
          const int species_class[gas_pcnst] = {
              0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2,
              2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
          const bool convproc_do_aer = false;
          const bool convproc_do_gas = true;
          mam4::convproc::assign_dotend(species_class, convproc_do_aer,
                                        convproc_do_gas, dotend);
        }
        for (int i = 0; i < gas_pcnst; ++i)
          dotend_dev[i] = dotend[i];
      });
  {
    auto host_view = Kokkos::create_mirror_view(dotend_dev);
    Kokkos::deep_copy(host_view, dotend_dev);
    for (int i = 0; i < gas_pcnst; ++i)
      dotend[i] = host_view[i];
  }
  for (int i = 0; i < gas_pcnst; ++i) {
    if (i < 9 || 14 < i) {
      // First values are set to species_class != 3
      REQUIRE(dotend[i] == false);
    } else {
      // Rest of values are set to species_class == 3
      REQUIRE(dotend[i] == true);
    }
  }
}
