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
