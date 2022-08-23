#include <catch2/catch.hpp>
#include <cmath>
#include <haero/mam4.hpp>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>

using namespace haero;

TEST_CASE("test_constructor", "mam4_gasaerexch_process") {
  mam4::AeroConfig mam4_config;
  mam4::GasAerExchProcess process(mam4_config);
  REQUIRE(process.name() == "MAM4 gas/aersol exchange");
  REQUIRE(process.aero_config() == mam4_config);
}

TEST_CASE("test_compute_tendencies", "mam4_gasaerexch_process") {
  int nlev = 72;
  Real pblh = 1000;
  Atmosphere atm(nlev, pblh);
  mam4::Prognostics progs(nlev);
  mam4::Diagnostics diags(nlev);
  mam4::Tendencies tends(nlev);

  mam4::AeroConfig mam4_config;
  mam4::GasAerExchProcess process(mam4_config);

  // Single-column dispatch.
  auto team_policy = TeamPolicy(1u, Kokkos::AUTO);
  Real t = 0.0, dt = 30.0;
  Kokkos::parallel_for(
      team_policy, KOKKOS_LAMBDA(const TeamType& team) {
        process.compute_tendencies(team, t, dt, atm, progs, diags, tends);
      });
}

TEST_CASE("test_multicol_compute_tendencies", "mam4_gasaerexch_process") {
  // Now we process multiple columns within a single dіspatch (mc means
  // "multi-column").
  int ncol = 8;
  DeviceType::view_1d<Atmosphere> mc_atm("mc_progs", ncol);
  DeviceType::view_1d<mam4::Prognostics> mc_progs("mc_atm", ncol);
  DeviceType::view_1d<mam4::Diagnostics> mc_diags("mc_diags", ncol);
  DeviceType::view_1d<mam4::Tendencies> mc_tends("mc_tends", ncol);
  const int nlev = 72;
  const Real pblh = 1000;
  Atmosphere atmosphere(nlev, pblh);
  mam4::Prognostics prognostics(nlev);
  mam4::Diagnostics diagnostics(nlev);
  mam4::Tendencies  tendencies(nlev);
  for (int icol = 0; icol < ncol; ++icol) {
    Kokkos::parallel_for(
        "Load multi-column views", 1, KOKKOS_LAMBDA(const int) {
          mc_atm(icol) = atmosphere;
          mc_progs(icol) = prognostics;
          mc_diags(icol) = diagnostics;
          mc_tends(icol) = tendencies;
        });
  }

  mam4::AeroConfig mam4_config;
  mam4::GasAerExchProcess process(mam4_config);

  // Dispatch over all the above columns.
  auto team_policy = TeamPolicy(ncol, Kokkos::AUTO);
  Real t = 0.0, dt = 30.0;
  Kokkos::parallel_for(
      team_policy, KOKKOS_LAMBDA(const TeamType& team) {
        const int icol = team.league_rank();
        process.compute_tendencies(team, t, dt, mc_atm(icol), mc_progs(icol),
                                   mc_diags(icol), mc_tends(icol));
      });
}
