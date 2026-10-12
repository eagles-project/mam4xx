// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/mam4.hpp>

#include <catch2/catch.hpp>
#include <mam4_test_config.hpp>

#include <cmath>
#include <string>

TEST_CASE("configured gas chemistry solver runs three timesteps",
          "[gas_chem][smoke]") {
  using mam4::Real;
  using namespace mam4::gas_chemistry;

  // The expected option comes directly from the CMake cache, independently
  // of the production macros that select the wrapper's solver call.
#if defined(MAM4XX_GAS_CHEMISTRY_SOLVER_LEGACY)
  REQUIRE(std::string(MAM4_TEST_GAS_CHEMISTRY_SOLVER) == "legacy");
#elif defined(MAM4XX_GAS_CHEMISTRY_SOLVER_DIRECT)
  REQUIRE(std::string(MAM4_TEST_GAS_CHEMISTRY_SOLVER) == "direct");
#endif

  constexpr int steps = 3;
  using RealView = mam4::DeviceType::view_1d<Real>;
  RealView history("gas_chemistry_smoke_history", (steps + 1) * gas_pcnst);

  Kokkos::parallel_for(
      "gas_chemistry_smoke", Kokkos::RangePolicy<>(0, 1),
      KOKKOS_LAMBDA(const int) {
        Real state[gas_pcnst];
        Real het_rates[gas_pcnst] = {};
        for (int j = 0; j < gas_pcnst; ++j) {
          // Synthetic values in the existing chemistry work-entry basis.
          state[j] = Real(j + 1) * Real(1e-9);
          het_rates[j] = j == o3_idx ? Real(0) : Real(1e-5); // s^-1
          history(j) = state[j];
        }

        Real photo_rates[mam4::mo_photo::phtcnt] = {};
        photo_rates[0] = Real(1e-5); // H2O2 photolysis frequency [s^-1]
        Real forcing[extcnt] = {};
        forcing[0] = Real(1e6); // SO2 source [molecules cm^-3 s^-1]
        forcing[8] = Real(1e6); // SOAG source [molecules cm^-3 s^-1]

        // Prescribed M, N2, O2, H2O, OH, NO3, HO2, O3 [molecules cm^-3].
        Real invariants[nfs] = {Real(2.5e19), Real(1.95e19), Real(5.25e18),
                               Real(1e17),   Real(1e6),    Real(1e7),
                               Real(1e8),    Real(1e12)};

        for (int step = 1; step <= steps; ++step) {
          mam4::microphysics::gas_phase_chemistry(
              Real(300), Real(60), photo_rates, forcing, invariants, het_rates,
              state);
          for (int j = 0; j < gas_pcnst; ++j) {
            history(step * gas_pcnst + j) = state[j];
          }
        }
      });
  Kokkos::fence();

  auto host_history = Kokkos::create_mirror_view(history);
  Kokkos::deep_copy(host_history, history);
  for (int step = 1; step <= steps; ++step) {
    CAPTURE(step);
    for (int j = 0; j < gas_pcnst; ++j) {
      CAPTURE(j);
      REQUIRE(std::isfinite(host_history(step * gas_pcnst + j)));
      REQUIRE(host_history(step * gas_pcnst + j) >= 0);
    }
    REQUIRE(host_history(step * gas_pcnst + o3_idx) == host_history(o3_idx));
    // DMS has losses and no source, so every call must advance the state.
    REQUIRE(host_history(step * gas_pcnst + 4) <
            host_history((step - 1) * gas_pcnst + 4));
  }
}
