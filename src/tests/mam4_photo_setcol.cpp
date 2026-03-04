// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include "atmosphere_utils.hpp"
#include "testing.hpp"
#include <catch2/catch.hpp>
#include <ekat_logger.hpp>
#include <mam4xx/mam4.hpp>

// Precision-dependent tolerance traits
template <typename T> struct PrecisionTolerance;

template <> struct PrecisionTolerance<float> {
  static constexpr float tol = 1e-5f; // Single precision tolerance
};

template <> struct PrecisionTolerance<double> {
  static constexpr double tol = 1e-12; // Double precision tolerance
};

using namespace mam4;
using namespace haero;

// Test compute_o3_column_density: serial reference vs parallel implementation
TEST_CASE("compute_o3_column_density", "mo_photo") {

  ekat::Comm comm;

  ekat::logger::Logger<> logger("compute_o3_column_density tests",
                                ekat::logger::LogLevel::debug, comm);
  using View1D = DeviceType::view_1d<Real>;
  using View1DHost = typename HostType::view_1d<Real>;

  // Initialize random number generator for reproducibility
  std::default_random_engine generator(98765);
  std::uniform_real_distribution<double> unif_dist(1e-8, 1e-5);

  constexpr int pver = mam4::nlev;
  constexpr Real tol = PrecisionTolerance<Real>::tol;

  // Test inputs
  const Real o3_col_deltas_0 =
      1.6e15;                    // Top-of-atmosphere column density [1/cm²]
  const Real mw_o3 = 48.0 / 1e3; // Ozone molecular weight [kg/mol]
  constexpr Real xfactor = 2.8704e21 / (9.80616 * 1.38044); // From set_ub_col

  // Atmospheric profile parameters (humid atmosphere with RH ~32%-98%)
  Real pblh = 1000;         // Planetary boundary layer height [m]
  const Real Tv0 = 300;     // Reference virtual temperature [K]
  const Real Gammav = 0.01; // Virtual temperature lapse rate [K/m]
  const Real qv0 = 0.015;   // Surface specific humidity [kg/kg]
  const Real qv1 = 7.5e-4;  // Specific humidity lapse rate [1/m]
  Atmosphere atm =
      mam4::init_atm_const_tv_lapse_rate(pver, pblh, Tv0, Gammav, qv0, qv1);

  // Create host views for inputs
  View1DHost pdel_host("pdel_host", pver);     // Pressure thickness [Pa]
  View1DHost mmr_o3_host("mmr_o3_host", pver); // Mass mixing ratio [kg/kg]
  View1DHost o3_col_dens_ref("o3_col_dens_ref", pver); // Reference output

  const auto pdel = atm.hydrostatic_dp; // Pressure thickness array [Pa]
  Kokkos::deep_copy(pdel_host, pdel);
  // Initialize random input data
  for (int k = 0; k < pver; ++k) {
    mmr_o3_host(k) = unif_dist(generator);
  }

  // ===== SERIAL REFERENCE IMPLEMENTATION =====
  // Step 1: Compute column density deltas for each level
  View1DHost o3_col_deltas_host("o3_col_deltas_host", pver + 1);
  o3_col_deltas_host(0) = o3_col_deltas_0;

  Real running_sum = 0.0;
  for (int kk = 0; kk < nlev; ++kk) {
    const Real vmr_o3_kk =
        mam4::conversions::vmr_from_mmr(mmr_o3_host(kk), mw_o3);
    const Real delta_kk = xfactor * pdel_host(kk) * vmr_o3_kk;

    o3_col_dens_ref(kk) = o3_col_deltas_0 + running_sum + 0.5 * delta_kk;

    running_sum += delta_kk;
  }

  // ===== PARALLEL IMPLEMENTATION USING compute_o3_column_density =====
  // Copy inputs to device
  View1D mmr_o3("mmr_o3", pver);
  View1D o3_col_dens("o3_col_dens", pver);
  Kokkos::deep_copy(mmr_o3, mmr_o3_host);

  auto team_policy = ThreadTeamPolicy(1, Kokkos::AUTO);
  Kokkos::parallel_for(
      "compute_o3_column", team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
        mam4::microphysics::compute_o3_column_density(
            team,
            pdel,            // Pressure thickness array [nlev]
            mmr_o3,          // Ozone mass mixing ratio [nlev]
            o3_col_deltas_0, // Top-of-atmosphere column density [1/cm²]
            mw_o3,           // Ozone molecular weight [kg/mol]
            o3_col_dens      // Output: column density [1/cm²] [nlev]
        );
      });
  Kokkos::fence();

  // Copy results back to host for comparison
  auto o3_col_dens_host = Kokkos::create_mirror_view(o3_col_dens);
  Kokkos::deep_copy(o3_col_dens_host, o3_col_dens);

  // Compare serial reference with parallel implementation

  for (int k = 0; k < pver; ++k) {
    const Real diff =
        std::abs(o3_col_dens_host(k) - o3_col_dens_ref(k)) / o3_col_dens_ref(k);
    if (diff >= tol) {
      std::ostringstream ss;
      ss << "diff : [ ";
      ss << "Level " << k << ": diff = " << diff
         << ", parallel = " << o3_col_dens_host(k)
         << ", serial = " << o3_col_dens_ref(k) << "\n";
      ss << "]";
      logger.debug(ss.str());
    }
    REQUIRE(diff <= tol);
  }
}
