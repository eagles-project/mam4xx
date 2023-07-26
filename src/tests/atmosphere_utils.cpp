// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include "atmosphere_utils.hpp"
#include <mam4xx/conversions.hpp>

#include <ekat/ekat_assert.hpp>
#include <haero/testing.hpp>

namespace mam4 {

Atmosphere init_atm_const_tv_lapse_rate(int num_levels, const Real pblh,
                                        const Real Tv0, const Real Gammav,
                                        const Real qv0, const Real qv1) {
  EKAT_REQUIRE_MSG(FloatingPoint<Real>::in_bounds(Tv0, 273, 323),
                   "unexpected Tv0, check units = K");
  EKAT_REQUIRE_MSG(FloatingPoint<Real>::in_bounds(Gammav, 0, 0.02),
                   "unexpected lapse rate, check units = K/m");

  const Real p0 = 1000e2;
  const Real ztop = 10e3;
  const Real dz = ztop / num_levels;

  auto d_temperature = haero::testing::create_column_view(num_levels);
  auto d_pressure = haero::testing::create_column_view(num_levels);
  auto d_mix = haero::testing::create_column_view(num_levels);
  auto d_liq = haero::testing::create_column_view(num_levels);
  auto d_nliq = haero::testing::create_column_view(num_levels);
  auto d_ice = haero::testing::create_column_view(num_levels);
  auto d_nice = haero::testing::create_column_view(num_levels);
  auto d_height = haero::testing::create_column_view(num_levels);
  auto d_hdp = haero::testing::create_column_view(num_levels);
  auto d_cf = haero::testing::create_column_view(num_levels);
  auto d_w = haero::testing::create_column_view(num_levels);
  auto h_temperature = Kokkos::create_mirror_view(d_temperature);
  auto h_pressure = Kokkos::create_mirror_view(d_pressure);
  auto h_mix = Kokkos::create_mirror_view(d_mix);
  auto h_height = Kokkos::create_mirror_view(d_height);
  auto h_hdp = Kokkos::create_mirror_view(d_hdp);

  Real psum = hydrostatic_pressure_at_height(ztop, p0, Tv0, Gammav);
  for (int k = 0; k < num_levels; ++k) {
    const Real z_up = ztop - k * dz;
    const Real z_mid = ztop - (k + 0.5) * dz;
    const Real z_down = ztop - (k + 1) * dz;
    const Real p_up = hydrostatic_pressure_at_height(z_up, p0, Tv0, Gammav);
    const Real p_down = hydrostatic_pressure_at_height(z_down, p0, Tv0, Gammav);
    const Real hdp = p_down - p_up;

    const Real w = init_specific_humidity(z_mid, qv0, qv1);
    const Real qv = conversions::specific_humidity_from_vapor_mixing_ratio(w);
    const Real tv = init_virtual_temperature(z_mid, Tv0, Gammav);

    h_temperature(k) =
        conversions::temperature_from_virtual_temperature(tv, qv);
    h_pressure(k) = hydrostatic_pressure_at_height(z_mid, p0, Tv0, Gammav);
    h_mix(k) = w;
    h_height(k) = z_mid;
    h_hdp(k) = hdp;

    psum += hdp;
  }

  // Assert equality based on relative values, rather than absolute tolerance
  EKAT_ASSERT(FloatingPoint<Real>::rel(psum, p0,
                                       std::numeric_limits<float>::epsilon()));

  Kokkos::deep_copy(d_temperature, h_temperature);
  Kokkos::deep_copy(d_pressure, h_pressure);
  Kokkos::deep_copy(d_mix, h_mix);
  Kokkos::deep_copy(d_height, h_height);
  Kokkos::deep_copy(d_hdp, h_hdp);

  Atmosphere atm(num_levels, d_temperature, d_pressure, d_mix, d_liq, d_nliq,
                 d_ice, d_nice, d_height, d_hdp, d_cf, d_w, pblh);

  return atm;
}

} // namespace mam4
