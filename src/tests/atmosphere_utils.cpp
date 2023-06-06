// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include "atmosphere_utils.hpp"
#include <mam4xx/conversions.hpp>

#include <haero/testing.hpp>
#include <ekat/ekat_assert.hpp>

namespace mam4 {

Atmosphere init_atm_const_tv_lapse_rate(int num_levels, const Real pblh,
    const Real Tv0, const Real Gammav, const Real qv0, const Real qv1) {
  EKAT_REQUIRE_MSG(FloatingPoint<Real>::in_bounds(Tv0, 273, 323),
                   "unexpected Tv0, check units = K");
  EKAT_REQUIRE_MSG(FloatingPoint<Real>::in_bounds(Gammav, 0, 0.02),
                   "unexpected lapse rate, check units = K/m");

  const Real p0 = 1000e2;
  const Real ztop = 10e3;
  const Real dz = ztop / num_levels;

  using HostColumnView = typename haero::HostType::view_1d<Real>;
  auto h_temperature = HostColumnView("T", num_levels);
  auto h_pressure = HostColumnView("p", num_levels);
  auto h_mix = HostColumnView("qv", num_levels);
  auto h_height = HostColumnView("h", num_levels);
  auto h_hdp = HostColumnView("hdp", num_levels);

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

  auto d_temperature = haero::testing::create_column_view(num_levels);
  auto d_pressure = haero::testing::create_column_view(num_levels);
  auto d_mix = haero::testing::create_column_view(num_levels);
  auto d_height = haero::testing::create_column_view(num_levels);
  auto d_hdp = haero::testing::create_column_view(num_levels);

  Kokkos::deep_copy(d_temperature, h_temperature);
  Kokkos::deep_copy(d_pressure, h_pressure);
  Kokkos::deep_copy(d_mix, h_mix);
  Kokkos::deep_copy(d_height, h_height);
  Kokkos::deep_copy(d_hdp, h_hdp);

  Atmosphere atm(num_levels, pblh);
  atm.temperature = d_temperature;
  atm.pressure = d_pressure;
  atm.vapor_mixing_ratio = d_mix;
  atm.height = d_height;
  atm.hydrostatic_dp = d_hdp;

  return atm;
}

} // namespace mam4
