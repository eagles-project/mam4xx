#include "atmosphere_init.hpp"
#include "mam4xx/conversions.hpp"

#include <ekat/ekat_assert.hpp>

namespace mam4 {

void init_atm_const_tv_lapse_rate(const Atmosphere& atm, const Real Tv0, const Real Gammav,
  const Real qv0, const Real qv1) {
  EKAT_REQUIRE_MSG(FloatingPoint<Real>::in_bounds(Tv0, 273, 323),
                           "unexpected Tv0, check units = K");
  EKAT_REQUIRE_MSG(FloatingPoint<Real>::in_bounds(Gammav, 0, 0.02),
                           "unexpected lapse rate, check units = K/m");
  const int nlev = atm.num_levels();

  const Real p0 = 1000e2;
  const Real ztop =10e3;
  const Real dz = ztop / nlev;

  auto h_temperature = Kokkos::create_mirror_view(atm.temperature);
  auto h_pressure = Kokkos::create_mirror_view(atm.pressure);
  auto h_mix = Kokkos::create_mirror_view(atm.vapor_mixing_ratio);
  auto h_height = Kokkos::create_mirror_view(atm.height);
  auto h_hdp = Kokkos::create_mirror_view(atm.hydrostatic_dp);


  Real psum = hydrostatic_pressure_at_height(ztop, p0, Tv0, Gammav);
  for (int k=0; k<nlev; ++k) {
    const int pack_idx = PackInfo::pack_idx(k);
    const int vec_idx = PackInfo::vec_idx(k);

    const Real z_up = ztop - k*dz;
    const Real z_mid= ztop - (k+0.5)*dz;
    const Real z_down = ztop - (k+1)*dz;
    const Real p_up = hydrostatic_pressure_at_height(z_up, p0, Tv0, Gammav);
    const Real p_down = hydrostatic_pressure_at_height(z_down, p0, Tv0, Gammav);
    const Real hdp = p_down - p_up;

    const Real w = init_specific_humidity(z_mid, qv0, qv1);
    const Real qv = conversions::specific_humidity_from_vapor_mixing_ratio(w);
    const Real tv = init_virtual_temperature(z_mid, Tv0, Gammav);

    h_temperature(pack_idx)[vec_idx] = conversions::temperature_from_virtual_temperature(
      tv, qv);
    h_pressure(pack_idx)[vec_idx] = hydrostatic_pressure_at_height(z_mid, p0, Tv0, Gammav);
    h_mix(pack_idx)[vec_idx] = w;
    h_height(pack_idx)[vec_idx] = z_mid;
    h_hdp(pack_idx)[vec_idx] = hdp;

    psum += hdp;
  }

  EKAT_ASSERT(FloatingPoint<Real>::equiv(psum, p0, std::numeric_limits<float>::epsilon()));

  Kokkos::deep_copy(atm.temperature, h_temperature);
  Kokkos::deep_copy(atm.pressure, h_pressure);
  Kokkos::deep_copy(atm.vapor_mixing_ratio, h_mix);
  Kokkos::deep_copy(atm.height, h_height);
  Kokkos::deep_copy(atm.hydrostatic_dp, h_hdp);
}

} // namespace mam4
