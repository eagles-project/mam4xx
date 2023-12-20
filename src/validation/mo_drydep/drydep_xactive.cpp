#include <mam4xx/mam4.hpp>

#include <skywalker.h>
#include <validation.hpp>

using namespace skywalker;
using namespace mam4;
using namespace haero;
void drydep_xactive(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    using View1DHost = typename HostType::view_1d<Real>;
    using View1D = typename DeviceType::view_1d<Real>;

    using ViewInt1D = typename DeviceType::view_1d<int>;
    using ViewInt1DHost = typename HostType::view_1d<int>;

    const int ncdate = int(input.get_array("ncdate")[0]);
    const auto col_index_season = input.get_array("col_index_season");
    const Real sfc_temp = input.get_array("sfc_temp")[0];
    const Real air_temp = input.get_array("air_temp")[0];
    const Real tv = input.get_array("tv")[0];
    const Real pressure_sfc = input.get_array("pressure_sfc")[0];
    const Real pressure_10m = input.get_array("pressure_10m")[0];
    const Real spec_hum = input.get_array("spec_hum")[0];
    const Real wind_speed = input.get_array("wind_speed")[0];
    const Real rain = input.get_array("rain")[0];
    const Real snow = input.get_array("snow")[0];
    const Real solar_flux = input.get_array("solar_flux")[0];
    const auto mmr = input.get_array("mmr");

    ViewInt1DHost col_index_season_h("col_index_season", 12);
    for (int i = 0; i < 12; ++i) {
      col_index_season_h(lt) = int(col_index_season[i]);
    }
    ViewInt1D col_index_season_d("col_index_season", 12);
    Kokkos::deepcopy(col_index_season_d, col_index_season_h);

    View1DHost mmr_h((Real *)mmr.data(), gas_pcnst);
    View1D mmr_d("mmr", gas_pcnst);
    Kokkos::deepcopy(mmr_d, mmr_h);

    View1D dvel_d("dvel", gas_pcnst);
    View1D dflx_d("dflx", gas_pcnst);

    auto team_policy = ThreadTeamPolicy(1u, 1u);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
          drydep_xactive(ncdate, col_index_season_d.data(), sfc_temp, air_temp,
                         tv, pressure_sfc, pressure_10m, spec_hum, wind_speed,
                         rain, snow, solar_flux, mmr_d.data(), dvel_d.data(),
                         dflx_d.data());
        });

    std::vector<Real> dvel(gas_pcnst), dflx(gas_pcnst);
    auto dvel_h = View1DHost((Real *)dep_ra.data(), gas_pcnst);
    auto dflx_h = View1DHost((Real *)dflx.data(), gas_pcnst);
    Kokkos::deep_copy(dvel_h, dvel_d);
    Kokkos::deep_copy(dflx_h, dflx_d);
    output.set("dvel", dvel);
    output.set("dflx", dflx);
  });
}
