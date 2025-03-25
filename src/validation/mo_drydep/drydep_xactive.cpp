#include <mam4xx/mam4.hpp>

#include <skywalker.hpp>
#include <validation.hpp>

using namespace skywalker;
using namespace mam4;
using namespace mam4::mo_drydep;
using namespace haero;
void drydep_xactive(const seq_drydep::Data &data, Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    using View1DHost = typename HostType::view_1d<Real>;
    using View1D = typename DeviceType::view_1d<Real>;

    using ViewInt1D = typename DeviceType::view_1d<int>;
    using ViewInt1DHost = typename HostType::view_1d<int>;

    const auto fraction_landuse = input.get_array("fraction_landuse");
    const int ncdate = int(input.get_array("ncdate")[0]);
    // ncdate is MMDDYY format so extract the month:
    const int month = ncdate / 10000;
    // const auto col_index_season = input.get_array("col_index_season");
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

    View1DHost fraction_landuse_h((Real *)fraction_landuse.data(), n_land_type);
    View1D fraction_landuse_d("fraction_landuse", n_land_type);
    Kokkos::deep_copy(fraction_landuse_d, fraction_landuse_h);

    // FIXME:
    // NOTE: the input yaml has index_season_lai that only contains
    // arrays with a zillion 1's in them, so rather than reading that monster,
    // into memory, just doing this for now
    ViewInt1DHost col_index_season_h("col_index_season", n_land_type);
    for (int i = 0; i < n_land_type; ++i) {
      col_index_season_h(i) = 0;
    }
    ViewInt1D col_index_season_d("col_index_season", n_land_type);
    Kokkos::deep_copy(col_index_season_d, col_index_season_h);

    View1DHost mmr_h((Real *)mmr.data(), gas_pcnst);
    View1D mmr_d("mmr", gas_pcnst);
    Kokkos::deep_copy(mmr_d, mmr_h);

    View1D dvel_d("dvel", gas_pcnst);
    View1D dflx_d("dflx", gas_pcnst);

    auto team_policy = ThreadTeamPolicy(1u, Kokkos::AUTO());
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
          int index_season[n_land_type] = {};
          for (int lt = 0; lt < mam4::mo_drydep::n_land_type; ++lt) {
            index_season[lt] = col_index_season_d(month - 1);
          }
          if (snow > 0.01) { // BAD_CONSTANT
            for (int lt = 0; lt < mam4::mo_drydep::n_land_type; ++lt) {
              index_season[lt] = 3;
            }
          }
          drydep_xactive(data, fraction_landuse_d.data(), index_season,
                         sfc_temp, air_temp, tv, pressure_sfc, pressure_10m,
                         spec_hum, wind_speed, rain, solar_flux, mmr_d.data(),
                         dvel_d, dflx_d);
        });

    std::vector<Real> dvel(gas_pcnst), dflx(gas_pcnst);
    auto dvel_h = View1DHost((Real *)dvel.data(), gas_pcnst);
    auto dflx_h = View1DHost((Real *)dflx.data(), gas_pcnst);
    Kokkos::deep_copy(dvel_h, dvel_d);
    Kokkos::deep_copy(dflx_h, dflx_d);
    output.set("dvel", dvel);
    output.set("dflx", dflx);
  });
}
