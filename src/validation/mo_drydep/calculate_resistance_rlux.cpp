#include <mam4xx/mam4.hpp>

#include <skywalker.hpp>
#include <validation.hpp>

using namespace skywalker;
using namespace mam4;
using namespace mam4::mo_drydep;
using namespace haero;
void calculate_resistance_rlux(const seq_drydep::Data &data,
                               Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    using View1DHost = typename HostType::view_1d<Real>;
    using View1D = typename DeviceType::view_1d<Real>;

    using ViewInt1D = typename DeviceType::view_1d<int>;
    using ViewInt1DHost = typename HostType::view_1d<int>;

    using ViewBool1D = typename DeviceType::view_1d<bool>;
    using ViewBool1DHost = typename HostType::view_1d<bool>;

    const int beglt = int(input.get_array("beglt")[0]) - 1;
    const int endlt = int(input.get_array("endlt")[0]) - 1;
    const auto index_season = input.get_array("index_season");
    const auto fr_lnduse = input.get_array("fr_lnduse");
    const bool has_rain = static_cast<bool>(input.get_array("has_rain")[0]);
    const bool has_dew = static_cast<bool>(input.get_array("has_dew")[0]);
    const Real sfc_temp = input.get_array("sfc_temp")[0];
    const Real qs = input.get_array("qs")[0];
    const Real spec_hum = input.get_array("spec_hum")[0];
    const auto heff = input.get_array("heff");
    const Real cts = input.get_array("cts")[0];

    ViewInt1DHost index_season_h("index_season", n_land_type);
    for (int lt = 0; lt < n_land_type; ++lt) {
      index_season_h(lt) = int(index_season[lt]) - 1;
    }
    ViewInt1D index_season_d("index_season", n_land_type);
    Kokkos::deep_copy(index_season_d, index_season_h);

    ViewBool1DHost fr_lnduse_h("fr_lnduse", n_land_type);
    for (int lt = 0; lt < n_land_type; ++lt) {
      fr_lnduse_h(lt) = static_cast<bool>(fr_lnduse[lt]);
    }
    ViewBool1D fr_lnduse_d("fr_lnduse", n_land_type);
    Kokkos::deep_copy(fr_lnduse_d, fr_lnduse_h);

    View1D heff_d("heff", nddvels);
    View1DHost heff_h((Real *)heff.data(), nddvels);
    Kokkos::deep_copy(heff_d, heff_h);

    View1D rlux_d("rlux", gas_pcnst * n_land_type);

    auto team_policy = ThreadTeamPolicy(1u, 1u);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
          Real rlux[gas_pcnst][n_land_type];
          calculate_resistance_rlux(data, beglt, endlt, index_season_d.data(),
                                    fr_lnduse_d.data(), has_rain, has_dew,
                                    sfc_temp, qs, spec_hum, heff_d.data(), cts,
                                    rlux);
          // shuffle array data into view
          int l = 0;
          for (int i = 0; i < gas_pcnst; ++i) {
            for (int lt = 0; lt < n_land_type; ++lt, ++l) {
              rlux_d(l) = rlux[i][lt];
            }
          }
        });

    std::vector<Real> rlux(gas_pcnst * n_land_type);
    auto rlux_h = View1DHost((Real *)rlux.data(), gas_pcnst * n_land_type);
    Kokkos::deep_copy(rlux_h, rlux_d);
    output.set("rlux", rlux);
  });
}
