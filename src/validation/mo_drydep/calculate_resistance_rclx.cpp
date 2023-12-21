#include <mam4xx/mam4.hpp>

#include <skywalker.hpp>
#include <validation.hpp>

using namespace skywalker;
using namespace mam4;
using namespace mam4::mo_drydep;
using namespace haero;
void calculate_resistance_rclx(Ensemble *ensemble) {
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
    const auto heff = input.get_array("heff");
    const Real cts = input.get_array("cts")[0];

    ViewInt1DHost index_season_h("index_season", n_land_type);
    for (int lt = 0; lt < n_land_type; ++lt) {
      index_season_h(lt) = int(index_season[lt]);
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

    View1D rclx_d("rclx", gas_pcnst * n_land_type);

    auto team_policy = ThreadTeamPolicy(1u, 1u);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
          Real rclx[gas_pcnst][n_land_type];
          calculate_resistance_rclx(beglt, endlt, index_season_d.data(),
                                    fr_lnduse_d.data(), heff_d.data(), cts,
                                    rclx);
          // shuffle array data into view
          int l = 0;
          for (int i = 0; i < gas_pcnst; ++i) {
            for (int lt = 0; lt < n_land_type; ++lt, ++l) {
              rclx_d(l) = rclx[i][lt];
            }
          }
        });

    std::vector<Real> rclx(gas_pcnst * n_land_type);
    auto rclx_h = View1DHost((Real *)rclx.data(), gas_pcnst * n_land_type);
    Kokkos::deep_copy(rclx_h, rclx_d);
    output.set("rclx", rclx);
  });
}
