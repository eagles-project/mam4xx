#include <mam4xx/mam4.hpp>

#include <skywalker.h>
#include <validation.hpp>

using namespace skywalker;
using namespace mam4;
using namespace haero;
void calculate_ustar_over_water(Ensemble *ensemble) {
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
    const bool unstable = static_cast<bool>(input.get("unstable"));
    const Real zl = input.get("zl");
    const Real uustar = input.get("uustar");
    const Real ribn = input.get("ribn");

    Viewint1DHost index_season_h("index_season", n_land_type);
    for (int lt = 0; lt < n_land_type; ++lt) {
      index_season_h(lt) = int(index_season[lt]);
    }
    ViewInt1D index_season_d("index_season", n_land_type);
    Kokkos::deepcopy(index_season_d, index_season_h);

    ViewBool1DHost fr_lnduse_h("fr_lnduse", n_land_type);
    for (int lt = 0; lt < n_land_type; ++lt) {
      fr_lnduse_h(lt) = static_cast<bool>(fr_lnduse[lt]);
    }
    ViewBool1D fr_lnduse_d("fr_lnduse", n_land_type);
    Kokkos::deepcopy(fr_lnduse_d, fr_lnduse_h);

    View1D ustar_d("ustar", n_land_type);
    View1D cvar_d("cvar", n_land_type);
    View1D bycp_d("bycp", n_land_type);

    auto team_policy = ThreadTeamPolicy(1u, 1u);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
          calculate_ustar_over_water(
              beglt, endlt, index_season_d.data(), fr_lnduse_d.data(), unstable,
              zl, uustar, ribn, ustar_d.data(), cvar_d.data(), bycp_d.data());
        });

    std::vector<Real> ustar(n_land_type);
    auto ustar_h = View1DHost((Real *)ustar.data(), n_land_type);
    Kokkos::deep_copy(ustar_h, ustar_d);
    output.set("ustar", ustar);

    std::vector<Real> cvar(n_land_type);
    auto cvar_h = View1DHost((Real *)cvar.data(), n_land_type);
    Kokkos::deep_copy(cvar_h, cvar_d);
    output.set("cvar", cvar);

    std::vector<Real> bycp(n_land_type);
    auto bycp_h = View1DHost((Real *)bycp.data(), n_land_type);
    Kokkos::deep_copy(bycp_h, bycp_d);
    output.set("bycp", bycp);
  });
}
