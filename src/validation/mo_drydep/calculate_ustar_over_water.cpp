#include <mam4xx/mam4.hpp>

#include <skywalker.hpp>
#include <validation.hpp>

using namespace skywalker;
using namespace mam4;
using namespace mam4::mo_drydep;
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
    const bool unstable = static_cast<bool>(input.get_array("unstable")[0]);
    const Real zl = input.get_array("zl")[0];
    const Real uustar = input.get_array("uustar")[0];
    const Real ribn = input.get_array("ribn")[0];

    const auto bycp = input.get_array("bycp");
    const auto cvar = input.get_array("cvar");
    const auto ustar = input.get_array("ustar");

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

    View1DHost ustar_h("ustar", n_land_type);
    View1DHost cvar_h("cvar", n_land_type);
    View1DHost bycp_h("bycp", n_land_type);
    for (int lt = 0; lt < n_land_type; ++lt) {
      ustar_h(lt) = ustar[lt];
      cvar_h(lt) = cvar[lt];
      bycp_h(lt) = bycp[lt];
    }
    View1D ustar_d("ustar", n_land_type);
    View1D cvar_d("cvar", n_land_type);
    View1D bycp_d("bycp", n_land_type);

    Kokkos::deep_copy(ustar_d, ustar_h);
    Kokkos::deep_copy(cvar_d, cvar_h);
    Kokkos::deep_copy(bycp_d, bycp_h);

    auto team_policy = ThreadTeamPolicy(1u, 1u);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
          calculate_ustar_over_water(
              beglt, endlt, index_season_d.data(), fr_lnduse_d.data(), unstable,
              zl, uustar, ribn, ustar_d.data(), cvar_d.data(), bycp_d.data());
        });

    Kokkos::deep_copy(ustar_h, ustar_d);
    Kokkos::deep_copy(cvar_h, cvar_d);
    Kokkos::deep_copy(bycp_h, bycp_d);
    std::vector<Real> bycp_out(n_land_type);
    std::vector<Real> cvar_out(n_land_type);
    std::vector<Real> ustar_out(n_land_type);

    for (int lt = 0; lt < n_land_type; ++lt) {
      ustar_out[lt] = ustar_h(lt);
      cvar_out[lt] = cvar_h(lt);
      bycp_out[lt] = bycp_h(lt);
    }

    output.set("ustar", ustar_out);
    output.set("cvar", cvar_out);
    output.set("bycp", bycp_out);
  });
}
