#include <mam4xx/mam4.hpp>

#include <skywalker.h>
#include <validation.hpp>

using namespace skywalker;
using namespace mam4;
using namespace haero;
void calculate_uustar(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    using View1DHost = typename HostType::view_1d<Real>;
    using View1D = typename DeviceType::view_1d<Real>;

    using ViewInt1D = typename DeviceType::view_1d<int>;
    using ViewInt1DHost = typename HostType::view_1d<int>;

    using ViewBool1D = typename DeviceType::view_1d<bool>;
    using ViewBool1DHost = typename HostType::view_1d<bool>;

    const auto index_season = input.get_array("index_season");
    const auto fr_lnduse = input.get_array("fr_lnduse");
    const bool unstable = static_cast<bool>(input.get("unstable"));
    const auto lcl_frc_landuse = input.get_array("lcl_frc_landuse");
    const Real va = input.get("va");
    const Real zl = input.get("zl");
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

    View1D lcl_frc_landuse_d("lcl_frc_landuse", n_land_type);
    View1DHost lcl_frc_landuse_h((Real *)lcl_frc_landuse.data(), n_land_type);
    Kokkos::deepcopy(lcl_frc_lnduse_d, lcl_frc_lnduse_h);

    Real uustar;

    auto team_policy = ThreadTeamPolicy(1u, 1u);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
          calculate_uustar(index_season_d.data(), fr_lnduse_d.data(), unstable,
                           lcl_frc_landuse_d.data(), va, zl, ribn, uustar);
        });

    output.set("uustar", uustar);
  });
}
