#include <mam4xx/mam4.hpp>

#include <skywalker.hpp>
#include <validation.hpp>

using namespace skywalker;
using namespace mam4;
using namespace mam4::mo_drydep;
using namespace haero;
void calculate_gas_drydep_vlc_and_flux(const seq_drydep::Data &data,
                                       Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    using View1DHost = typename HostType::view_1d<Real>;
    using View1D = typename DeviceType::view_1d<Real>;
    using View2D = typename DeviceType::view_2d<Real>;

    using ViewInt1D = typename DeviceType::view_1d<int>;
    using ViewInt1DHost = typename HostType::view_1d<int>;

    using ViewBool1D = typename DeviceType::view_1d<bool>;
    using ViewBool1DHost = typename HostType::view_1d<bool>;

    const int beglt = int(input.get_array("beglt")[0]) - 1;
    const int endlt = int(input.get_array("endlt")[0]) - 1;
    const auto index_season = input.get_array("index_season");
    const auto fr_lnduse = input.get_array("fr_lnduse");
    const auto lcl_frc_landuse = input.get_array("lcl_frc_landuse");
    const auto mmr = input.get_array("mmr");
    const auto dep_ra = input.get_array("dep_ra");
    const auto dep_rb = input.get_array("dep_rb");
    const Real term = input.get_array("term")[0];
    const auto rsmx = input.get_array("rsmx");
    const auto rlux = input.get_array("rlux");
    const auto rclx = input.get_array("rclx");
    const auto rgsx = input.get_array("rgsx");
    const Real rdc = input.get_array("rdc")[0];

    ViewInt1DHost index_season_h("index_season", n_land_type);
    for (int lt = 0; lt < n_land_type; ++lt) {
      // Fortran to C++ indexing
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

    // FIXME:
    // HACK: we currently have too many columns of lcl_frc_landuse, so we'll
    // just choose the first for now
    int lcl_frc_landuse_col = 0;

    View1DHost lcl_frc_landuse_h("lcl_frc_landuse", n_land_type);
    int ll = 0;
    for (int i = 0; i < 4; ++i) {
      for (int lt = 0; lt < n_land_type; ++lt, ++ll) {
        if (i == lcl_frc_landuse_col) {
          lcl_frc_landuse_h(lt) = lcl_frc_landuse[ll];
        } else {
          continue;
        }
      }
    }

    View1D lcl_frc_landuse_d("lcl_frc_landuse", n_land_type);
    Kokkos::deep_copy(lcl_frc_landuse_d, lcl_frc_landuse_h);

    // View1D mmr_d("mmr", gas_pcnst);
    // View1DHost mmr_h((Real *)mmr.data(), gas_pcnst);
    // Kokkos::deep_copy(mmr_d, mmr_h);

    // FIXME: change this to compile time num_levels from balwinder's PR
    View2D mmr_view("mmr_view", 72, gas_pcnst);
    mam4::validation::convert_1d_vector_to_2d_view_device(mmr, mmr_view);
    const auto mmr_d =
        Kokkos::subview(mmr_view, 72 - 1, Kokkos::ALL);

    View1D dep_ra_d("dep_ra", n_land_type);
    View1DHost dep_ra_h((Real *)dep_ra.data(), n_land_type);
    Kokkos::deep_copy(dep_ra_d, dep_ra_h);

    View1D dep_rb_d("dep_rb", n_land_type);
    View1DHost dep_rb_h((Real *)dep_rb.data(), n_land_type);
    Kokkos::deep_copy(dep_rb_d, dep_rb_h);

    View1D rsmx_d("rsmx", gas_pcnst * n_land_type);
    View1DHost rsmx_h((Real *)rsmx.data(), gas_pcnst * n_land_type);
    Kokkos::deep_copy(rsmx_d, rsmx_h);

    View1D rlux_d("rlux", gas_pcnst * n_land_type);
    View1DHost rlux_h((Real *)rlux.data(), gas_pcnst * n_land_type);
    Kokkos::deep_copy(rlux_d, rlux_h);

    View1D rclx_d("rclx", gas_pcnst * n_land_type);
    View1DHost rclx_h((Real *)rclx.data(), gas_pcnst * n_land_type);
    Kokkos::deep_copy(rclx_d, rclx_h);

    View1D rgsx_d("rgsx", gas_pcnst * n_land_type);
    View1DHost rgsx_h((Real *)rgsx.data(), gas_pcnst * n_land_type);
    Kokkos::deep_copy(rgsx_d, rgsx_h);

    View1D dvel_d("dvel", gas_pcnst);
    View1D dflx_d("dflx", gas_pcnst);
    Kokkos::deep_copy(dvel_d, 0.0);
    Kokkos::deep_copy(dflx_d, 0.0);

    auto team_policy = ThreadTeamPolicy(1u, 1u);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
          // shuffle data rsmx, rlux, rclx, rgsx arrays

          Real rsmx[gas_pcnst][n_land_type];
          Real rlux[gas_pcnst][n_land_type];
          Real rclx[gas_pcnst][n_land_type];
          Real rgsx[gas_pcnst][n_land_type];
          int l = 0;

          for (int i = 0; i < gas_pcnst; ++i) {
            for (int lt = 0; lt < n_land_type; ++lt, ++l) {
              rsmx[i][lt] = rsmx_d(l);
              rlux[i][lt] = rlux_d(l);
              rclx[i][lt] = rclx_d(l);
              rgsx[i][lt] = rgsx_d(l);
            }
          }
          calculate_gas_drydep_vlc_and_flux(
              data, beglt, endlt, index_season_d.data(), fr_lnduse_d.data(),
              lcl_frc_landuse_d.data(), mmr_d.data(), dep_ra_d.data(),
              dep_rb_d.data(), term, rsmx, rlux, rclx, rgsx, rdc, dvel_d.data(),
              dflx_d.data());
        });

    // std::vector<Real> dvel(gas_pcnst, 0.0);
    // auto dvel_h = View1DHost((Real *)dvel.data(), gas_pcnst);
    // Kokkos::deep_copy(dvel_h, dvel_d);
    // output.set("dvel", dvel);

    // std::vector<Real> dflx(gas_pcnst);
    // auto dflx_h = View1DHost((Real *)dflx.data(), gas_pcnst);
    // Kokkos::deep_copy(dflx_h, dflx_d);
    // output.set("dflx", dflx);

    View1DHost dvel_h("dvel_h", gas_pcnst);
    View1DHost dflx_h("dflx_h", gas_pcnst);

    Kokkos::deep_copy(dvel_h, dvel_d);
    Kokkos::deep_copy(dflx_h, dflx_d);
    std::vector<Real> dflx_out(gas_pcnst);
    std::vector<Real> dvel_out(gas_pcnst);

    for (int lt = 0; lt < gas_pcnst; ++lt) {
      dvel_out[lt] = dvel_h(lt);
      dflx_out[lt] = dflx_h(lt);
    }

    output.set("dvel", dvel_out);
    output.set("dflx", dflx_out);
  });
}
