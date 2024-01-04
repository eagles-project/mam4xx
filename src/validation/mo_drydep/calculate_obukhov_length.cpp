#include <mam4xx/mam4.hpp>

#include <skywalker.hpp>
#include <validation.hpp>

using namespace skywalker;
using namespace mam4;
using namespace mam4::mo_drydep;
using namespace haero;
void calculate_obukhov_length(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    using View1DHost = typename HostType::view_1d<Real>;
    using View1D = typename DeviceType::view_1d<Real>;

    using ViewBool1D = typename DeviceType::view_1d<bool>;
    using ViewBool1DHost = typename HostType::view_1d<bool>;

    const int beglt = int(input.get_array("beglt")[0]) - 1;
    const int endlt = int(input.get_array("endlt")[0]) - 1;
    const auto fr_lnduse = input.get_array("fr_lnduse");
    const bool unstable = static_cast<bool>(input.get_array("unstable")[0]);
    const Real tha = input.get_array("tha")[0];
    const Real thg = input.get_array("thg")[0];
    const auto ustar = input.get_array("ustar");
    const auto cvar = input.get_array("cvar");
    const Real va = input.get_array("va")[0];
    const auto bycp = input.get_array("bycp");
    const Real ribn = input.get_array("ribn")[0];

    ViewBool1DHost fr_lnduse_h("fr_lnduse", n_land_type);
    for (int lt = 0; lt < n_land_type; ++lt) {
      fr_lnduse_h(lt) = static_cast<bool>(fr_lnduse[lt]);
    }
    ViewBool1D fr_lnduse_d("fr_lnduse", n_land_type);
    Kokkos::deep_copy(fr_lnduse_d, fr_lnduse_h);

    View1DHost ustar_h((Real *)ustar.data(), n_land_type);
    View1D ustar_d("ustar", n_land_type);
    Kokkos::deep_copy(ustar_d, ustar_h);

    View1DHost cvar_h((Real *)cvar.data(), n_land_type);
    View1D cvar_d("ustar", n_land_type);
    Kokkos::deep_copy(cvar_d, cvar_h);

    View1DHost bycp_h((Real *)bycp.data(), n_land_type);
    View1D bycp_d("bycp", n_land_type);
    Kokkos::deep_copy(bycp_d, bycp_h);

    View1D obklen_d("obklen", n_land_type);
    Kokkos::deep_copy(obklen_d, 0);

    auto team_policy = ThreadTeamPolicy(1u, 1u);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
          calculate_obukhov_length(beglt, endlt, fr_lnduse_d.data(), unstable,
                                   tha, thg, ustar_d.data(), cvar_d.data(), va,
                                   bycp_d.data(), ribn, obklen_d.data());
        });

    std::vector<Real> obklen(n_land_type);
    auto obklen_h = View1DHost((Real *)obklen.data(), n_land_type);
    Kokkos::deep_copy(obklen_h, obklen_d);
    output.set("obklen", obklen);
  });
}
