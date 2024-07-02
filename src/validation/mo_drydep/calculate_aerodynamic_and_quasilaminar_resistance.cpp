#include <mam4xx/mam4.hpp>

#include <skywalker.hpp>
#include <validation.hpp>

using namespace skywalker;
using namespace mam4;
using namespace mam4::mo_drydep;
using namespace haero;
void calculate_aerodynamic_and_quasilaminar_resistance(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    using View1DHost = typename HostType::view_1d<Real>;
    using View1D = typename DeviceType::view_1d<Real>;

    using ViewBool1D = typename DeviceType::view_1d<bool>;
    using ViewBool1DHost = typename HostType::view_1d<bool>;

    const int beglt = int(input.get_array("beglt")[0]) - 1;
    const int endlt = int(input.get_array("endlt")[0]) - 1;
    const Real zl = input.get_array("zl")[0];
    const auto fr_lnduse = input.get_array("fr_lnduse");
    const auto obklen = input.get_array("obklen");
    const auto ustar = input.get_array("ustar");
    const auto cvar = input.get_array("cvar");

    ViewBool1DHost fr_lnduse_h("fr_lnduse", n_land_type);
    for (int lt = 0; lt < n_land_type; ++lt) {
      fr_lnduse_h(lt) = static_cast<bool>(fr_lnduse[lt]);
    }
    ViewBool1D fr_lnduse_d("fr_lnduse", n_land_type);
    Kokkos::deep_copy(fr_lnduse_d, fr_lnduse_h);

    View1DHost obklen_h((Real *)obklen.data(), n_land_type);
    View1D obklen_d("obklen", n_land_type);
    Kokkos::deep_copy(obklen_d, obklen_h);

    View1DHost ustar_h((Real *)ustar.data(), n_land_type);
    View1D ustar_d("ustar", n_land_type);
    Kokkos::deep_copy(ustar_d, ustar_h);

    View1DHost cvar_h((Real *)cvar.data(), n_land_type);
    View1D cvar_d("cvar", n_land_type);
    Kokkos::deep_copy(cvar_d, cvar_h);

    View1D dep_ra_d("dep_ra", n_land_type);
    View1D dep_rb_d("dep_rb", n_land_type);

    auto team_policy = ThreadTeamPolicy(1u, 1u);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
          calculate_aerodynamic_and_quasilaminar_resistance(
              beglt, endlt, fr_lnduse_d.data(), zl, obklen_d.data(),
              ustar_d.data(), cvar_d.data(), dep_ra_d.data(), dep_rb_d.data());
        });

    std::vector<Real> dep_ra(n_land_type), dep_rb(n_land_type);
    auto dep_ra_h = View1DHost((Real *)dep_ra.data(), n_land_type);
    Kokkos::deep_copy(dep_ra_h, dep_ra_d);
    auto dep_rb_h = View1DHost((Real *)dep_rb.data(), n_land_type);
    Kokkos::deep_copy(dep_rb_h, dep_rb_d);
    output.set("dep_ra", dep_ra);
    output.set("dep_rb", dep_rb);
  });
}
