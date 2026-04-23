// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/mam4.hpp>
#include <validation.hpp>

namespace mam4::mo_sethet {

//=================================================================================
KOKKOS_INLINE_FUNCTION
void old_gas_washout(
    const ThreadTeam &team,
    const int plev,               // calculate from this level below //in
    const Real xkgm,              // mass flux on rain drop //in
    const Real xliq_ik,           // liquid rain water content [gm/m^3] // in
    const ColumnView xhen_i,      // henry's law constant
    const ConstColumnView tfld_i, // temperature [K]
    const ColumnView delz_i,      // layer depth about interfaces [cm]  // in
    const ColumnView xgas) {      // gas concentration // inout
  //------------------------------------------------------------------------
  // calculate gas washout by cloud if not saturated
  //------------------------------------------------------------------------
  // FIXME: BAD CONSTANTS
  const Real const0 = boltz_cgs * 1.0e-6; // [atmospheres/deg k/cm^3]
  constexpr Real geo_fac =
      6.0; // geometry factor (surf area/volume = geo_fac/diameter)
  constexpr Real xrm = .189;  // mean diameter of rain drop [cm]
  constexpr Real xum = 748.0; // mean rain drop terminal velocity [cm/s]

  // -----------------------------------------------------------------
  //       ... calculate the saturation concentration eqca
  // -----------------------------------------------------------------
  // total of ca between level plev and kk [#/cm3]
  Real allca = 0.0;

  // This loop causes problems because plev is the only level this
  // function should be changing but it changes values in other levels
  // which prevents gas_washout from running in parallel over all the
  // levels in a column.
  for (int k = plev; k < pver; k++) {
    const Real xeqca =
        xgas(k) / (xliq_ik * avo2 + 1.0 / (xhen_i(k) * const0 * tfld_i(k))) *
        xliq_ik * avo2;
    //-----------------------------------------------------------------
    //       ... calculate ca; inside cloud concentration in  #/cm3(air)
    //-----------------------------------------------------------------
    const Real xca =
        geo_fac * xkgm * xgas(k) / (xrm * xum) * delz_i(k) * xliq_ik * cm3_2_m3;

    // -----------------------------------------------------------------
    //       ... if is not saturated (take hno3 as an example)
    //               hno3(gas)_new = hno3(gas)_old - hno3(h2o)
    //           otherwise
    //               hno3(gas)_new = hno3(gas)_old
    // -----------------------------------------------------------------
    allca += xca;
    if (allca < xeqca) {
      xgas(k) = mam4::max(xgas(k) - xca, 0.0);
    }
  }
} // end subroutine old_gas_washout

} // namespace mam4::mo_sethet

using namespace skywalker;
using namespace mam4::mo_sethet;

void gas_washout(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    using View1DHost = typename mam4::HostType::view_1d<Real>;
    constexpr int pver = mam4::nlev;

    const int plev =
        input.get_array("plev")[0] - 1; // 1-based to 0-based index conversion
    const Real xkgm = input.get_array("xkgm")[0];
    const auto xliq_i_in = input.get_array("xliq_ik");
    const auto xhen_i_in = input.get_array("xhen_i");
    const auto tfld_i_in = input.get_array("tfld_i");
    const auto delz_i_in = input.get_array("delz_i");
    const auto xgas_in = input.get_array("xgas");

    mam4::ColumnView xliq_i, xhen_i, tfld_i, delz_i, xgas_old, xgas_new;
    auto xliq_i_host = View1DHost((Real *)xliq_i_in.data(), pver);
    auto xhen_i_host = View1DHost((Real *)xhen_i_in.data(), pver);
    auto tfld_i_host = View1DHost((Real *)tfld_i_in.data(), pver);
    auto delz_i_host = View1DHost((Real *)delz_i_in.data(), pver);
    auto xgas_host = View1DHost((Real *)xgas_in.data(), pver);
    xliq_i = mam4::testing::create_column_view(pver);
    xhen_i = mam4::testing::create_column_view(pver);
    tfld_i = mam4::testing::create_column_view(pver);
    delz_i = mam4::testing::create_column_view(pver);
    xgas_old = mam4::testing::create_column_view(pver);
    xgas_new = mam4::testing::create_column_view(pver);
    Kokkos::deep_copy(xliq_i, xliq_i_host);
    Kokkos::deep_copy(xhen_i, xhen_i_host);
    Kokkos::deep_copy(tfld_i, tfld_i_host);
    Kokkos::deep_copy(delz_i, delz_i_host);
    Kokkos::deep_copy(xgas_old, xgas_host);
    Kokkos::deep_copy(xgas_new, xgas_host);

    // rain just needs to be non-zero to trigger this test
    auto rain_i = mam4::testing::create_column_view(pver);
    Kokkos::deep_copy(rain_i, 0.1);

    auto team_policy = mam4::ThreadTeamPolicy(1u, Kokkos::AUTO);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const mam4::ThreadTeam &team) {
          Kokkos::single(Kokkos::PerTeam(team), [=]() {
            for (int kk = plev; kk < pver; ++kk)
              old_gas_washout(team, kk, xkgm, xliq_i[kk], xhen_i, tfld_i,
                              delz_i, xgas_old);
          });
        });
    // now the new one
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const mam4::ThreadTeam &team) {
          gas_washout(team, plev, pver, xkgm, xliq_i, rain_i, xhen_i, tfld_i,
                      delz_i, xgas_new);
        });

    Kokkos::deep_copy(xgas_host, xgas_old);
    std::vector<Real> xgas_old_out(pver);
    for (int k = 0; k < pver; k++) {
      xgas_old_out[k] = xgas_host(k);
    }
    Kokkos::deep_copy(xgas_host, xgas_new);
    std::vector<Real> xgas_new_out(pver);
    for (int k = 0; k < pver; k++) {
      xgas_new_out[k] = xgas_host(k);
    }

    // compute an error norm
    Real l2 = 0.0;
    for (int k = 0; k < pver; k++) {
      const Real diff = xgas_new_out[k] - xgas_old_out[k];
      l2 += diff * diff;
    }
    l2 = sqrt(l2);
    printf("xgas error (L2): %g\n", l2);

    output.set("xgas", xgas_new_out);
  });
}
