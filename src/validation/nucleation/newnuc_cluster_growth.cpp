// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/nucleation.hpp>

#include <haero/constants.hpp>
#include <mam4xx/mam4_types.hpp>
#include <skywalker.hpp>
#include <validation.hpp>

#include <iostream>

using namespace skywalker;
using namespace mam4;

void newnuc_cluster_growth(Ensemble *ensemble) {
  constexpr Real rgas = Constants::r_gas; // [J/K/mol]
  constexpr Real pi = Constants::pi;
  constexpr Real avogadro = Constants::avogadro;              // [#/mol]
  constexpr Real mw_so4a = Constants::molec_weight_so4 * 1e3; // [g/mol]
  constexpr Real mw_nh4a = Constants::molec_weight_nh4 * 1e3; // [g/mol]

  // We don't need any settings for this particular test.
  // Settings settings = ensemble->settings();

  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    // Fetch ensemble parameters

    Real dnclusterdt = input.get("dnclusterdt");
    Real cnum_h2so4 = input.get("cnum_h2so4");
    Real cnum_nh3 = input.get("cnum_nh3");
    Real radius_cluster = input.get("radius_cluster");

    Real dplom_mode[1] = {input.get("dplom_mode")};
    Real dphim_mode[1] = {input.get("dphim_mode")};
    int nsize = static_cast<int>(input.get("nsize"));

    Real deltat = input.get("deltat");
    Real temp = input.get("temperature");
    Real relhumnn = input.get("relative_humidity");

    Real pmid = input.get("pmid");
    Real cair = pmid / (temp * rgas);

    Real accom_coef_h2so4 = input.get("accom_coef_h2so4");
    Real mw_so4a_host = input.get("mw_so4a_host");

    Real qnh3_cur = input.get("qnh3_cur");
    Real qh2so4_cur = input.get("qh2so4_cur");
    Real so4vol = input.get("so4vol");
    Real tmp_uptkrate = input.get("tmp_uptkrate");

    // Call the cluster growth function on device.
    DeviceType::view_1d<Real> return_vals("Return from Device", 5);
    Kokkos::parallel_for(
        "newnuc_cluster_growth", 1, KOKKOS_LAMBDA(int i) {
          // computed outputs
          int isize_group;
          Real dens_nh4so4a, qh2so4_del, qnh3_del, qso4a_del, qnh4a_del,
              qnuma_del;
          nucleation::newnuc_cluster_growth(
              dnclusterdt, cnum_h2so4, cnum_nh3, radius_cluster, dplom_mode,
              dphim_mode, nsize, deltat, temp, relhumnn, cair, accom_coef_h2so4,
              mw_so4a, mw_so4a_host, mw_nh4a, avogadro, pi, qnh3_cur,
              qh2so4_cur, so4vol, tmp_uptkrate, isize_group, dens_nh4so4a,
              qh2so4_del, qnh3_del, qso4a_del, qnh4a_del, qnuma_del);
          return_vals[0] = qh2so4_del;
          return_vals[1] = qnh3_del;
          return_vals[2] = qso4a_del;
          return_vals[3] = qnh4a_del;
          return_vals[4] = qnuma_del;
        });
    auto host_vals = Kokkos::create_mirror_view(return_vals);
    Kokkos::deep_copy(host_vals, return_vals);
    const Real qh2so4_del = host_vals[0];
    const Real qnh3_del = host_vals[1];
    const Real qso4a_del = host_vals[2];
    const Real qnh4a_del = host_vals[3];
    const Real qnuma_del = host_vals[4];

    output.set("qh2so4_del", qh2so4_del);
    output.set("qnh3_del", qnh3_del);
    output.set("qso4a_del", qso4a_del);
    output.set("qnh4a_del", qnh4a_del);
    output.set("qnuma_del", qnuma_del);
  });
}
