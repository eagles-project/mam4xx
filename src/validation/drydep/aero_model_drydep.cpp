// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <iostream>
#include <mam4xx/drydep.hpp>
#include <skywalker.hpp>
#include <validation.hpp>

using namespace skywalker;
using namespace mam4;

void aero_model_drydep(Ensemble *ensemble) {

  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    const int nlev = 72;
    const Real fraction_landuse[DryDeposition::n_land_type] = {
        0.20918898065265040e-02, 0.10112323792561469e+00,
        0.19104123086831826e+00, 0.56703179010502225e+00,
        0.00000000000000000e+00, 0.42019237748858657e-01,
        0.85693761223933115e-01, 0.66234294754917442e-02,
        0.00000000000000000e+00, 0.00000000000000000e+00,
        0.43754228462347953e-02};

    EKAT_REQUIRE_MSG(input.has_array("ntot_amode"),
                     "Required name: ntot_amode");
    EKAT_REQUIRE_MSG(1 == input.get_array("ntot_amode").size(),
                     "ntot_amode array should be exactly 1 entry long.");
    EKAT_REQUIRE_MSG(4 == int(input.get_array("ntot_amode").front()),
                     "ntot_amode array should be exactly 4.");

    for (std::string s :
         {"tair", "pmid", "pint", "pdel", "obklen", "ustar", "landfrac",
          "icefrac", "ocnfrac", "fricvelin", "ram1in", "dt"})
      EKAT_REQUIRE_MSG(input.has_array(s), "Required name: " + s);

    for (std::string s : {"statq_", "qqcw_"}) {
      for (int i = 15; i < ConvProc::gas_pcnst; ++i) {
        const std::string name = s + std::to_string(i + 1);
        EKAT_REQUIRE_MSG(input.has_array(name), "Required name: " + name);
        EKAT_REQUIRE_MSG(nlev == input.get_array(name).size(),
                         name + " array should be exactly 72 entries long.");
      }
    }

    for (std::string s : {"dgncur_awet", "wetdens"}) {
      EKAT_REQUIRE_MSG(input.has_array(s), "Required name: " + s);
      EKAT_REQUIRE_MSG(nlev * AeroConfig::num_modes() ==
                           input.get_array(s).size(),
                       s + " array should be exactly 288 entry long.");
    }

    for (std::string s : {"tair", "pmid", "pdel"})
      EKAT_REQUIRE_MSG(nlev == input.get_array(s).size(),
                       s + " array should be exactly 72 entries long.");

    for (std::string s : {"pint"})
      EKAT_REQUIRE_MSG(1 + nlev == input.get_array(s).size(),
                       s + " array should be exactly 72 entries long.");

    for (std::string s : {"obklen", "ustar", "landfrac", "icefrac", "ocnfrac",
                          "fricvelin", "ram1in", "dt"})
      EKAT_REQUIRE_MSG(1 == input.get_array(s).size(),
                       s + " array should be exactly 1 entry long.");

    auto to_dev = [](const std::vector<Real> &vec) {
      ColumnView dev = mam4::validation::create_column_view(vec.size());
      auto host = Kokkos::create_mirror_view(dev);
      for (int i = 0; i < vec.size(); ++i)
        host[i] = vec[i];
      Kokkos::deep_copy(dev, host);
      return dev;
    };

    haero::ConstColumnView tair = to_dev(input.get_array("tair"));
    haero::ConstColumnView pmid = to_dev(input.get_array("pmid"));
    haero::ConstColumnView pint = to_dev(input.get_array("pint"));
    haero::ConstColumnView pdel = to_dev(input.get_array("pdel"));

    auto state_q_mem =
        mam4::validation::create_column_view(nlev * ConvProc::gas_pcnst);
    Diagnostics::ColumnTracerView state_q(state_q_mem.data(), nlev,
                                          ConvProc::gas_pcnst);
    {
      auto host_q = Kokkos::create_mirror_view(state_q);
      for (int i = 15; i < ConvProc::gas_pcnst; ++i) {
        const std::string name = "statq_" + std::to_string(i + 1);
        const std::vector<Real> vec = input.get_array(name);
        for (int j = 0; j < vec.size(); ++j)
          host_q(j, i) = vec[j];
      }
      Kokkos::deep_copy(state_q, host_q);
    }

    Kokkos::View<Real *> qqcw[ConvProc::gas_pcnst];
    for (int i = 0; i < ConvProc::gas_pcnst; ++i) {
      const std::string name = "qqcw_" + std::to_string(i + 1);
      if (i < 15)
        qqcw[i] = mam4::validation::create_column_view(nlev);
      else
        qqcw[i] = to_dev(input.get_array(name));
    }

    ColumnView dgncur_awet[AeroConfig::num_modes()];
    {
      std::vector<Real> dgncur_tot = input.get_array("dgncur_awet");
      for (int m = 0; m < AeroConfig::num_modes(); ++m) {
        std::vector<Real> dgncur(nlev);
        for (int lev = 0; lev < nlev; ++lev)
          dgncur[lev] = dgncur_tot[m * nlev + lev]; // col
        // dgncur[lev] = dgncur_tot[m + lev*AeroConfig::num_modes()]; // row
        dgncur_awet[m] = to_dev(dgncur);
      }
    }

    ColumnView wetdens[AeroConfig::num_modes()];
    {
      std::vector<Real> wetdens_tot = input.get_array("wetdens");
      for (int m = 0; m < AeroConfig::num_modes(); ++m) {
        std::vector<Real> wet(nlev);
        for (int lev = 0; lev < nlev; ++lev)
          wet[lev] = wetdens_tot[m * nlev + lev]; // col
        // wet[lev] = wetdens_tot[m + lev*AeroConfig::num_modes()]; // row
        wetdens[m] = to_dev(wet);
      }
    }

    const Real obklen = input.get_array("obklen").front();
    const Real ustar = input.get_array("ustar").front();
    const Real landfrac = input.get_array("landfrac").front();
    const Real icefrac = input.get_array("icefrac").front();
    const Real ocnfrac = input.get_array("ocnfrac").front();
    const Real fricvelin = input.get_array("fricvelin").front();
    const Real ram1in = input.get_array("ram1in").front();
    const Real dt = input.get_array("dt").front();

    auto ptend_q_mem =
        mam4::validation::create_column_view(nlev * ConvProc::gas_pcnst);
    Diagnostics::ColumnTracerView ptend_q(ptend_q_mem.data(), nlev,
                                          ConvProc::gas_pcnst);

    ColumnView aerdepdryis =
        mam4::validation::create_column_view(ConvProc::gas_pcnst);
    ColumnView aerdepdrycw =
        mam4::validation::create_column_view(ConvProc::gas_pcnst);

    ColumnView rho = mam4::validation::create_column_view(nlev);
    Kokkos::View<Real *> vlc_dry[AeroConfig::num_modes()][4],
        vlc_trb[AeroConfig::num_modes()][4],
        vlc_grv[AeroConfig::num_modes()][4];
    for (int j = 0; j < AeroConfig::num_modes(); ++j) {
      for (int i = 0; i < 4; ++i) {
        Kokkos::resize(vlc_dry[j][i], nlev);
        Kokkos::resize(vlc_trb[j][i], nlev);
        Kokkos::resize(vlc_grv[j][i], nlev);
      }
    }
    Kokkos::View<Real *> dqdt_tmp[ConvProc::gas_pcnst];
    for (int i = 0; i < ConvProc::gas_pcnst; ++i)
      Kokkos::resize(dqdt_tmp[i], nlev);

    auto team_policy = haero::ThreadTeamPolicy(1u, 1u);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
          bool ptend_lq[ConvProc::gas_pcnst];
          mam4::aero_model_drydep(
              team, nlev, fraction_landuse, tair, pmid, pint, pdel, state_q,
              dgncur_awet, wetdens, qqcw, obklen, ustar, landfrac, icefrac,
              ocnfrac, fricvelin, ram1in, ptend_q, ptend_lq, dt, aerdepdrycw,
              aerdepdryis, rho, vlc_dry, vlc_trb, vlc_grv, dqdt_tmp);
        });
    Kokkos::fence();
    auto to_host = [](haero::ConstColumnView dev) {
      auto host = Kokkos::create_mirror_view(dev);
      Kokkos::deep_copy(host, dev);
      std::vector<Real> vec(host.size());
      for (int i = 0; i < vec.size(); ++i)
        vec[i] = host[i];
      return vec;
    };
    auto ptend_host = Kokkos::create_mirror_view(ptend_q);
    std::vector<Real> is_host = to_host(aerdepdryis);
    std::vector<Real> cw_host = to_host(aerdepdrycw);
    Kokkos::deep_copy(ptend_host, ptend_q);
    std::vector<Real> ptend(nlev);
    for (int m = 15; m < ConvProc::gas_pcnst; ++m) {
      for (int lev = 0; lev < nlev; ++lev)
        ptend[lev] = ptend_host(lev, m);
      output.set("ptendq_" + std::to_string(m + 1), ptend);
      output.set("qqcw_" + std::to_string(m + 1), to_host(qqcw[m]));
      output.set("aerdepdryis_" + std::to_string(m + 1), is_host[m]);
      output.set("aerdepdrycw_" + std::to_string(m + 1), cw_host[m]);
    }
  });
}
