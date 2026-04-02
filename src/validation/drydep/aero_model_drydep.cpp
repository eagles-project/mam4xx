// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/drydep.hpp>
#include <validation.hpp>

using namespace skywalker;

void aero_model_drydep(Ensemble *ensemble) {

  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    const int nlev = mam4::nlev;
    const int aerosol_index = mam4::DryDeposition::index_interstitial_aerosols;

    const Real fraction_landuse[mam4::DryDeposition::n_land_type] = {
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
      for (int i = aerosol_index; i < mam4::aero_model::pcnst; ++i) {
        const std::string name = s + std::to_string(i + 1);
        EKAT_REQUIRE_MSG(input.has_array(name), "Required name: " + name);
        EKAT_REQUIRE_MSG(nlev == input.get_array(name).size(),
                         name + " array should be exactly 72 entries long.");
      }
    }

    for (std::string s : {"dgncur_awet", "wetdens"}) {
      EKAT_REQUIRE_MSG(input.has_array(s), "Required name: " + s);
      EKAT_REQUIRE_MSG(nlev * mam4::AeroConfig::num_modes() ==
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
      mam4::ColumnView dev = mam4::validation::create_column_view(vec.size());
      auto host = Kokkos::create_mirror_view(dev);
      for (int i = 0; i < vec.size(); ++i)
        host[i] = vec[i];
      Kokkos::deep_copy(dev, host);
      return dev;
    };

    mam4::ConstColumnView tair = to_dev(input.get_array("tair"));
    mam4::ConstColumnView pmid = to_dev(input.get_array("pmid"));
    mam4::ConstColumnView pint = to_dev(input.get_array("pint"));
    mam4::ConstColumnView pdel = to_dev(input.get_array("pdel"));

    auto state_q_mem =
        mam4::validation::create_column_view(nlev * mam4::aero_model::pcnst);
    mam4::Diagnostics::ColumnTracerView state_q(state_q_mem.data(), nlev,
                                                mam4::aero_model::pcnst);
    {
      auto host_q = Kokkos::create_mirror_view(state_q);
      for (int i = aerosol_index; i < mam4::aero_model::pcnst; ++i) {
        const std::string name = "statq_" + std::to_string(i + 1);
        const std::vector<Real> vec = input.get_array(name);
        for (int j = 0; j < vec.size(); ++j)
          host_q(j, i) = vec[j];
      }
      Kokkos::deep_copy(state_q, host_q);
    }

    Kokkos::View<Real *> qqcw[mam4::aero_model::pcnst];
    for (int i = 0; i < mam4::aero_model::pcnst; ++i) {
      const std::string name = "qqcw_" + std::to_string(i + 1);
      if (i < aerosol_index)
        qqcw[i] = mam4::validation::create_column_view(nlev);
      else
        qqcw[i] = to_dev(input.get_array(name));
    }
    mam4::ColumnView qqcw_tends[mam4::aero_model::pcnst];
    for (int i = 0; i < mam4::aero_model::pcnst; ++i)
      qqcw_tends[i] = qqcw[i];

    mam4::ConstColumnView dgncur_awet[mam4::AeroConfig::num_modes()];
    {
      std::vector<Real> dgncur_tot = input.get_array("dgncur_awet");
      for (int m = 0; m < mam4::AeroConfig::num_modes(); ++m) {
        std::vector<Real> dgncur(nlev);
        for (int lev = 0; lev < nlev; ++lev)
          dgncur[lev] = dgncur_tot[m * nlev + lev]; // col
        // dgncur[lev] = dgncur_tot[m + lev*AeroConfig::num_modes()]; // row
        dgncur_awet[m] = to_dev(dgncur);
      }
    }

    mam4::ConstColumnView wetdens[mam4::AeroConfig::num_modes()];
    {
      std::vector<Real> wetdens_tot = input.get_array("wetdens");
      for (int m = 0; m < mam4::AeroConfig::num_modes(); ++m) {
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
        mam4::validation::create_column_view(nlev * mam4::aero_model::pcnst);
    mam4::Diagnostics::ColumnTracerView ptend_q(ptend_q_mem.data(), nlev,
                                                mam4::aero_model::pcnst);

    mam4::ColumnView aerdepdryis =
        mam4::validation::create_column_view(mam4::aero_model::pcnst);
    mam4::ColumnView aerdepdrycw =
        mam4::validation::create_column_view(mam4::aero_model::pcnst);

    mam4::ColumnView rho = mam4::validation::create_column_view(nlev);
    const int aerosol_categories = mam4::DryDeposition::aerosol_categories;
    Kokkos::View<Real *> vlc_dry[mam4::AeroConfig::num_modes()]
                                [aerosol_categories],
        vlc_grv[mam4::AeroConfig::num_modes()][aerosol_categories];
    for (int j = 0; j < mam4::AeroConfig::num_modes(); ++j) {
      for (int i = 0; i < aerosol_categories; ++i) {
        Kokkos::resize(vlc_dry[j][i], nlev);
        Kokkos::resize(vlc_grv[j][i], nlev);
      }
    }
    mam4::ColumnView dry[mam4::AeroConfig::num_modes()][aerosol_categories];
    mam4::ColumnView grv[mam4::AeroConfig::num_modes()][aerosol_categories];
    for (int i = 0; i < mam4::AeroConfig::num_modes(); ++i)
      for (int j = 0; j < aerosol_categories; ++j) {
        dry[i][j] = vlc_dry[i][j];
        grv[i][j] = vlc_grv[i][j];
      }
    Kokkos::View<Real *> dqdt_tmp[mam4::aero_model::pcnst];
    for (int i = 0; i < mam4::aero_model::pcnst; ++i)
      Kokkos::resize(dqdt_tmp[i], nlev);
    mam4::ColumnView dqdt[mam4::aero_model::pcnst];
    for (int i = 0; i < mam4::aero_model::pcnst; ++i)
      dqdt[i] = dqdt_tmp[i];

    auto team_policy = mam4::ThreadTeamPolicy(1u, 1u);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const mam4::ThreadTeam &team) {
          Real vlc_trb[mam4::AeroConfig::num_modes()][aerosol_categories] = {};
          bool ptend_lq[mam4::aero_model::pcnst];
          mam4::aero_model_drydep(
              team, fraction_landuse, tair, pmid, pint, pdel, state_q,
              dgncur_awet, wetdens, obklen, ustar, landfrac, icefrac, ocnfrac,
              fricvelin, ram1in, dt, qqcw_tends, ptend_q, ptend_lq, aerdepdrycw,
              aerdepdryis, rho, dry, vlc_trb, grv, dqdt);
        });
    Kokkos::fence();
    auto to_host = [](mam4::ConstColumnView dev) {
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
    for (int m = aerosol_index; m < mam4::aero_model::pcnst; ++m) {
      for (int lev = 0; lev < nlev; ++lev)
        ptend[lev] = ptend_host(lev, m);
      output.set("ptendq_" + std::to_string(m + 1), ptend);
      output.set("qqcw_" + std::to_string(m + 1), to_host(qqcw[m]));
      output.set("aerdepdryis_" + std::to_string(m + 1), is_host[m]);
      output.set("aerdepdrycw_" + std::to_string(m + 1), cw_host[m]);
    }
  });
}
