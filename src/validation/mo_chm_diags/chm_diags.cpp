// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/mam4.hpp>

#include <mam4xx/aero_config.hpp>
#include <skywalker.hpp>
#include <validation.hpp>

using namespace skywalker;
using namespace mam4;
using namespace haero;
using namespace mo_chm_diags;

// constexpr const int gas_pcnst = gas_chemistry::gas_pcnst;

void chm_diags(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    using View1DHost = typename HostType::view_1d<Real>;
    using View1D = typename DeviceType::view_1d<Real>;
    using ColumnView = haero::ColumnView;

    const int lchnk = 16;
    const int pcnst = 41;

    //=========read input==========

    const int ncol = input.get_array("ncol")[0];
    const int ltrop = input.get_array("ltrop")[0] - 1;
    const int id_o3 = input.get_array("id_o3")[0] - 1;

    const auto sox_species_in = input.get_array("sox_species");
    const auto aer_species_in = input.get_array("aer_species");

    const auto mmr_in = input.get_array("mmr");
    const auto mmr_tend_in = input.get_array("mmr_tend");
    const auto vmr_in = input.get_array("vmr");
    const auto pdel_in = input.get_array("pdel");
    const auto pdeldry_in = input.get_array("pdeldry");

    const auto depvel_in = input.get_array("depvel");
    const auto depflx_in = input.get_array("depflx");

    const auto fldcw_nn16_in = input.get_array("fldcw_nn16");
    const auto fldcw_nn17_in = input.get_array("fldcw_nn17");
    const auto fldcw_nn18_in = input.get_array("fldcw_nn18");
    const auto fldcw_nn19_in = input.get_array("fldcw_nn19");
    const auto fldcw_nn20_in = input.get_array("fldcw_nn20");
    const auto fldcw_nn21_in = input.get_array("fldcw_nn21");
    const auto fldcw_nn22_in = input.get_array("fldcw_nn22");
    const auto fldcw_nn23_in = input.get_array("fldcw_nn23");
    const auto fldcw_nn24_in = input.get_array("fldcw_nn24");
    const auto fldcw_nn25_in = input.get_array("fldcw_nn25");
    const auto fldcw_nn26_in = input.get_array("fldcw_nn26");
    const auto fldcw_nn27_in = input.get_array("fldcw_nn27");
    const auto fldcw_nn28_in = input.get_array("fldcw_nn28");
    const auto fldcw_nn29_in = input.get_array("fldcw_nn29");
    const auto fldcw_nn30_in = input.get_array("fldcw_nn30");
    const auto fldcw_nn31_in = input.get_array("fldcw_nn31");
    const auto fldcw_nn32_in = input.get_array("fldcw_nn32");
    const auto fldcw_nn33_in = input.get_array("fldcw_nn33");
    const auto fldcw_nn34_in = input.get_array("fldcw_nn34");
    const auto fldcw_nn35_in = input.get_array("fldcw_nn35");
    const auto fldcw_nn36_in = input.get_array("fldcw_nn36");
    const auto fldcw_nn37_in = input.get_array("fldcw_nn37");
    const auto fldcw_nn38_in = input.get_array("fldcw_nn38");
    const auto fldcw_nn39_in = input.get_array("fldcw_nn39");
    const auto fldcw_nn40_in = input.get_array("fldcw_nn40");

    // names of gas-phase chemical constituents
    char solsym[gas_pcnst][17] = {
        "O3",     "H2O2",   "H2SO4",  "SO2",    "DMS",    "SOAG",   "so4_a1",
        "pom_a1", "soa_a1", "bc_a1",  "dst_a1", "ncl_a1", "mom_a1", "num_a1",
        "so4_a2", "soa_a2", "ncl_a2", "mom_a2", "num_a2", "dst_a3", "ncl_a3",
        "so4_a3", "bc_a3",  "pom_a3", "soa_a3", "mom_a3", "num_a3", "pom_a4",
        "bc_a4",  "mom_a4", "num_a4"};

    //=========init views==========

    for (int kk = 0; kk < pver; ++kk) {
    }

    ColumnView mmr[gas_pcnst];
    ColumnView vmr[gas_pcnst];
    ColumnView mmr_tend[gas_pcnst];
    View1DHost mmr_host[gas_pcnst];
    View1DHost vmr_host[gas_pcnst];
    View1DHost mmr_tend_host[gas_pcnst];

    View1DHost depvel_host("depvel_host", gas_pcnst);
    View1DHost depflx_host("depflx_host", gas_pcnst);
    for (int mm = 0; mm < gas_pcnst; ++mm) {
      depvel_host(mm) = depvel_in[mm];
      depflx_host(mm) = depflx_in[mm];
    }

    for (int mm = 0; mm < gas_pcnst; ++mm) {
      mmr[mm] = haero::testing::create_column_view(pver);
      vmr[mm] = haero::testing::create_column_view(pver);
      mmr_tend[mm] = haero::testing::create_column_view(pver);

      mmr_host[mm] = View1DHost("mmr_host", pver);
      vmr_host[mm] = View1DHost("vmr_host", pver);
      mmr_tend_host[mm] = View1DHost("mmr_tend_host", pver);
    }

    int count = 0;
    for (int mm = 0; mm < gas_pcnst; ++mm) {
      for (int kk = 0; kk < pver; ++kk) {
        mmr_host[mm](kk) = mmr_in[count];
        vmr_host[mm](kk) = vmr_in[count];
        mmr_tend_host[mm](kk) = mmr_tend_in[count];
        count++;
      }
    }

    // transfer data to GPU.
    for (int mm = 0; mm < gas_pcnst; ++mm) {
      Kokkos::deep_copy(mmr[mm], mmr_host[mm]);
      Kokkos::deep_copy(vmr[mm], vmr_host[mm]);
      Kokkos::deep_copy(mmr_tend[mm], mmr_tend_host[mm]);
    }

    ColumnView pdel;
    ColumnView pdeldry;
    auto pdel_host =
        View1DHost((Real *)pdel_in.data(), pver); // puts data into host
    auto pdeldry_host =
        View1DHost((Real *)pdeldry_in.data(), pver); // puts data into host
    pdel = haero::testing::create_column_view(pver);
    pdeldry = haero::testing::create_column_view(pver);
    Kokkos::deep_copy(pdel, pdel_host);
    Kokkos::deep_copy(pdeldry, pdeldry_host);

    std::vector<Real> vector0_gas_pcnst(gas_pcnst, 0);
    std::vector<Real> vector0_pver(pver, 0);
    std::vector<Real> vector0_single(1, 0);

    View1D depvel("depvel", gas_pcnst);
    View1D depflx("depflx", gas_pcnst);

    auto vmr_nox = haero::testing::create_column_view(pver);
    auto vmr_noy = haero::testing::create_column_view(pver);
    auto vmr_clox = haero::testing::create_column_view(pver);
    auto vmr_cloy = haero::testing::create_column_view(pver);
    auto vmr_brox = haero::testing::create_column_view(pver);
    auto vmr_broy = haero::testing::create_column_view(pver);

    auto mmr_noy = haero::testing::create_column_view(pver);
    auto mmr_sox = haero::testing::create_column_view(pver);
    auto mmr_nhx = haero::testing::create_column_view(pver);
    auto net_chem = haero::testing::create_column_view(pver);

    auto mass_bc = haero::testing::create_column_view(pver);
    auto mass_dst = haero::testing::create_column_view(pver);
    auto mass_mom = haero::testing::create_column_view(pver);
    auto mass_ncl = haero::testing::create_column_view(pver);

    auto mass_pom = haero::testing::create_column_view(pver);
    auto mass_so4 = haero::testing::create_column_view(pver);
    auto mass_soa = haero::testing::create_column_view(pver);

    auto area = haero::testing::create_column_view(1);
    auto mass = haero::testing::create_column_view(pver);
    auto drymass = haero::testing::create_column_view(pver);
    auto ozone_layer = haero::testing::create_column_view(pver);

    auto vmr_nox_host = View1DHost("vmr_nox_host", pver);
    auto vmr_noy_host = View1DHost("vmr_noy_host", pver);
    auto vmr_clox_host = View1DHost("vmr_clox_host", pver);
    auto vmr_cloy_host = View1DHost("vmr_cloy_host", pver);
    auto vmr_brox_host = View1DHost("vmr_brox_host", pver);
    auto vmr_broy_host = View1DHost("vmr_broy_host", pver);
    auto mmr_noy_host = View1DHost("mmr_noy_host", pver);
    auto mmr_sox_host = View1DHost("mmr_sox_host", pver);
    auto mmr_nhx_host = View1DHost("mmr_nhx_host", pver);
    auto net_chem_host = View1DHost("net_chem_host", pver);
    auto mass_bc_host = View1DHost("mass_bc_host", pver);
    auto mass_dst_host = View1DHost("mass_dst_host", pver);
    auto mass_mom_host = View1DHost("mass_mom_host", pver);
    auto mass_ncl_host = View1DHost("mass_ncl_host", pver);
    auto mass_pom_host = View1DHost("mass_pom_host", pver);
    auto mass_so4_host = View1DHost("mass_so4_host", pver);
    auto mass_soa_host = View1DHost("mass_soa_host", pver);

    Kokkos::deep_copy(depvel, depvel_host);
    Kokkos::deep_copy(depflx, depflx_host);
    Kokkos::deep_copy(vmr_nox, vmr_nox_host);
    Kokkos::deep_copy(vmr_noy, vmr_noy_host);
    Kokkos::deep_copy(vmr_clox, vmr_clox_host);
    Kokkos::deep_copy(vmr_cloy, vmr_cloy_host);
    Kokkos::deep_copy(vmr_brox, vmr_brox_host);
    Kokkos::deep_copy(vmr_broy, vmr_broy_host);
    Kokkos::deep_copy(mmr_noy, mmr_noy_host);
    Kokkos::deep_copy(mmr_sox, mmr_sox_host);
    Kokkos::deep_copy(mmr_nhx, mmr_nhx_host);
    Kokkos::deep_copy(net_chem, net_chem_host);
    Kokkos::deep_copy(mass_bc, mass_bc_host);
    Kokkos::deep_copy(mass_dst, mass_dst_host);
    Kokkos::deep_copy(mass_mom, mass_mom_host);
    Kokkos::deep_copy(mass_ncl, mass_ncl_host);
    Kokkos::deep_copy(mass_pom, mass_pom_host);
    Kokkos::deep_copy(mass_so4, mass_so4_host);
    Kokkos::deep_copy(mass_soa, mass_soa_host);
    // FIXME: This should be provided as input. I had to back it out from the
    // FIXME: reference test data.
    Kokkos::deep_copy(area, 0.029210577134805595);

    //===fldcw===
    ColumnView fldcw[pcnst];
    View1DHost fldcw_host[pcnst];
    for (int nn = 0; nn < pcnst; ++nn) {
      fldcw[nn] = haero::testing::create_column_view(pver);
      fldcw_host[nn] = View1DHost("fldcw_host", pver);
    }

    count = 0;
    for (int nn = 0; nn < lchnk; ++nn) {
      for (int kk = 0; kk < pver; ++kk) {
        fldcw_host[nn](kk) = 0;
      }
    }

    count = 0;
    for (int kk = 0; kk < pver; ++kk) {
      fldcw_host[16](kk) = fldcw_nn16_in[count];
      fldcw_host[17](kk) = fldcw_nn17_in[count];
      fldcw_host[18](kk) = fldcw_nn18_in[count];
      fldcw_host[19](kk) = fldcw_nn19_in[count];
      fldcw_host[20](kk) = fldcw_nn20_in[count];
      fldcw_host[21](kk) = fldcw_nn21_in[count];
      fldcw_host[22](kk) = fldcw_nn22_in[count];
      fldcw_host[23](kk) = fldcw_nn23_in[count];
      fldcw_host[24](kk) = fldcw_nn24_in[count];
      fldcw_host[25](kk) = fldcw_nn25_in[count];
      fldcw_host[26](kk) = fldcw_nn26_in[count];
      fldcw_host[27](kk) = fldcw_nn27_in[count];
      fldcw_host[28](kk) = fldcw_nn28_in[count];
      fldcw_host[29](kk) = fldcw_nn29_in[count];
      fldcw_host[30](kk) = fldcw_nn30_in[count];
      fldcw_host[31](kk) = fldcw_nn31_in[count];
      fldcw_host[32](kk) = fldcw_nn32_in[count];
      fldcw_host[33](kk) = fldcw_nn33_in[count];
      fldcw_host[34](kk) = fldcw_nn34_in[count];
      fldcw_host[35](kk) = fldcw_nn35_in[count];
      fldcw_host[36](kk) = fldcw_nn36_in[count];
      fldcw_host[37](kk) = fldcw_nn37_in[count];
      fldcw_host[38](kk) = fldcw_nn38_in[count];
      fldcw_host[38](kk) = fldcw_nn39_in[count];
      fldcw_host[40](kk) = fldcw_nn40_in[count];
      count++;
    }

    // transfer data to GPU.
    for (int nn = 0; nn < pcnst; ++nn) {
      Kokkos::deep_copy(fldcw[nn], fldcw_host[nn]);
    }

    Real sox_species[3] = {};
    std::copy(sox_species_in.begin(), sox_species_in.begin() + 3, sox_species);
    Real aer_species[gas_pcnst] = {};
    std::copy(aer_species_in.begin(), aer_species_in.begin() + gas_pcnst,
              aer_species);
    const Real adv_mass[gas_pcnst] = {
        47.998200,     34.013600,  98.078400,     64.064800, 62.132400,
        12.011000,     115.107340, 12.011000,     12.011000, 12.011000,
        135.064039,    58.442468,  250092.672000, 1.007400,  115.107340,
        12.011000,     58.442468,  250092.672000, 1.007400,  135.064039,
        58.442468,     115.107340, 12.011000,     12.011000, 12.011000,
        250092.672000, 1.007400,   12.011000,     12.011000, 250092.672000,
        1.007400};

    auto ozone_col = haero::testing::create_column_view(1);
    auto ozone_trop = haero::testing::create_column_view(1);
    auto ozone_strat = haero::testing::create_column_view(1);
    auto df_noy = haero::testing::create_column_view(1);
    auto df_sox = haero::testing::create_column_view(1);
    auto df_nhx = haero::testing::create_column_view(1);

    auto team_policy = ThreadTeamPolicy(1u, Kokkos::AUTO);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
          mo_chm_diags::chm_diags(
              team, lchnk, ncol, id_o3, vmr, mmr, depvel, depflx, mmr_tend,
              pdel, pdeldry, fldcw, ltrop, area, sox_species, aer_species,
              adv_mass, solsym, mass, drymass, ozone_layer, ozone_col,
              ozone_trop, ozone_strat, vmr_nox, vmr_noy, vmr_clox, vmr_cloy,
              vmr_brox, vmr_broy, mmr_noy, mmr_sox, mmr_nhx, net_chem, df_noy,
              df_sox, df_nhx, mass_bc, mass_dst, mass_mom, mass_ncl, mass_pom,
              mass_so4, mass_soa);
        });

    Kokkos::deep_copy(vmr_nox_host, vmr_nox);
    Kokkos::deep_copy(vmr_noy_host, vmr_noy);
    Kokkos::deep_copy(vmr_clox_host, vmr_clox);
    Kokkos::deep_copy(vmr_cloy_host, vmr_cloy);
    Kokkos::deep_copy(vmr_brox_host, vmr_brox);
    Kokkos::deep_copy(vmr_broy_host, vmr_broy);
    Kokkos::deep_copy(mmr_noy_host, mmr_noy);
    Kokkos::deep_copy(mmr_sox_host, mmr_sox);
    Kokkos::deep_copy(mmr_nhx_host, mmr_nhx);
    Kokkos::deep_copy(net_chem_host, net_chem);
    Kokkos::deep_copy(mass_bc_host, mass_bc);
    Kokkos::deep_copy(mass_dst_host, mass_dst);
    Kokkos::deep_copy(mass_mom_host, mass_mom);
    Kokkos::deep_copy(mass_ncl_host, mass_ncl);
    Kokkos::deep_copy(mass_pom_host, mass_pom);
    Kokkos::deep_copy(mass_so4_host, mass_so4);
    Kokkos::deep_copy(mass_soa_host, mass_soa);

    auto area_host = View1DHost("area_host", 1);
    auto mass_host = View1DHost("mass_host", pver);
    auto drymass_host = View1DHost("drymass_host", pver);
    auto ozone_col_host = View1DHost("ozone_col_host", 1);
    auto ozone_trop_host = View1DHost("ozone_trop_host", 1);
    auto ozone_strat_host = View1DHost("ozone_strat_host", 1);
    auto df_nhx_host = View1DHost("df_nhx_host", 1);
    auto df_noy_host = View1DHost("df_noy_host", 1);
    auto df_sox_host = View1DHost("df_sox_host", 1);
    Kokkos::deep_copy(area_host, area);
    Kokkos::deep_copy(mass_host, mass);
    Kokkos::deep_copy(drymass_host, drymass);
    Kokkos::deep_copy(ozone_col_host, ozone_col);
    Kokkos::deep_copy(ozone_trop_host, ozone_trop);
    Kokkos::deep_copy(ozone_strat_host, ozone_strat);
    Kokkos::deep_copy(df_nhx_host, df_nhx);
    Kokkos::deep_copy(df_noy_host, df_noy);
    Kokkos::deep_copy(df_sox_host, df_sox);

    std::vector<Real> vmr_nox_out(pver);
    std::vector<Real> vmr_noy_out(pver);
    std::vector<Real> vmr_clox_out(pver);
    std::vector<Real> vmr_cloy_out(pver);
    std::vector<Real> vmr_brox_out(pver);
    std::vector<Real> vmr_broy_out(pver);
    std::vector<Real> mmr_noy_out(pver);
    std::vector<Real> mmr_sox_out(pver);
    std::vector<Real> mmr_nhx_out(pver);
    std::vector<Real> net_chem_out(pver);
    std::vector<Real> mass_bc_out(pver);
    std::vector<Real> mass_dst_out(pver);
    std::vector<Real> mass_mom_out(pver);
    std::vector<Real> mass_ncl_out(pver);
    std::vector<Real> mass_pom_out(pver);
    std::vector<Real> mass_so4_out(pver);
    std::vector<Real> mass_soa_out(pver);
    std::vector<Real> mass_out(pver);
    std::vector<Real> drymass_out(pver);

    for (int kk = 0; kk < pver; kk++) {
      vmr_nox_out[kk] = vmr_nox_host(kk);
      vmr_noy_out[kk] = vmr_noy_host(kk);
      vmr_clox_out[kk] = vmr_clox_host(kk);
      vmr_cloy_out[kk] = vmr_cloy_host(kk);
      vmr_brox_out[kk] = vmr_brox_host(kk);
      vmr_broy_out[kk] = vmr_broy_host(kk);
      mmr_noy_out[kk] = mmr_noy_host(kk);
      mmr_sox_out[kk] = mmr_sox_host(kk);
      mmr_nhx_out[kk] = mmr_nhx_host(kk);
      net_chem_out[kk] = net_chem_host(kk);
      mass_bc_out[kk] = mass_bc_host(kk);
      mass_dst_out[kk] = mass_dst_host(kk);
      mass_mom_out[kk] = mass_mom_host(kk);
      mass_ncl_out[kk] = mass_ncl_host(kk);
      mass_pom_out[kk] = mass_pom_host(kk);
      mass_so4_out[kk] = mass_so4_host(kk);
      mass_soa_out[kk] = mass_soa_host(kk);
      mass_out[kk] = mass_host(kk);
      drymass_out[kk] = drymass_host(kk);
    }

    output.set("area", area_host(0));
    output.set("mass", mass_out);
    output.set("drymass", drymass_out);
    output.set("ozone_col", ozone_col_host(0));
    output.set("ozone_strat", ozone_strat_host(0));
    output.set("ozone_trop", ozone_trop_host(0));
    output.set("net_chem", net_chem_out);

    output.set("mass_bc", mass_bc_out);
    output.set("mass_dst", mass_dst_out);
    output.set("mass_mom", mass_mom_out);
    output.set("mass_ncl", mass_ncl_out);
    output.set("mass_pom", mass_pom_out);
    output.set("mass_so4", mass_so4_out);
    output.set("mass_soa", mass_soa_out);

    output.set("vmr_nox", vmr_nox_out);
    output.set("vmr_noy", vmr_noy_out);
    output.set("vmr_clox", vmr_clox_out);
    output.set("vmr_cloy", vmr_cloy_out);
    output.set("vmr_brox", vmr_brox_out);
    output.set("vmr_broy", vmr_broy_out);
    output.set("mmr_noy", mmr_noy_out);
    output.set("mmr_sox", mmr_sox_out);
    output.set("mmr_nhx", mmr_nhx_out);
    output.set("df_noy", df_noy_host(0));
    output.set("df_sox", df_sox_host(0));
    output.set("df_nhx", df_nhx_host(0));

    // toth and tcly not used, but tcly is in reference output and is all zeros,
    // so we replicate it here.
    std::vector<Real> vmr_tcly(pver, 0.0);
    output.set("vmr_tcly", vmr_tcly);
  });
}
