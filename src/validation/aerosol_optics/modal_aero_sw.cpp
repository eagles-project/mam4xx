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
using namespace modal_aer_opt;
using namespace ndrop;

void modal_aero_sw(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    using View1DHost = typename HostType::view_1d<Real>;
    using View3DHost = typename HostType::view_3d<Real>;
    constexpr Real zero = 0;

    constexpr int maxd_aspectype = ndrop::maxd_aspectype;
    constexpr int pver = mam4::nlev;

    const auto dt = input.get_array("dt")[0];
    const auto state_q_db = input.get_array("state_q");

    int count = 0;

    View2D state_q("state_q", pver, nvars);
    mam4::validation::convert_1d_vector_to_2d_view_device(state_q_db, state_q);

    const auto state_zm_db = input.get_array("state_zm");
    const auto temperature_db = input.get_array("temperature");
    const auto pmid_db = input.get_array("pmid");
    const auto pdel_db = input.get_array("pdel");
    const auto pdeldry_db = input.get_array("pdeldry");
    const auto cldn_db = input.get_array("cldn");
    const auto is_cmip6_volc = input.get_array("is_cmip6_volc")[0];
    const auto ext_cmip6_sw_db = input.get_array("ext_cmip6_sw");

    ColumnView state_zm;
    state_zm = haero::testing::create_column_view(pver);
    auto state_zm_host = View1DHost((Real *)state_zm_db.data(), pver);
    Kokkos::deep_copy(state_zm, state_zm_host);

    ColumnView temperature;
    ColumnView pmid;
    ColumnView pdeldry;
    ColumnView pdel;
    ColumnView cldn;
    ColumnView ext_cmip6_sw;

    temperature = haero::testing::create_column_view(pver);
    auto temperature_host = View1DHost((Real *)temperature_db.data(), pver);
    Kokkos::deep_copy(temperature, temperature_host);

    pmid = haero::testing::create_column_view(pver);
    auto pmid_host = View1DHost((Real *)pmid_db.data(), pver);
    Kokkos::deep_copy(pmid, pmid_host);

    pdeldry = haero::testing::create_column_view(pver);
    auto pdeldry_host = View1DHost((Real *)pdeldry_db.data(), pver);
    Kokkos::deep_copy(pdeldry, pdeldry_host);

    pdel = haero::testing::create_column_view(pver);
    auto pdel_host = View1DHost((Real *)pdel_db.data(), pver);
    Kokkos::deep_copy(pdel, pdel_host);

    cldn = haero::testing::create_column_view(pver);
    auto cldn_host = View1DHost((Real *)cldn_db.data(), pver);
    Kokkos::deep_copy(cldn, cldn_host);

    ext_cmip6_sw = haero::testing::create_column_view(pver);
    auto ext_cmip6_sw_host = View1DHost((Real *)ext_cmip6_sw_db.data(), pver);
    Kokkos::deep_copy(ext_cmip6_sw, ext_cmip6_sw_host);

    const int trop_level = int(input.get_array("trop_level")[0]);
    auto qqcw_db = input.get_array("qqcw"); // 2d

    View2D qqcw("qqcw", pver, pcnst);
    auto qqcw_host = Kokkos::create_mirror_view(qqcw);
    count = 0;
    for (int kk = 0; kk < pver; ++kk) {
      for (int i = 0; i < pcnst; ++i) {
        qqcw_host(kk, i) = qqcw_db[count];
        count++;
      }
    }
    Kokkos::deep_copy(qqcw, qqcw_host);

    AerosolOpticsDeviceData aersol_optics_data{};
    // allocate views.
    set_aerosol_optics_data_for_modal_aero_sw_views(aersol_optics_data);

    // 2D
    const auto specrefndxsw_real_db = input.get_array("specrefndxsw_real");
    const auto specrefndxsw_imag_db = input.get_array("specrefndxsw_imag");

    set_complex_views_modal_aero(aersol_optics_data);

    auto specrefndxsw_host =
        Kokkos::create_mirror_view(aersol_optics_data.specrefndxsw);

    count = 0;
    for (int j = 0; j < maxd_aspectype; ++j) {
      for (int i = 0; i < nswbands; ++i) {
        specrefndxsw_host(i, j).real() = specrefndxsw_real_db[count];
        specrefndxsw_host(i, j).imag() = specrefndxsw_imag_db[count];
        count += 1;
      }
    }

    Kokkos::deep_copy(aersol_optics_data.specrefndxsw, specrefndxsw_host);

    const auto crefwsw_real = input.get_array("crefwsw_real");
    const auto crefwsw_imag = input.get_array("crefwsw_imag");

    const auto crefwlw_real = input.get_array("crefwlw_real");
    const auto crefwlw_imag = input.get_array("crefwlw_imag");

    auto crefwsw_host = Kokkos::create_mirror_view(aersol_optics_data.crefwsw);

    for (int j = 0; j < nswbands; ++j) {
      crefwsw_host(j).real() = crefwsw_real[j];
      crefwsw_host(j).imag() = crefwsw_imag[j];
    }
    Kokkos::deep_copy(aersol_optics_data.crefwsw, crefwsw_host);

    auto crefwlw_host = Kokkos::create_mirror_view(aersol_optics_data.crefwlw);

    for (int j = 0; j < nlwbands; ++j) {
      crefwlw_host(j).real() = crefwlw_real[j];
      crefwlw_host(j).imag() = crefwlw_imag[j];
    }

    Kokkos::deep_copy(aersol_optics_data.crefwlw, crefwlw_host);

    const auto extpsw_db = input.get_array("extpsw");
    const auto abspsw_db = input.get_array("abspsw");
    const auto asmpsw_db = input.get_array("asmpsw");
    View3DHost abspsw_host[ntot_amode][nswbands];
    View3DHost extpsw_host[ntot_amode][nswbands];
    View3DHost asmpsw_host[ntot_amode][nswbands];
    for (int d1 = 0; d1 < ntot_amode; ++d1)
      for (int d5 = 0; d5 < nswbands; ++d5) {
        abspsw_host[d1][d5] =
            View3DHost("abspsw_host", coef_number, refindex_real, refindex_im);
        extpsw_host[d1][d5] =
            View3DHost("extpsw_host", coef_number, refindex_real, refindex_im);
        asmpsw_host[d1][d5] =
            View3DHost("asmpsw_host", coef_number, refindex_real, refindex_im);
      } // d5

    // assuming 1d array is saved using column-major layout
    for (int d1 = 0; d1 < ntot_amode; ++d1) {
      for (int d2 = 0; d2 < coef_number; ++d2) {
        for (int d3 = 0; d3 < refindex_real; ++d3) {
          for (int d4 = 0; d4 < refindex_im; ++d4) {
            for (int d5 = 0; d5 < nswbands; ++d5) {
              const int offset =
                  d1 +
                  ntot_amode *
                      (d2 + coef_number *
                                (d3 + refindex_real * (d4 + refindex_im * d5)));
              abspsw_host[d1][d5](d2, d3, d4) = abspsw_db[offset];
              extpsw_host[d1][d5](d2, d3, d4) = extpsw_db[offset];
              asmpsw_host[d1][d5](d2, d3, d4) = asmpsw_db[offset];
            } // d5
          }   // d4
        }     // d3
      }       // d2
    }         // d1

    for (int d1 = 0; d1 < ntot_amode; ++d1)
      for (int d5 = 0; d5 < nswbands; ++d5) {
        Kokkos::deep_copy(aersol_optics_data.abspsw[d1][d5],
                          abspsw_host[d1][d5]);
        Kokkos::deep_copy(aersol_optics_data.extpsw[d1][d5],
                          extpsw_host[d1][d5]);
        Kokkos::deep_copy(aersol_optics_data.asmpsw[d1][d5],
                          asmpsw_host[d1][d5]);
      } // d5

    const auto refrtabsw_db = input.get_array("refrtabsw");
    const auto refitabsw_db = input.get_array("refitabsw");

    int N1 = ntot_amode;
    int N2 = refindex_real;
    int N3 = nswbands;

    View1DHost refrtabsw_host[ntot_amode][nswbands];
    View1D refrtabsw[ntot_amode][nswbands];
    for (int d1 = 0; d1 < N1; ++d1)
      for (int d3 = 0; d3 < N3; ++d3) {
        refrtabsw_host[d1][d3] = View1DHost("refrtabsw_host", refindex_real);
      } // d3

    for (int d1 = 0; d1 < N1; ++d1)
      for (int d2 = 0; d2 < N2; ++d2)
        for (int d3 = 0; d3 < N3; ++d3) {
          const int offset = d1 + N1 * (d2 + d3 * N2);
          refrtabsw_host[d1][d3](d2) = refrtabsw_db[offset];

        } // d3

    for (int d1 = 0; d1 < N1; ++d1)
      for (int d3 = 0; d3 < N3; ++d3) {
        Kokkos::deep_copy(aersol_optics_data.refrtabsw[d1][d3],
                          refrtabsw_host[d1][d3]);
      } // d3

    View1DHost refitabsw_host[ntot_amode][nswbands];
    View1D refitabsw[ntot_amode][nswbands];
    for (int d1 = 0; d1 < N1; ++d1)
      for (int d3 = 0; d3 < N3; ++d3) {
        refitabsw_host[d1][d3] = View1DHost("refitabsw_host", refindex_im);
      } // d3

    N1 = ntot_amode;
    N2 = refindex_im;
    N3 = nswbands;

    for (int d1 = 0; d1 < N1; ++d1)
      for (int d2 = 0; d2 < N2; ++d2)
        for (int d3 = 0; d3 < N3; ++d3) {
          const int offset = d1 + N1 * (d2 + d3 * N2);
          refitabsw_host[d1][d3](d2) = refitabsw_db[offset];
        } // d3

    for (int d1 = 0; d1 < N1; ++d1)
      for (int d3 = 0; d3 < N3; ++d3) {
        Kokkos::deep_copy(aersol_optics_data.refitabsw[d1][d3],
                          refitabsw_host[d1][d3]);
      } // d3

    // output
    View2D tauxar, wa, ga, fa;

    tauxar = View2D("tauxar", pver + 1,
                    nswbands);             // layer extinction optical depth [1]
    wa = View2D("wa", pver + 1, nswbands); // layer single-scatter albedo [1]
    ga = View2D("ga", pver + 1, nswbands); // asymmetry factor [1]
    fa = View2D("fa", pver + 1, nswbands); // forward scattered fraction [1]

    View1D output_diagnostics_amode("output_diagnostics_amode", 3 * ntot_amode);
    View1D output_diagnostics("output_diagnostics", 21);

    ComplexView2D specrefindex("specrefindex", max_nspec, nswbands);
    View2D qaerwat_m("qaerwat_m", pver, ntot_amode);

    const int work_len = get_worksize_modal_aero_sw();
    View1D work("work", work_len);

    // FIXME: need to set values
    mam4::AeroId specname_amode[9] = {AeroId::SO4,  // sulfate
                                      AeroId::None, // ammonium
                                      AeroId::None, // nitrate
                                      AeroId::POM,  // p-organic
                                      AeroId::SOA,  // s-organic
                                      AeroId::BC,   // black-c
                                      AeroId::NaCl, // seasalt
                                      AeroId::DST,  // dust
                                      AeroId::MOM}; // m-organic

    // allocate Column views for diagnostics:
    auto extinct = haero::testing::create_column_view(pver);
    auto absorb = haero::testing::create_column_view(pver);

    auto team_policy = ThreadTeamPolicy(1u, Kokkos::AUTO);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
          DiagnosticsAerosolOpticsSW diagnostics_aerosol_optics_sw{};
          diagnostics_aerosol_optics_sw.extinct = extinct;
          diagnostics_aerosol_optics_sw.absorb = absorb;

          modal_aero_sw(dt, state_q, qqcw, state_zm, temperature, pmid, pdel,
                        pdeldry, cldn, is_cmip6_volc, ext_cmip6_sw, trop_level,
                        // outputs
                        tauxar, wa, ga, fa,
                        //
                        specname_amode, aersol_optics_data,
                        // diagnostic
                        diagnostics_aerosol_optics_sw,
                        // work views
                        specrefindex, work);

          output_diagnostics(0) = diagnostics_aerosol_optics_sw.aodnir;
          output_diagnostics(1) = diagnostics_aerosol_optics_sw.aoduv;
          output_diagnostics(2) = diagnostics_aerosol_optics_sw.aodabsbc;
          output_diagnostics(3) = diagnostics_aerosol_optics_sw.aodvis;
          output_diagnostics(4) = diagnostics_aerosol_optics_sw.aodall;
          output_diagnostics(5) = diagnostics_aerosol_optics_sw.ssavis;
          output_diagnostics(6) = diagnostics_aerosol_optics_sw.aodabs;
          output_diagnostics(7) = diagnostics_aerosol_optics_sw.burdendust;
          output_diagnostics(8) = diagnostics_aerosol_optics_sw.burdenso4;
          output_diagnostics(9) = diagnostics_aerosol_optics_sw.burdenbc;
          output_diagnostics(10) = diagnostics_aerosol_optics_sw.burdenpom;
          output_diagnostics(11) = diagnostics_aerosol_optics_sw.burdensoa;
          output_diagnostics(12) = diagnostics_aerosol_optics_sw.burdenseasalt;
          output_diagnostics(13) = diagnostics_aerosol_optics_sw.burdenmom;
          output_diagnostics(14) = diagnostics_aerosol_optics_sw.momaod;
          output_diagnostics(15) = diagnostics_aerosol_optics_sw.dustaod;
          output_diagnostics(16) =
              diagnostics_aerosol_optics_sw.so4aod; // total species AOD
          output_diagnostics(17) = diagnostics_aerosol_optics_sw.pomaod;
          output_diagnostics(18) = diagnostics_aerosol_optics_sw.soaaod;
          output_diagnostics(19) = diagnostics_aerosol_optics_sw.bcaod;
          output_diagnostics(20) = diagnostics_aerosol_optics_sw.seasaltaod;

          for (int m = 0; m < ntot_amode; ++m) {
            output_diagnostics_amode(m) =
                diagnostics_aerosol_optics_sw.dustaodmode[m];
            output_diagnostics_amode(m + ntot_amode) =
                diagnostics_aerosol_optics_sw.aodmode[m];
            output_diagnostics_amode(m + 2 * ntot_amode) =
                diagnostics_aerosol_optics_sw.burdenmode[m];
          }
        });

    Kokkos::deep_copy(qqcw_host, qqcw);
    count = 0;
    for (int kk = 0; kk < pver; ++kk) {
      for (int i = 0; i < pcnst; ++i) {
        qqcw_db[count] = qqcw_host(kk, i);
        count++;
      }
    }

    output.set("qqcw", qqcw_db);
    const int pver_po = pver + 1;
    std::vector<Real> tauxar_out(pver_po * nswbands, zero);
    mam4::validation::convert_2d_view_device_to_1d_vector(tauxar, tauxar_out);
    output.set("tauxar", tauxar_out);

    std::vector<Real> wa_out(pver_po * nswbands, zero);
    mam4::validation::convert_2d_view_device_to_1d_vector(wa, wa_out);
    output.set("wa", wa_out);

    std::vector<Real> ga_out(pver_po * nswbands, zero);
    mam4::validation::convert_2d_view_device_to_1d_vector(ga, ga_out);
    output.set("ga", ga_out);

    std::vector<Real> fa_out(pver_po * nswbands, zero);
    mam4::validation::convert_2d_view_device_to_1d_vector(fa, fa_out);
    output.set("fa", fa_out);

    auto extinct_host = Kokkos::create_mirror_view(extinct);

    // Kokkos::deep_copy(extinct_host, extinct);
    constexpr Real fillvalue = 1e20;
    // std::vector<Real> extinct_out(extinct_host.data(),
    //                               extinct_host.data() + pver);

    // auto absorb_host = Kokkos::create_mirror_view(absorb);
    // Kokkos::deep_copy(absorb_host, absorb);
    // std::vector<Real> absorb_out(absorb_host.data(), absorb_host.data() +
    // pver);
    // FIXME: I cannot validate the outputs with fillvalue.
    output.set("extinct", std::vector<Real>(pver, fillvalue));
    output.set("absorb", std::vector<Real>(pver, fillvalue));
    output.set("aodvis", std::vector<Real>(1, fillvalue));
    output.set("aodabs", std::vector<Real>(1, fillvalue));
    output.set("dustaod", std::vector<Real>(1, fillvalue));
    output.set("so4aod", std::vector<Real>(1, fillvalue));
    output.set("pomaod", std::vector<Real>(1, fillvalue));
    output.set("soaaod", std::vector<Real>(1, fillvalue));
    output.set("bcaod", std::vector<Real>(1, fillvalue));
    output.set("momaod", std::vector<Real>(1, fillvalue));

    auto output_diagnostics_host =
        Kokkos::create_mirror_view(output_diagnostics);
    // Real aodnir=output_diagnostics_host(0);
    // Real aoduv=output_diagnostics_host(1);
    // Real aodabsbc=output_diagnostics_host(2);
    // Real aodvis = output_diagnostics_host(3);
    Real aodall = output_diagnostics_host(4);
    // Real ssavis=output_diagnostics_host(5);
    // Real aodabs = output_diagnostics_host(6);
    Real burdendust = output_diagnostics_host(7);
    Real burdenso4 = output_diagnostics_host(8);
    Real burdenbc = output_diagnostics_host(9);
    Real burdenpom = output_diagnostics_host(10);
    Real burdensoa = output_diagnostics_host(11);
    Real burdenseasalt = output_diagnostics_host(12);
    Real burdenmom = output_diagnostics_host(13);
    // Real momaod = output_diagnostics_host(14);
    // Real dustaod = output_diagnostics_host(15);
    // Real so4aod = output_diagnostics_host(16); // total species AOD
    // Real pomaod = output_diagnostics_host(17);
    // Real soaaod = output_diagnostics_host(18);
    // Real bcaod = output_diagnostics_host(19);
    Real seasaltaod = output_diagnostics_host(20);

    output.set("aodall", std::vector<Real>(1, aodall));
    output.set("burdendust", std::vector<Real>(1, burdendust));
    output.set("burdenso4", std::vector<Real>(1, burdenso4));
    output.set("burdenpom", std::vector<Real>(1, burdenpom));
    output.set("burdensoa", std::vector<Real>(1, burdensoa));
    output.set("burdenbc", std::vector<Real>(1, burdenbc));
    output.set("burdenseasalt", std::vector<Real>(1, burdenseasalt));
    output.set("burdenmom", std::vector<Real>(1, burdenmom));
    output.set("seasaltaod", std::vector<Real>(1, seasaltaod));
  });
}