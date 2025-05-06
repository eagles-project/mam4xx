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
using namespace validation;

void aer_rad_props_sw(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    using View1DHost = typename HostType::view_1d<Real>;
    using View3DHost = typename HostType::view_3d<Real>;
    constexpr Real zero = 0;
    Real pblh = 1000;

    constexpr int maxd_aspectype = ndrop::maxd_aspectype;
    constexpr int pver = mam4::nlev;

    const auto dt = input.get_array("dt")[0];
    const auto state_q_db = input.get_array("state_q");

    int count = 0;

    View2D state_q("state_q", pver, pcnst);
    mam4::validation::convert_1d_vector_to_2d_view_device(state_q_db, state_q);

    const auto zm_db = input.get_array("zm");
    const auto temperature_db = input.get_array("temperature");
    const auto pmid_db = input.get_array("pmid");
    const auto pdel_db = input.get_array("pdel");
    const auto pdeldry_db = input.get_array("pdeldry");
    const auto cldn_db = input.get_array("cldn");
    const auto zi_db = input.get_array("zi");
    const auto pint_db = input.get_array("pint");

    ColumnView zm;
    zm = haero::testing::create_column_view(pver);
    auto zm_host = View1DHost((Real *)zm_db.data(), pver);
    Kokkos::deep_copy(zm, zm_host);

    ColumnView temperature;
    ColumnView pmid;
    ColumnView pdeldry;
    ColumnView pdel;
    ColumnView cldn;
    ColumnView zi;
    ColumnView pint;

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

    zi = haero::testing::create_column_view(pver);
    auto zi_host = View1DHost((Real *)zi_db.data(), pver);
    Kokkos::deep_copy(zi, zi_host);

    pint = haero::testing::create_column_view(pver);
    auto pint_host = View1DHost((Real *)pint_db.data(), pver);
    Kokkos::deep_copy(pint, pint_host);

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

    const auto ext_cmip6_sw_db = input.get_array("ext_cmip6_sw");
    // We need to reshape ext_cmip6_sw
    View2D ext_cmip6_sw("ext_cmip6_sw", nswbands, pver);
    auto ext_cmip6_sw_host = Kokkos::create_mirror_view(ext_cmip6_sw);
    count = 0;
    for (int d1 = 0; d1 < nswbands; ++d1) {
      for (int d2 = 0; d2 < pver; ++d2) {
        // reshape  (nswbands,pver) -> (pver,nswbands)
        // unit conversion from km to m
        ext_cmip6_sw_host(d1, d2) = ext_cmip6_sw_db[count] * 1e-3;
        count++;
      }
    }

    Kokkos::deep_copy(ext_cmip6_sw, ext_cmip6_sw_host);

    const auto ssa_cmip6_sw_db = input.get_array("ssa_cmip6_sw");

    View2D ssa_cmip6_sw("ssa_cmip6_sw", pver, nswbands);
    mam4::validation::convert_1d_vector_to_2d_view_device(ssa_cmip6_sw_db,
                                                          ssa_cmip6_sw);

    const auto af_cmip6_sw_db = input.get_array("af_cmip6_sw");
    View2D af_cmip6_sw("af_cmip6_sw", pver, nswbands);
    mam4::validation::convert_1d_vector_to_2d_view_device(af_cmip6_sw_db,
                                                          af_cmip6_sw);

    AerosolOpticsDeviceData aersol_optics_data{};
    // allocate views.
    set_aerosol_optics_data_for_modal_aero_sw_views(aersol_optics_data);

    // 2D
    const auto specrefndxsw_real_db = input.get_array("specrefndxsw_real");
    const auto specrefndxsw_imag_db = input.get_array("specrefndxsw_imag");

    set_complex_views_modal_aero(aersol_optics_data);

    auto specrefndxsw_host = ComplexView2D::HostMirror(
        "specrefndxsw_host", nswbands, maxd_aspectype);
    count = 0;
    for (int j = 0; j < maxd_aspectype; ++j) {
      for (int i = 0; i < nswbands; ++i) {
        specrefndxsw_host(i, j).real() = specrefndxsw_real_db[count];
        specrefndxsw_host(i, j).imag() = specrefndxsw_imag_db[count];
        count += 1;
      }
    }

    // reshape specrefndxsw_host and copy it to device
    set_device_specrefindex(aersol_optics_data.specrefindex_sw, "short_wave",
                            specrefndxsw_host);

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
    View2D tau, tau_w, tau_w_g, tau_w_f;

    tau =
        View2D("tau", nswbands, pver + 1); // layer extinction optical depth [1]
    tau_w =
        View2D("tau_w", nswbands, pver + 1); // layer single-scatter albedo [1]
    tau_w_g = View2D("tau_w_g", nswbands, pver + 1); // asymmetry factor [1]
    tau_w_f =
        View2D("tau_w_f", nswbands, pver + 1); // forward scattered fraction [1]
    View1D output_diagnostics("output_diagnostics", 21);
    View2D output_diagnostics_amode("output_diagnostics_amode", 3, ntot_amode);

    View2D qaerwat_m("qaerwat_m", pver, ntot_amode);

    const int work_len = modal_aer_opt::get_work_len_aerosol_optics();
    View1D work("work", work_len);

    auto vapor_mixing_ratio = create_column_view(nlev);
    auto liquid_mixing_ratio = create_column_view(nlev); //
    auto ice_mixing_ratio = create_column_view(nlev);    //
    auto cloud_liquid_number_mixing_ratio = create_column_view(nlev);
    auto cloud_ice_number_mixing_ratio = create_column_view(nlev);

    // Some variables of state_q are part of atm.
    // We need deep_copy because of executation error due to different layout
    // q[0] = atm.vapor_mixing_ratio(klev);               // qv
    Kokkos::deep_copy(vapor_mixing_ratio,
                      Kokkos::subview(state_q, Kokkos::ALL(), 0));
    // q[1] = atm.liquid_mixing_ratio(klev);              // qc
    Kokkos::deep_copy(liquid_mixing_ratio,
                      Kokkos::subview(state_q, Kokkos::ALL(), 1));
    // q[2] = atm.ice_mixing_ratio(klev);                 // qi
    Kokkos::deep_copy(ice_mixing_ratio,
                      Kokkos::subview(state_q, Kokkos::ALL(), 2));
    // q[3] = atm.cloud_liquid_number_mixing_ratio(klev); //  nc
    Kokkos::deep_copy(cloud_liquid_number_mixing_ratio,
                      Kokkos::subview(state_q, Kokkos::ALL(), 3));
    // q[4] = atm.cloud_ice_number_mixing_ratio(klev);    // ni
    Kokkos::deep_copy(cloud_ice_number_mixing_ratio,
                      Kokkos::subview(state_q, Kokkos::ALL(), 4));

    auto &height = zm;
    auto &cloud_fraction = cldn;
    auto &interface_pressure = pint;
    auto &hydrostatic_dp = pdeldry;
    auto updraft_vel_ice_nucleation = create_column_view(nlev);

    auto atm = Atmosphere(nlev, temperature, pmid, vapor_mixing_ratio,
                          liquid_mixing_ratio, cloud_liquid_number_mixing_ratio,
                          ice_mixing_ratio, cloud_ice_number_mixing_ratio,
                          height, hydrostatic_dp, interface_pressure,
                          cloud_fraction, updraft_vel_ice_nucleation, pblh);

    mam4::Prognostics progs = validation::create_prognostics(nlev);

    mam4::modal_aer_opt::CalcsizeData cal_data;
    cal_data.initialize();

    auto team_policy = ThreadTeamPolicy(1u, Kokkos::AUTO);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
          static constexpr int nlev_loc = nlev;
          // we need to inject validation values to progs.
          auto progs_in = progs;
          Kokkos::parallel_for(
              Kokkos::TeamVectorRange(team, nlev_loc), [&](int kk) {
                // copy data from prog to stateq
                const auto &state_q_kk = ekat::subview(state_q, kk);
                const auto &qqcw_kk = ekat::subview(qqcw, kk);
                utils::inject_qqcw_to_prognostics(qqcw_kk.data(), progs_in, kk);
                utils::inject_stateq_to_prognostics(state_q_kk.data(), progs_in,
                                                    kk);
              });
          team.team_barrier();
          Real aodvis = 0.0;

          aer_rad_props::aer_rad_props_sw(
              team, dt, progs_in, atm, zi, pdel, ssa_cmip6_sw, af_cmip6_sw,
              ext_cmip6_sw, tau, tau_w, tau_w_g, tau_w_f,
              // FIXME
              aersol_optics_data, cal_data, aodvis, work);

          team.team_barrier();
          // 2. Let's extract state_q and qqcw from prog.
          Kokkos::parallel_for(
              Kokkos::TeamVectorRange(team, nlev_loc), [&](int kk) {
                const auto state_q_kk = ekat::subview(state_q, kk);
                const auto qqcw_kk = ekat::subview(qqcw, kk);
                utils::extract_stateq_from_prognostics(progs_in, atm,
                                                       state_q_kk, kk);

                utils::extract_qqcw_from_prognostics(progs_in, qqcw_kk,
                                                     kk);
              });
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
    std::vector<Real> tau_out(pver_po * nswbands, zero);
    mam4::validation::convert_transpose_2d_view_device_to_1d_vector(tau,
                                                                    tau_out);
    output.set("tau", tau_out);

    std::vector<Real> tau_w_out(pver_po * nswbands, zero);
    mam4::validation::convert_transpose_2d_view_device_to_1d_vector(tau_w,
                                                                    tau_w_out);
    output.set("tau_w", tau_w_out);

    std::vector<Real> tau_w_g_out(pver_po * nswbands, zero);
    mam4::validation::convert_transpose_2d_view_device_to_1d_vector(
        tau_w_g, tau_w_g_out);
    output.set("tau_w_g", tau_w_g_out);

    std::vector<Real> tau_w_f_out(pver_po * nswbands, zero);
    mam4::validation::convert_transpose_2d_view_device_to_1d_vector(
        tau_w_f, tau_w_f_out);
    output.set("tau_w_f", tau_w_f_out);
  });
}
