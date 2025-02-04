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

void modal_aero_lw(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    constexpr int pver = mam4::nlev;
    constexpr int maxd_aspectype = ndrop::maxd_aspectype;
    using View1DHost = typename HostType::view_1d<Real>;
    using View3DHost = typename HostType::view_3d<Real>;
    constexpr Real zero = 0;
    Real pblh = 1000;

    const auto dt = input.get_array("dt")[0];
    // const Real t = zero;
    const auto state_q_db = input.get_array("state_q");
    auto qqcw_db = input.get_array("qqcw"); // 2d

    View2D state_q("state_q", pver, pcnst);
    mam4::validation::convert_1d_vector_to_2d_view_device(state_q_db, state_q);

    View2D qqcw("qqcw", pver, pcnst);
    auto qqcw_host = Kokkos::create_mirror_view(qqcw);

    int count = 0;
    for (int kk = 0; kk < pver; ++kk) {
      for (int i = 0; i < pcnst; ++i) {
        qqcw_host(kk, i) = qqcw_db[count];
        count++;
      }
    }
    Kokkos::deep_copy(qqcw, qqcw_host);

    const auto temperature_db = input.get_array("temperature");
    const auto pmid_db = input.get_array("pmid");
    const auto pdel_db = input.get_array("pdel");
    const auto pdeldry_db = input.get_array("pdeldry");
    const auto cldn_db = input.get_array("cldn");

    ColumnView temperature;
    ColumnView pmid;
    ColumnView pdeldry;
    ColumnView pdel;
    ColumnView cldn;

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

    const auto specrefndxlw_real_db = input.get_array("specrefndxlw_real");
    const auto specrefndxlw_imag_db = input.get_array("specrefndxlw_imag");

    AerosolOpticsDeviceData aersol_optics_data{};
    set_complex_views_modal_aero(aersol_optics_data);

    auto specrefndxlw_host = ComplexView2D::HostMirror(
        "specrefndxlw_host", nlwbands, maxd_aspectype);

    count = 0;
    for (int j = 0; j < maxd_aspectype; ++j) {
      for (int i = 0; i < nlwbands; ++i) {
        specrefndxlw_host(i, j).real() = specrefndxlw_real_db[count];
        specrefndxlw_host(i, j).imag() = specrefndxlw_imag_db[count];
        count += 1;
      }
    }

    set_device_specrefindex(aersol_optics_data.specrefindex_lw, "long_wave",
                            specrefndxlw_host);

    const auto crefwsw_real = input.get_array("crefwsw_real");
    const auto crefwsw_imag = input.get_array("crefwsw_imag");

    const auto crefwlw_real = input.get_array("crefwlw_real");
    const auto crefwlw_imag = input.get_array("crefwlw_imag");

    auto crefwlw_host = Kokkos::create_mirror_view(aersol_optics_data.crefwlw);

    auto crefwsw_host = Kokkos::create_mirror_view(aersol_optics_data.crefwsw);

    for (int j = 0; j < nswbands; ++j) {
      crefwsw_host[j].real() = crefwsw_real[j];
      crefwsw_host[j].imag() = crefwsw_imag[j];
    }

    for (int j = 0; j < nlwbands; ++j) {
      crefwlw_host[j].real() = crefwlw_real[j];
      crefwlw_host[j].imag() = crefwlw_imag[j];
    }

    Kokkos::deep_copy(aersol_optics_data.crefwlw, crefwlw_host);
    Kokkos::deep_copy(aersol_optics_data.crefwsw, crefwsw_host);

    const auto absplw_db = input.get_array("absplw");

    printf("ntot_amode %d nlwbands %d  ", ntot_amode, nlwbands);

    View3DHost absplw3_host[ntot_amode][nlwbands];
    for (int d1 = 0; d1 < ntot_amode; ++d1)
      for (int d5 = 0; d5 < nlwbands; ++d5) {
        absplw3_host[d1][d5] =
            View3DHost("t1", coef_number, refindex_real, refindex_im);
      }

    // assuming 1d array is saved using column-major layout
    for (int d1 = 0; d1 < ntot_amode; ++d1) {
      for (int d2 = 0; d2 < coef_number; ++d2) {
        for (int d3 = 0; d3 < refindex_real; ++d3) {
          for (int d4 = 0; d4 < refindex_im; ++d4) {
            for (int d5 = 0; d5 < nlwbands; ++d5) {
              const int offset =
                  d1 +
                  ntot_amode *
                      (d2 + coef_number *
                                (d3 + refindex_real * (d4 + refindex_im * d5)));
              absplw3_host[d1][d5](d2, d3, d4) = absplw_db[offset];
            } // d5
          }   // d4
        }     // d3
      }       // d2
    }         // d1

    set_aerosol_optics_data_for_modal_aero_lw_views(aersol_optics_data);

    for (int d1 = 0; d1 < ntot_amode; ++d1)
      for (int d5 = 0; d5 < nlwbands; ++d5) {
        Kokkos::deep_copy(aersol_optics_data.absplw[d1][d5],
                          absplw3_host[d1][d5]);
      }

    const auto refrtablw_db = input.get_array("refrtablw");
    const auto refitablw_db = input.get_array("refitablw");

    int N1 = ntot_amode;
    int N2 = refindex_real;
    int N3 = nlwbands;
    View1DHost refrtablw_host[N1][N3];

    for (int d1 = 0; d1 < N1; ++d1)
      for (int d3 = 0; d3 < N3; ++d3) {
        refrtablw_host[d1][d3] = View1DHost("refrtablw", refindex_real);
      } // d3

    for (int d1 = 0; d1 < N1; ++d1)
      for (int d2 = 0; d2 < N2; ++d2)
        for (int d3 = 0; d3 < N3; ++d3) {
          const int offset = d1 + N1 * (d2 + d3 * N2);
          refrtablw_host[d1][d3](d2) = refrtablw_db[offset];

        } // d3

    for (int d1 = 0; d1 < N1; ++d1)
      for (int d3 = 0; d3 < N3; ++d3) {
        Kokkos::deep_copy(aersol_optics_data.refrtablw[d1][d3],
                          refrtablw_host[d1][d3]);
      } // d3

    N1 = ntot_amode;
    N2 = refindex_im;
    N3 = nlwbands;
    View1DHost refitablw_host[ntot_amode][nlwbands];

    for (int d1 = 0; d1 < N1; ++d1)
      for (int d3 = 0; d3 < N3; ++d3) {
        refitablw_host[d1][d3] = View1DHost("refitablw", refindex_im);
      } // d3

    for (int d1 = 0; d1 < N1; ++d1)
      for (int d2 = 0; d2 < N2; ++d2)
        for (int d3 = 0; d3 < N3; ++d3) {
          const int offset = d1 + N1 * (d2 + d3 * N2);
          refitablw_host[d1][d3](d2) = refitablw_db[offset];
        } // d3

    for (int d1 = 0; d1 < N1; ++d1)
      for (int d3 = 0; d3 < N3; ++d3) {
        Kokkos::deep_copy(aersol_optics_data.refitablw[d1][d3],
                          refitablw_host[d1][d3]);
      } // d3

    View2D tauxar("tauxar", nlwbands, pver);

    ColumnView hydrostatic_dp = create_column_view(nlev);

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

    auto height = create_column_view(nlev);
    auto interface_pressure = create_column_view(nlev + 1);
    auto &cloud_fraction = cldn;
    auto updraft_vel_ice_nucleation = create_column_view(nlev);

    auto atm = Atmosphere(nlev, temperature, pmid, vapor_mixing_ratio,
                          liquid_mixing_ratio, cloud_liquid_number_mixing_ratio,
                          ice_mixing_ratio, cloud_ice_number_mixing_ratio,
                          height, hydrostatic_dp, interface_pressure,
                          cloud_fraction, updraft_vel_ice_nucleation, pblh);

    mam4::Prognostics progs = validation::create_prognostics(nlev);

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

          modal_aero_lw(team, dt, progs_in, atm, pdel, pdeldry,
                        aersol_optics_data,
                        // outputs
                        tauxar);
          team.team_barrier();
          // 2. Let's extract state_q and qqcw from prog.
          Kokkos::parallel_for(
              Kokkos::TeamVectorRange(team, pver), [&](int kk) {
                const auto state_q_kk = ekat::subview(state_q, kk);
                const auto qqcw_kk = ekat::subview(qqcw, kk);
                utils::extract_stateq_from_prognostics(progs_in, atm,
                                                       state_q_kk.data(), kk);
                utils::extract_qqcw_from_prognostics(progs_in, qqcw_kk.data(),
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

    std::vector<Real> tauxar_out(pver * nlwbands, zero);
    mam4::validation::convert_transpose_2d_view_device_to_1d_vector(tauxar,
                                                                    tauxar_out);
    output.set("tauxar", tauxar_out);
  });
}
