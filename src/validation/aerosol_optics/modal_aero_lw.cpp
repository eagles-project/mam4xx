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

void modal_aero_lw(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    constexpr int pver = mam4::nlev;
    constexpr int maxd_aspectype = ndrop::maxd_aspectype;
    using View1DHost = typename HostType::view_1d<Real>;
    using View3DHost = typename HostType::view_3d<Real>;
    constexpr Real zero = 0;

    const auto dt = input.get_array("dt")[0];
    // const Real t = zero;
    const auto state_q_db = input.get_array("state_q");
    const auto qqcw_db = input.get_array("qqcw"); // 2d

    int count = 0;

    View2D state_q("state_q", pver, nvars);
    mam4::validation::convert_1d_vector_to_2d_view_device(state_q_db, state_q);

    View2D qqcw("qqcw", pver, pcnst);
    auto qqcw_host = Kokkos::create_mirror_view(qqcw);
    count = 0;

    for (int i = 0; i < pcnst; ++i) {
      // input data is store on the cpu.
      for (int kk = 0; kk < pver; ++kk) {
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

    const auto sigmag_amode_db = input.get_array("sigmag_amode");

    Real sigmag_amode[ntot_amode] = {};

    for (int imode = 0; imode < ntot_amode; ++imode) {
      sigmag_amode[imode] = sigmag_amode_db[imode];
    }

    const auto specrefndxlw_real_db = input.get_array("specrefndxlw_real");
    const auto specrefndxlw_imag_db = input.get_array("specrefndxlw_imag");

    AerosolOpticsDeviceData aersol_optics_data{};
    set_complex_views_modal_aero(aersol_optics_data);

    auto specrefndxlw_host =
        Kokkos::create_mirror_view(aersol_optics_data.specrefndxlw);

    count = 0;
    for (int j = 0; j < maxd_aspectype; ++j) {
      for (int i = 0; i < nlwbands; ++i) {
        specrefndxlw_host(i, j).real() = specrefndxlw_real_db[count];
        specrefndxlw_host(i, j).imag() = specrefndxlw_imag_db[count];
        count += 1;
      }
    }

    Kokkos::deep_copy(aersol_optics_data.specrefndxlw, specrefndxlw_host);

    const auto crefwsw_real = input.get_array("crefwsw_real");
    const auto crefwsw_imag = input.get_array("crefwsw_imag");

    const auto crefwlw_real = input.get_array("crefwlw_real");
    const auto crefwlw_imag = input.get_array("crefwlw_imag");

    Kokkos::complex<Real> crefwlw[nlwbands];

    Kokkos::complex<Real> crefwsw[nswbands];

    for (int j = 0; j < nswbands; ++j) {
      crefwsw[j].real() = crefwsw_real[j];
      crefwsw[j].imag() = crefwsw_imag[j];
    }

    for (int j = 0; j < nlwbands; ++j) {
      crefwlw[j].real() = crefwlw_real[j];
      crefwlw[j].imag() = crefwlw_imag[j];
    }

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

    // work views
    ComplexView2D specrefindex("specrefindex", max_nspec, nlwbands);

    const int wlen = get_worksize_modal_aero_lw();
    View1D work("work", wlen);

    View2D tauxar("tauxar", pver, nlwbands);
    auto team_policy = ThreadTeamPolicy(1u, Kokkos::AUTO);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {

          modal_aero_lw(dt, state_q, qqcw,
                        temperature, pmid,
                        pdel, pdeldry, cldn,
                        aersol_optics_data,
                        tauxar,
                        // parameters
                        // work views
                        specrefindex, work);
        });
    std::vector<Real> qqcw_out(pver * pcnst, zero);
    mam4::validation::convert_2d_view_device_to_1d_vector(qqcw, qqcw_out);

    output.set("qqcw", qqcw_out);

    std::vector<Real> tauxar_out(pver * nlwbands, zero);
    mam4::validation::convert_2d_view_device_to_1d_vector(tauxar, tauxar_out);
    output.set("tauxar", tauxar_out);
  });
}