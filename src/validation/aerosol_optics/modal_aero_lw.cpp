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
    constexpr int ncnst_tot = ndrop::ncnst_tot;
    using View1DHost = typename HostType::view_1d<Real>;
    constexpr Real zero = 0;

    const auto dt = input.get_array("dt")[0];
    const auto state_q_db = input.get_array("state_q");

    int count = 0;

    View2D state_q("state_q", pver, nvars);
    auto state_host = Kokkos::create_mirror_view(state_q);

    for (int i = 0; i < nvars; ++i) {
      // input data is store on the cpu.
      for (int kk = 0; kk < pver; ++kk) {
        state_host(kk, i) = state_q_db[count];
        count++;
      }
    }

    Kokkos::deep_copy(state_q, state_host);

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

    const auto qqcw_db = input.get_array("qqcw"); // 2d

    ColumnView qqcw[ncnst_tot];
    View1DHost qqcw_host[ncnst_tot];

    count = 0;
    for (int i = 0; i < ncnst_tot; ++i) {
      qqcw[i] = haero::testing::create_column_view(pver);
      qqcw_host[i] = View1DHost("qqcw_host", pver);
    }

    for (int kk = 0; kk < pver; ++kk) {
      for (int i = 0; i < ncnst_tot; ++i) {
        qqcw_host[i](kk) = qqcw_db[count];
        count++;
      }
    }

    // transfer data to GPU.
    for (int i = 0; i < ncnst_tot; ++i) {
      Kokkos::deep_copy(qqcw[i], qqcw_host[i]);
    }

    // FIXME need to set these arras
    Real sigmag_amode[ntot_amode] = {};

    const auto specrefndxlw_real_db = input.get_array("specrefndxlw_real");
    const auto specrefndxlw_imag_db = input.get_array("specrefndxlw_imag");

    ComplexView2D specrefndxlw("specrefndxlw", nswbands, maxd_aspectype);
    auto specrefndxlw_host = Kokkos::create_mirror_view(specrefndxlw);

    count = 0;
    for (int j = 0; j < maxd_aspectype; ++j) {
      for (int i = 0; i < nswbands; ++i) {
        specrefndxlw_host(i, j).real() = specrefndxlw_real_db[count];
        specrefndxlw_host(i, j).imag() = specrefndxlw_imag_db[count];
        count += 1;
      }
    }

    Kokkos::deep_copy(specrefndxlw, specrefndxlw_host);

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
    View5D absplw("absplw", ntot_amode, coef_number, refindex_real, refindex_im,
                  nlwbands);

    auto absplw_host = Kokkos::create_mirror_view(absplw);

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
              absplw_host(d1, d2, d3, d4, d5) = absplw_db[offset];
            } // d5
          }   // d4
        }     // d3
      }       // d2
    }         // d1

    View3D refrtablw, refitablw;

    const auto refrtablw_db = input.get_array("refrtablw");
    const auto refitablw_db = input.get_array("refitablw");

    refrtablw = View3D("refrtablw", ntot_amode, refindex_real, nlwbands);
    auto refrtablw_host = Kokkos::create_mirror_view(refrtablw);

    int N1 = ntot_amode;
    int N2 = refindex_real;
    int N3 = nlwbands;

    for (int d1 = 0; d1 < N1; ++d1)
      for (int d2 = 0; d2 < N2; ++d2)
        for (int d3 = 0; d3 < N3; ++d3) {
          const int offset = d1 + N1 * (d2 + d3 * N2);
          refrtablw_host(d1, d2, d3) = refrtablw_db[offset];

        } // d3

    Kokkos::deep_copy(refrtablw, refrtablw_host);

    refitablw = View3D("refitablw", ntot_amode, refindex_im, nlwbands);
    auto refitablw_host = Kokkos::create_mirror_view(refitablw);

    N1 = ntot_amode;
    N2 = refindex_im;
    N3 = nlwbands;

    for (int d1 = 0; d1 < N1; ++d1)
      for (int d2 = 0; d2 < N2; ++d2)
        for (int d3 = 0; d3 < N3; ++d3) {
          const int offset = d1 + N1 * (d2 + d3 * N2);
          refitablw_host(d1, d2, d3) = refitablw_db[offset];
        } // d3

    Kokkos::deep_copy(refitablw, refitablw_host);

    // work views
    ColumnView mass;
    ColumnView radsurf;
    ColumnView logradsurf;

    mass = haero::testing::create_column_view(pver);
    radsurf = haero::testing::create_column_view(pver);
    logradsurf = haero::testing::create_column_view(pver);

    View2D cheb("cheb", ncoef, pver);
    View2D dgnumwet_m("dgnumwet_m", pver, ntot_amode);
    View2D dgnumdry_m("dgnumdry_m", pver, ntot_amode);

    ComplexView2D specrefindex("specrefindex", max_nspec, nlwbands);
    View2D qaerwat_m("qaerwat_m", pver, ntot_amode);

    // output

    View2D tauxar("tauxar", pver, nlwbands);

    auto team_policy = ThreadTeamPolicy(1u, Kokkos::AUTO);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
          int nspec_amode[ntot_amode];
          int lspectype_amode[maxd_aspectype][ntot_amode];
          int lmassptr_amode[maxd_aspectype][ntot_amode];
          Real specdens_amode[maxd_aspectype];
          Real spechygro[maxd_aspectype];
          int numptr_amode[ntot_amode];
          int mam_idx[ntot_amode][nspec_max];
          int mam_cnst_idx[ntot_amode][nspec_max];

          get_e3sm_parameters(nspec_amode, lspectype_amode, lmassptr_amode,
                              numptr_amode, specdens_amode, spechygro, mam_idx,
                              mam_cnst_idx);

          modal_aero_lw(dt, state_q, temperature, pmid, pdel, pdeldry, cldn,
                        qqcw, tauxar,
                        // parameters
                        nspec_amode, sigmag_amode, lmassptr_amode,
                        specdens_amode, lspectype_amode, specrefndxlw, crefwlw,
                        crefwsw, absplw, refrtablw, refitablw,
                        // work views
                        mass, cheb, dgnumwet_m, dgnumdry_m, radsurf, logradsurf,
                        specrefindex, qaerwat_m);
        });

    std::vector<Real> output_qqcw;

    // transfer data to host
    for (int i = 0; i < ncnst_tot; ++i) {
      Kokkos::deep_copy(qqcw_host[i], qqcw[i]);
    }

    for (int kk = 0; kk < pver; ++kk) {
      for (int i = 0; i < ncnst_tot; ++i) {
        output_qqcw.push_back(qqcw_host[i](kk));
      }
    }

    output.set("qqcw", output_qqcw);

    std::vector<Real> tauxar_out(pver * nlwbands, zero);
    mam4::validation::convert_2d_view_device_to_1d_vector(tauxar, tauxar_out);
    output.set("tauxar", tauxar_out);
  });
}