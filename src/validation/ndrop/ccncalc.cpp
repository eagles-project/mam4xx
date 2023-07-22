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
void ccncalc(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    // number of vertical points.
    // validation test from standalone ndrop.
    const Real zero = 0;
    const int maxd_aspectype = ndrop::maxd_aspectype;
    const int ntot_amode = AeroConfig::num_modes();
    const int nvars = ndrop::nvars;
    const int psat = ndrop::psat;

    const int pver = ndrop::pver;
    const auto state_q_db = input.get_array("state_q");

    const auto tair_db = input.get_array("temp");
    const auto pmid_db = input.get_array("pmid");

    const int top_lev = ndrop::top_lev;

    const int nspec_max = ndrop::nspec_max;

    using View2D = ndrop::View2D;
    using View1DHost = typename HostType::view_1d<Real>;

    View2D state_q("state_q", pver, nvars);
    auto state_host = Kokkos::create_mirror_view(state_q);

    int count = 0;
    for (int i = 0; i < nvars; ++i) {
      // input data is store on the cpu.
      for (int kk = 0; kk < pver; ++kk) {
        state_host(kk, i) = state_q_db[count];
        count++;
      }
    }

    Kokkos::deep_copy(state_q, state_host);

    ColumnView tair;
    ColumnView pmid;
    tair = haero::testing::create_column_view(pver);
    pmid = haero::testing::create_column_view(pver);

    auto tair_host = View1DHost((Real *)tair_db.data(), pver);
    auto pmid_host = View1DHost((Real *)pmid_db.data(), pver);

    Kokkos::deep_copy(tair, tair_host);
    Kokkos::deep_copy(pmid, pmid_host);

    View2D ccn("ccn", pver, psat);

    Kokkos::parallel_for(
        "ccncalc", pver - top_lev, KOKKOS_LAMBDA(int k) {
          Real qcldbrn[maxd_aspectype][ntot_amode] = {{zero}};
          Real qcldbrn_num[ntot_amode] = {zero};

          const int kk = k + top_lev;
          const auto state_q_k = Kokkos::subview(state_q, kk, Kokkos::ALL());
          const Real air_density =
              conversions::density_of_ideal_gas(tair(kk), pmid(kk));

          int nspec_amode[ntot_amode];
          int lspectype_amode[maxd_aspectype][ntot_amode];
          int lmassptr_amode[maxd_aspectype][ntot_amode];
          Real specdens_amode[maxd_aspectype];
          Real spechygro[maxd_aspectype];
          int numptr_amode[ntot_amode];
          int mam_idx[ntot_amode][nspec_max];
          int mam_cnst_idx[ntot_amode][nspec_max];

          ndrop::get_e3sm_parameters(
              nspec_amode, lspectype_amode, lmassptr_amode, numptr_amode,
              specdens_amode, spechygro, mam_idx, mam_cnst_idx);

          Real exp45logsig[AeroConfig::num_modes()],
              alogsig[AeroConfig::num_modes()],
              num2vol_ratio_min_nmodes[AeroConfig::num_modes()],
              num2vol_ratio_max_nmodes[AeroConfig::num_modes()] = {};

          Real aten = zero;

          ndrop::ndrop_init(exp45logsig, alogsig, aten,
                            num2vol_ratio_min_nmodes,  // voltonumbhi_amode
                            num2vol_ratio_max_nmodes); // voltonumblo_amode

          const auto ccn_k = Kokkos::subview(ccn, kk, Kokkos::ALL());
          ndrop::ccncalc(state_q_k.data(), tair(kk), qcldbrn, qcldbrn_num,
                         air_density, lspectype_amode, specdens_amode,
                         spechygro, lmassptr_amode, num2vol_ratio_min_nmodes,
                         num2vol_ratio_max_nmodes, numptr_amode, nspec_amode,
                         exp45logsig, alogsig, ccn_k.data());
        });

    auto ccn_host = Kokkos::create_mirror_view(ccn);
    Kokkos::deep_copy(ccn_host, ccn);
    for (int i = 0; i < psat; ++i) {
      std::vector<Real> ccn_v(pver);
      for (int kk = 0; kk < pver; ++kk) {
        ccn_v[kk] = ccn_host(kk, i);
      }
      output.set("ccn_" + std::to_string(i + 1), ccn_v);
    }
  });
}
