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

void loadaer(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    const Real zero = 0;
    const int ntot_amode = AeroConfig::num_modes();
    const int maxd_aspectype = ndrop::maxd_aspectype;
    const int nspec_max = ndrop::nspec_max;

    const auto state_q = input.get_array("state_q");
    const Real air_density = input.get_array("cs")[0];
    const Real phase = input.get_array("phase")[0];

    Real qcldbrn[maxd_aspectype][ntot_amode] = {};

    if (input.has_array("qcldbrn1d")) {
      const auto qcldbrn1d_1d = input.get_array("qcldbrn1d");
      int count = 0;
      for (int i = 0; i < ntot_amode; ++i) {
        for (int j = 0; j < maxd_aspectype; ++j) {
          qcldbrn[j][i] = qcldbrn1d_1d[count];
          count++;
        }
      }
    }

    std::vector<Real> qcldbrn1d_num;
    if (input.has_array("qcldbrn1d")) {
      qcldbrn1d_num = input.get_array("qcldbrn1d_num");
    } else {
      qcldbrn1d_num = {zero, zero, zero, zero};
    }
    std::vector<Real> naerosol(ntot_amode, zero), vaerosol(ntot_amode, zero),
        hygro(ntot_amode, zero);
    int nspec_amode[ntot_amode];
    int lspectype_amode[maxd_aspectype][ntot_amode];
    int lmassptr_amode[maxd_aspectype][ntot_amode];
    Real specdens_amode[maxd_aspectype];
    Real spechygro[maxd_aspectype];
    int numptr_amode[ntot_amode];
    int mam_idx[ntot_amode][nspec_max];
    int mam_cnst_idx[ntot_amode][nspec_max];

    ndrop::get_e3sm_parameters(nspec_amode, lspectype_amode, lmassptr_amode,
                               numptr_amode, specdens_amode, spechygro, mam_idx,
                               mam_cnst_idx);
    Real exp45logsig[AeroConfig::num_modes()], alogsig[AeroConfig::num_modes()],
        num2vol_ratio_min_nmodes[AeroConfig::num_modes()],
        num2vol_ratio_max_nmodes[AeroConfig::num_modes()] = {};

    Real aten = zero;

    ndrop::ndrop_init(exp45logsig, alogsig, aten,
                      num2vol_ratio_min_nmodes,  // voltonumbhi_amode
                      num2vol_ratio_max_nmodes); // voltonumblo_amode

    ndrop::loadaer(state_q.data(), nspec_amode, air_density, phase,
                   lspectype_amode, specdens_amode, spechygro, lmassptr_amode,
                   num2vol_ratio_min_nmodes, num2vol_ratio_max_nmodes,
                   numptr_amode, qcldbrn, qcldbrn1d_num.data(), naerosol.data(),
                   vaerosol.data(), hygro.data());

    output.set("naerosol", naerosol);
    output.set("vaerosol", vaerosol);
    output.set("hygro", hygro);
  });
}
