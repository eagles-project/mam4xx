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

void modal_size_parameters(Ensemble *ensemble) {
  using mam4::nlev;
  ensemble->process([=](const Input &input, Output &output) {
    constexpr Real zero = 0;

    const Real sigma_logr_aer = input.get_array("sigma_logr_aer")[0];
    const auto dgnumwet = input.get_array("dgnumwet");
    const int ismethod2 = int(input.get_array("ismethod2")[0]);

    std::vector<Real> radsurf(nlev, zero);
    std::vector<Real> logradsurf(nlev, zero);

    std::vector<std::vector<Real>> cheb_db(nlev,
                                           std::vector<Real>(ncoef, zero));

    for (int kk = 0; kk < nlev; ++kk) {
      auto &cheb = cheb_db[kk];
      modal_size_parameters(sigma_logr_aer,
                            dgnumwet[kk], // in
                            radsurf[kk], logradsurf[kk], cheb.data(),
                            ismethod2);
    } // kk
    output.set("radsurf", radsurf);
    output.set("logradsurf", logradsurf);

    std::vector<Real> cheb_1d;
    for (int kk = 0; kk < nlev; ++kk) {
      for (int i = 0; i < ncoef; ++i) {
        cheb_1d.push_back(cheb_db[kk][i]);
      }
    } // kk

    output.set("cheb", cheb_1d);
  });
}
