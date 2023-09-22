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
void binterp(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    constexpr Real zero = 0;

    const auto table_db = input.get_array("table");
    const auto ref_real = input.get_array("ref_real")[0];
    const auto ref_img = input.get_array("ref_img")[0];
    const auto ref_real_tab = input.get_array("ref_real_tab");
    const auto ref_img_tab = input.get_array("ref_img_tab");

    int itab = 0;
    int jtab = 0;
    std::vector<Real> ttab(1, zero);
    std::vector<Real> utab(1, zero);
    int itab_1 = 0;
    int ncoef = 5;
    int prefr = 7;
    int prefi = 10;
    // FIXME: reshape data from FORTRAN
    View3D table("table", ncoef, prefr, prefi);

    std::vector<Real> coef(ncoef, zero);
    // FIXME: It will not compile in CUDA.
    binterp(table, ref_real, ref_img, ref_real_tab.data(), ref_img_tab.data(),
            itab, jtab, ttab[0], utab[0], coef.data(), itab_1);

    output.set("itab", std::vector<Real>(1, itab + 1));
    output.set("jtab", std::vector<Real>(1, jtab + 1));
    output.set("ttab", ttab);
    output.set("utab", utab);
    output.set("coef", coef);
  });
}
