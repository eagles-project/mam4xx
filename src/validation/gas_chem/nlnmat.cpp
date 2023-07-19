// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/mam4.hpp>

#include <mam4xx/aero_config.hpp>
#include <mam4xx/gas_chem.hpp>
#include <skywalker.hpp>
#include <validation.hpp>

using namespace skywalker;
using namespace mam4;
using namespace gas_chemistry;

void nlnmat(Ensemble *ensemble) {

  ensemble->process([=](const Input &input, Output &output) {
    const Real zero = 0;
    std::vector<Real> mat(nzcnt, zero);
    const auto lmat = input.get_array("lmat");
    const auto dti = input.get_array("dti")[0];
    nlnmat(mat.data(), lmat.data(), dti);

    output.set("mat", mat);
  });
}
