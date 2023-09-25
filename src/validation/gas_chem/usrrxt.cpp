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

void usrrxt(Ensemble *ensemble) {

  ensemble->process([=](const Input &input, Output &output) {
    const Real temperature = input.get_array("temp")[0];
    const Real mtot = input.get_array("mtot")[0];

    auto rxt = input.get_array("rxt");
    const auto invariants = input.get_array("invariants");
    const int usr_HO2_HO2_ndx = int(input.get_array("usr_HO2_HO2_ndx")[0]) - 1;
    const int usr_DMS_OH_ndx = int(input.get_array("usr_DMS_OH_ndx")[0]) - 1;
    const int usr_SO2_OH_ndx = int(input.get_array("usr_SO2_OH_ndx")[0]) - 1;
    const int inv_h2o_ndx = int(input.get_array("inv_h2o_ndx")[0]) - 1;

    usrrxt(rxt.data(), // inout
           temperature, invariants.data(), mtot, usr_HO2_HO2_ndx,
           usr_DMS_OH_ndx, usr_SO2_OH_ndx, inv_h2o_ndx);

    output.set("rxt", rxt);
  });
}
