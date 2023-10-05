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

void get_dtdz(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    const Real pm = input.get_array("pm")[0];
    const Real pmk = input.get_array("pmk")[0];
    const Real pmid1d_up = input.get_array("pmid1d_up")[0];
    const Real pmid1d_down = input.get_array("pmid1d_down")[0];
    const Real temp1d_up = input.get_array("temp1d_up")[0];
    const Real temp1d_down = input.get_array("temp1d_down")[0];
    //const Real cnst_kap = input.get_array("cnst_kap")[0];
    //const Real cnst_faktor = input.get_array("cnst_faktor")[0];

    Real dtdz = 0;
    Real tm = 0;

    tropopause::get_dtdz(pm, pmk, pmid1d_up, pmid1d_down, temp1d_up, temp1d_down,
                        dtdz, tm);

    output.set("dtdz", dtdz);
    output.set("tm", tm);
  });
}
