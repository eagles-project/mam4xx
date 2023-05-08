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

void ccncalc(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {

#if 0    
    const Real zero = 0;
    Real scavratenum = zero;
    Real scavratevol = zero;
    const Real state_q_db = input.get_array("state_q");
    const Real tar_db = input.get_array("tair");
    const Real rhoaero = input.get_array("rhoaero")[0];
    const Real temp = input.get_array("temp")[0];
    const Real press = input.get_array("press")[0];


    ccncalc(state_q,
            tair,
            qcldbrn,
            qcldbrn_num,
            air_density,
            lspectype_amode,
            specdens_amode,
            spechygro,
            lmassptr_amode,
            voltonumbhi_amode,
            voltonumblo_amode,
            numptr_amode,
            nspec_amode,
            ccn); 


    output.set("scavratenum", std::vector(1, scavratenum));
    output.set("scavratevol", std::vector(1, scavratevol));
#endif

  });
}
