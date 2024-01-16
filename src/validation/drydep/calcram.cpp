// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <iostream>
#include <mam4xx/drydep.hpp>
#include <skywalker.hpp>
#include <validation.hpp>

using namespace skywalker;
using namespace mam4;

void calcram(Ensemble *ensemble) {

  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    EKAT_REQUIRE_MSG(input.has_array("landfrac"), "Required name: landfrac");
    EKAT_REQUIRE_MSG(input.has_array("icefrac"), "Required name: icefrac");
    EKAT_REQUIRE_MSG(input.has_array("ocnfrac"), "Required name: ocnfrac");
    EKAT_REQUIRE_MSG(input.has_array("obklen"), "Required name: obklen");
    EKAT_REQUIRE_MSG(input.has_array("ustar"), "Required name: ustar");
    EKAT_REQUIRE_MSG(input.has_array("ram1_in"), "Required name: ram1_in");
    EKAT_REQUIRE_MSG(input.has_array("tair"), "Required name: tair");
    EKAT_REQUIRE_MSG(input.has_array("pmid"), "Required name: pmid");
    EKAT_REQUIRE_MSG(input.has_array("pdel"), "Required name: pdel");
    EKAT_REQUIRE_MSG(input.has_array("fv_in"), "Required name: fv_in");

    for (std::string s : {"landfrac", "icefrac", "ocnfrac", "obklen", "ustar",
                          "ram1_in", "tair", "pmid", "pdel", "fv_in"}) {
      EKAT_REQUIRE_MSG(1 == input.get_array(s).size(),
                       s + " array should be exactly 1 entry long.");
    }
    const Real landfrac = input.get_array("landfrac").front();
    const Real icefrac = input.get_array("icefrac").front();
    const Real ocnfrac = input.get_array("ocnfrac").front();
    const Real obklen = input.get_array("obklen").front();
    const Real ustar = input.get_array("ustar").front();
    const Real ram1_in = input.get_array("ram1_in").front();
    const Real tair = input.get_array("tair").front();
    const Real pmid = input.get_array("pmid").front();
    const Real pdel = input.get_array("pdel").front();
    const Real fv_in = input.get_array("fv_in").front();

    typedef typename Kokkos::MinMax<Real>::value_type Calcram;
    Calcram cal_cram;
    Kokkos::parallel_reduce(
        1,
        KOKKOS_LAMBDA(const int, Calcram &vlc) {
          Real ram1_out, fv_out;
          drydep::calcram(landfrac, icefrac, ocnfrac, obklen, ustar, tair, pmid,
                          pdel, ram1_in, fv_in, ram1_out, fv_out);
          vlc.min_val = ram1_out;
          vlc.max_val = fv_out;
        },
        Kokkos::MinMax<Real>(cal_cram));

    output.set("ram1_out", cal_cram.min_val);
    output.set("fv_out", cal_cram.max_val);
  });
}
