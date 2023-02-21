// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/coagulation.hpp>
#include <skywalker.hpp>
#include <validation.hpp>

using namespace skywalker;
using namespace mam4;

void getcoags_wrapper_f(Ensemble *ensemble) {
  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    if (!input.has_array("airtemp")) {
      std::cerr << "Required name: "
                << "airtemp" << std::endl;
      exit(1);
    }
    if (!input.has_array("airprs")) {
      std::cerr << "Required name: "
                << "airprs" << std::endl;
      exit(1);
    }
    if (!input.has_array("dgatk")) {
      std::cerr << "Required name: "
                << "dgatk" << std::endl;
      exit(1);
    }
    if (!input.has_array("dgacc")) {
      std::cerr << "Required name: "
                << "dgacc" << std::endl;
      exit(1);
    }
    if (!input.has_array("sgatk")) {
      std::cerr << "Required name: "
                << "sgatk" << std::endl;
      exit(1);
    }
    if (!input.has_array("sgacc")) {
      std::cerr << "Required name: "
                << "sgacc" << std::endl;
      exit(1);
    }
    if (!input.has_array("xxlsgat")) {
      std::cerr << "Required name: "
                << "xxlsgat" << std::endl;
      exit(1);
    }
    if (!input.has_array("xxlsgac")) {
      std::cerr << "Required name: "
                << "xxlsgac" << std::endl;
      exit(1);
    }
    if (!input.has_array("pdensat")) {
      std::cerr << "Required name: "
                << "pdensat" << std::endl;
      exit(1);
    }
    if (!input.has_array("pdensac")) {
      std::cerr << "Required name: "
                << "pdensac" << std::endl;
      exit(1);
    }

    // read input data
    auto airtemp = input.get_array("airtemp");
    auto airprs = input.get_array("airprs");
    auto dgatk = input.get_array("dgatk");
    auto dgacc = input.get_array("dgacc");
    auto sgatk = input.get_array("sgatk");
    auto sgacc = input.get_array("sgacc");
    auto xxlsgat = input.get_array("xxlsgat");
    auto xxlsgac = input.get_array("xxlsgac");
    auto pdensat = input.get_array("pdensat");
    auto pdensac = input.get_array("pdensac");

    Real betaij0 = 0.0;
    Real betaij3 = 0.0;
    Real betaii0 = 0.0;
    Real betajj0 = 0.0;

    coagulation::getcoags_wrapper_f(airtemp[0], airprs[0], dgatk[0], dgacc[0],
                                    sgatk[0], sgacc[0], xxlsgat[0], xxlsgac[0],
                                    pdensat[0], pdensac[0], betaij0, betaij3,
                                    betaii0, betajj0);

    // write output data
    output.set("betaij0", betaij0);
    output.set("betaij3", betaij3);
    output.set("betaii0", betaii0);
    output.set("betajj0", betajj0);
  });
}