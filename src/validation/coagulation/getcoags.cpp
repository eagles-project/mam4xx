// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/coagulation.hpp>
#include <skywalker.hpp>
#include <validation.hpp>

using namespace skywalker;
using namespace mam4;

void getcoags(Ensemble *ensemble) {

  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    if (!input.has_array("lamda")) {
      std::cerr << "Required name: "
                << "lamda" << std::endl;
      exit(1);
    }
    if (!input.has_array("kfmat")) {
      std::cerr << "Required name: "
                << "kfmat" << std::endl;
      exit(1);
    }
    if (!input.has_array("kfmac")) {
      std::cerr << "Required name: "
                << "kfmac" << std::endl;
      exit(1);
    }
    if (!input.has_array("kfmatac")) {
      std::cerr << "Required name: "
                << "kfmatac" << std::endl;
      exit(1);
    }
    if (!input.has_array("knc")) {
      std::cerr << "Required name: "
                << "knc" << std::endl;
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

    // read input data
    auto lamdaa = input.get_array("lamda");
    auto kfmat = input.get_array("kfmat");
    auto kfmac = input.get_array("kfmac");
    auto kfmatac = input.get_array("kfmatac");
    auto knc = input.get_array("knc");
    auto dgatk = input.get_array("dgatk");
    auto dgacc = input.get_array("dgacc");
    auto sgatk = input.get_array("sgatk");
    auto sgacc = input.get_array("sgacc");
    auto xxlsgat = input.get_array("xxlsgat");
    auto xxlsgac = input.get_array("xxlsgac");

    Real qn11 = 0.0;
    Real qn22 = 0.0;
    Real qn12 = 0.0;
    Real qv12 = 0.0;

    coagulation::getcoags(lamdaa[0], kfmatac[0], kfmat[0], kfmac[0], knc[0],
                          dgatk[0], dgacc[0], sgatk[0], sgacc[0], xxlsgat[0],
                          xxlsgac[0], qn11, qn22, qn12, qv12);

    output.set("qn11", qn11);
    output.set("qn22", qn22);
    output.set("qn12", qn12);
    output.set("qv12", qv12);
  });
}