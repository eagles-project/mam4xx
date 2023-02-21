// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/coagulation.hpp>
#include <skywalker.hpp>
#include <validation.hpp>

using namespace skywalker;
using namespace mam4;

void coag_num_update(Ensemble *ensemble) {

  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    if (!input.has_array("deltat")) {
      std::cerr << "Required name: "
                << "deltat" << std::endl;
      exit(1);
    }
    if (!input.has_array("ybetaij0")) {
      std::cerr << "Required name: "
                << "ybetaij0" << std::endl;
      exit(1);
    }
    if (!input.has_array("ybetaii0")) {
      std::cerr << "Required name: "
                << "ybetaii0" << std::endl;
      exit(1);
    }
    if (!input.has_array("ybetajj0")) {
      std::cerr << "Required name: "
                << "ybetajj0" << std::endl;
      exit(1);
    }
    if (!input.has_array("qnum_bgn")) {
      std::cerr << "Required name: "
                << "qnum_bgn" << std::endl;
      exit(1);
    }
    if (!input.has_array("qnum_end")) {
      std::cerr << "Required name: "
                << "qnum_end" << std::endl;
      exit(1);
    }

    auto deltat = input.get_array("deltat");
    auto ybetaij0 = input.get_array("ybetaij0");
    auto ybetaii0 = input.get_array("ybetaii0");
    auto ybetajj0 = input.get_array("ybetajj0");
    auto qnum_bgn = input.get_array("qnum_bgn");
    auto qnum_end = input.get_array("qnum_end");

    std::vector<Real> qnum_tavg(AeroConfig::num_modes() + 1);
    coagulation::mam_coag_num_update(
        ybetaij0.data(), ybetaii0.data(), ybetajj0.data(), deltat[0],
        qnum_bgn.data(), qnum_end.data(), qnum_tavg.data());

    output.set("qnum_end", qnum_end);
    output.set("qnum_tavg", qnum_tavg);
  });
}