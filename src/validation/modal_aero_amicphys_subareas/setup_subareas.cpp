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
void setup_subareas(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    // Ensemble parameters
    // Declare array of strings for input names
    std::string input_arrays[] = {};

    // Iterate over input_arrays and error if not in input
    for (std::string name : input_arrays) {
      if (!input.has_array(name.c_str())) {
        std::cerr << "Required name for array: " << name << std::endl;
        exit(1);
      }
    }

    using View1D = typename DeviceType::view_1d<Real>;
    using View1DHost = typename HostType::view_1d<Real>;

    constexpr int subarea_max = microphysics::maxsubarea();

    const Real cld = input.get_array("cld")[0];

    View1DHost nsubarea_h("nsubarea", 1);
    View1D nsubarea_d("nsubarea", 1);
    Kokkos::deep_copy(nsubarea_h, 0.0);
    Kokkos::deep_copy(nsubarea_d, 0.0);
    View1DHost ncldy_subarea_h("ncldy_subarea", 1);
    View1D ncldy_subarea_d("ncldy_subarea", 1);
    Kokkos::deep_copy(ncldy_subarea_h, 0.0);
    Kokkos::deep_copy(ncldy_subarea_d, 0.0);
    View1DHost jclea_h("jclea", 1);
    View1D jclea_d("jclea", 1);
    Kokkos::deep_copy(jclea_h, 0.0);
    Kokkos::deep_copy(jclea_d, 0.0);
    View1DHost jcldy_h("jcldy", 1);
    View1D jcldy_d("jcldy", 1);
    Kokkos::deep_copy(jcldy_h, 0.0);
    Kokkos::deep_copy(jcldy_d, 0.0);
    View1DHost iscldy_subarea_h("iscldy_subarea", subarea_max);
    View1D iscldy_subarea_d("iscldy_subarea", subarea_max);
    Kokkos::deep_copy(iscldy_subarea_h, 0.0);
    Kokkos::deep_copy(iscldy_subarea_d, 0.0);
    View1DHost afracsub_h("afracsub", subarea_max);
    View1D afracsub_d("afracsub", subarea_max);
    Kokkos::deep_copy(afracsub_h, 0.0);
    Kokkos::deep_copy(afracsub_d, 0.0);
    View1DHost fclea_h("fclea", 1);
    View1D fclea_d("fclea", 1);
    Kokkos::deep_copy(fclea_h, 0.0);
    Kokkos::deep_copy(fclea_d, 0.0);
    View1DHost fcldy_h("fcldy", 1);
    View1D fcldy_d("fcldy", 1);
    Kokkos::deep_copy(fcldy_h, 0.0);
    Kokkos::deep_copy(fcldy_d, 0.0);

    auto team_policy = ThreadTeamPolicy(1u, Kokkos::AUTO);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
          int nsubarea = 0;
          int ncldy_subarea = 0;
          int jclea = 0;
          int jcldy = 0;
          bool iscldy_subarea[subarea_max] = {0};
          Real afracsub[subarea_max] = {0};
          Real fclea = 0;
          Real fcldy = 0;
          mam4::microphysics::setup_subareas(cld, nsubarea, ncldy_subarea,
                                             jclea, jcldy, iscldy_subarea,
                                             afracsub, fclea, fcldy);
          nsubarea_d(0) = nsubarea;
          ncldy_subarea_d(0) = ncldy_subarea;
          jclea_d(0) = jclea;
          jcldy_d(0) = jcldy;
          fclea_d(0) = fclea;
          fcldy_d(0) = fcldy;
          for (int i = 0; i < subarea_max; ++i) {
            iscldy_subarea_d(i) = iscldy_subarea[i];
            afracsub_d(i) = afracsub[i];
          }
        });

    Kokkos::deep_copy(nsubarea_h, nsubarea_d);
    Kokkos::deep_copy(ncldy_subarea_h, ncldy_subarea_d);
    Kokkos::deep_copy(jclea_h, jclea_d);
    Kokkos::deep_copy(jcldy_h, jcldy_d);
    Kokkos::deep_copy(fclea_h, fclea_d);
    Kokkos::deep_copy(fcldy_h, fcldy_d);

    const int nsubarea_out = nsubarea_h(0);
    const int ncldy_subarea_out = ncldy_subarea_h(0);
    const int jclea_out = jclea_h(0);
    const int jcldy_out = jcldy_h(0);
    const Real fclea_out = fclea_h(0);
    const Real fcldy_out = fcldy_h(0);

    Kokkos::deep_copy(iscldy_subarea_h, iscldy_subarea_d);
    Kokkos::deep_copy(afracsub_h, afracsub_d);
    std::vector<Real> iscldy_subarea_out;
    std::vector<Real> afracsub_out;

    // NOTE: we go i = [1, 3) here due to the weird indexing convention in
    // microphysics::set_subarea_gases_and_aerosols(), which may be changed in
    // the future
    for (int i = 1; i < subarea_max; ++i) {
      iscldy_subarea_out.push_back(iscldy_subarea_h(i));
      afracsub_out.push_back(afracsub_h(i));
    }

    output.set("nsubarea", nsubarea_out);
    output.set("ncldy_subarea", ncldy_subarea_out);
    output.set("jclea", jclea_out);
    output.set("jcldy", jcldy_out);
    output.set("fclea", fclea_out);
    output.set("fcldy", fcldy_out);
    output.set("iscldy_subarea", iscldy_subarea_out);
    output.set("afracsub", afracsub_out);
  });
}
