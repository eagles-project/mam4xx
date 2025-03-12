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
void set_subarea_rh(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    // Ensemble parameters
    // Declare array of strings for input names
    std::string input_arrays[] = {"ncldy_subarea", "jclea",     "jcldy",
                                  "afracsub",      "relhumgcm", "maxsubarea"};

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

    const auto ncldy_subarea_ = input.get_array("ncldy_subarea")[0];
    const auto jclea_ = input.get_array("jclea")[0];
    const auto jcldy_ = input.get_array("jcldy")[0];
    const auto afracsub_ = input.get_array("afracsub");
    const auto relhumgcm_ = input.get_array("relhumgcm")[0];

    const int ncldy_subarea = ncldy_subarea_;
    const int jclea = jclea_;
    const int jcldy = jcldy_;
    Real afracsub[subarea_max];
    const Real relhumgcm = relhumgcm_;

    for (int i = 1; i < subarea_max; ++i) {
      afracsub[i] = afracsub_[i - 1];
    }

    View1DHost relhumsub_h("relhumsub", subarea_max);
    View1D relhumsub_d("relhumsub", subarea_max);
    Kokkos::deep_copy(relhumsub_h, 0.0);
    Kokkos::deep_copy(relhumsub_d, 0.0);

    auto team_policy = ThreadTeamPolicy(1u, Kokkos::AUTO);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
          Real relhumsub[subarea_max] = {{0.0}};
          mam4::microphysics::set_subarea_rh(ncldy_subarea, jclea, jcldy,
                                             afracsub, relhumgcm, relhumsub);
          for (int i = 0; i < subarea_max; ++i) {
            relhumsub_d(i) = relhumsub[i];
          }
        });

    Kokkos::deep_copy(relhumsub_h, relhumsub_d);
    std::vector<Real> relhumsub_out;

    // NOTE: we go i = [1, 3) here due to the weird indexing convention in
    // microphysics::set_subarea_gases_and_aerosols(), which may be changed in
    // the future
    for (int i = 1; i < subarea_max; ++i) {
      relhumsub_out.push_back(relhumsub_h(i));
    }

    output.set("relhumsub", relhumsub_out);
  });
}
