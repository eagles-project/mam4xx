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

void modal_aero_gravit_settling_velocity(Ensemble *ensemble) {

  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    const int nlev = 72;
    EKAT_REQUIRE_MSG(input.has_array("moment"), "Required name: moment");
    EKAT_REQUIRE_MSG(input.has_array("tair"), "Required name: tair");
    EKAT_REQUIRE_MSG(input.has_array("pmid"), "Required name: pmid");
    EKAT_REQUIRE_MSG(input.has_array("radius_part"),
                     "Required name: radius_part");
    EKAT_REQUIRE_MSG(input.has_array("density_part"),
                     "Required name: density_part");
    EKAT_REQUIRE_MSG(input.has_array("sig_part"), "Required name: sig_part");

    for (std::string s : {"moment", "radius_max"}) {
      EKAT_REQUIRE_MSG(1 == input.get_array(s).size(),
                       s + " array should be exactly 1 entry long.");
    }
    for (std::string s :
         {"tair", "pmid", "radius_part", "density_part", "sig_part"}) {
      EKAT_REQUIRE_MSG(nlev == input.get_array(s).size(),
                       s + " array should be exactly 72 entries long.");
    }

    auto moment = int(input.get_array("moment").front());
    auto radius_max = input.get_array("radius_max").front();

    auto tair = input.get_array("tair");
    auto pmid = input.get_array("pmid");
    auto radius_part = input.get_array("radius_part");
    auto density_part = input.get_array("density_part");
    auto sig_part = input.get_array("sig_part");

    auto to_dev = [](const std::vector<Real> &vec) {
      ColumnView dev = mam4::validation::create_column_view(vec.size());
      auto host = Kokkos::create_mirror_view(dev);
      for (int i = 0; i < vec.size(); ++i)
        host[i] = vec[i];
      Kokkos::deep_copy(dev, host);
      return dev;
    };
    auto tair_dev = to_dev(tair);
    auto pmid_dev = to_dev(pmid);
    auto radius_part_dev = to_dev(radius_part);
    auto density_part_dev = to_dev(density_part);
    auto sig_part_dev = to_dev(sig_part);

    ColumnView vlc_grv = mam4::validation::create_column_view(nlev);
    Kokkos::parallel_for(
        "modal_aero_depvel_part", nlev, KOKKOS_LAMBDA(int lev) {
          vlc_grv[lev] = drydep::modal_aero_gravit_settling_velocity(
              moment, radius_max, tair_dev[lev], pmid_dev[lev],
              radius_part_dev[lev], density_part_dev[lev], sig_part_dev[lev]);
        });
    auto host = Kokkos::create_mirror_view(vlc_grv);
    Kokkos::deep_copy(host, vlc_grv);
    std::vector<Real> grv(host.size());
    for (int i = 0; i < grv.size(); ++i)
      grv[i] = host[i];
    output.set("vlc_grv", grv);
  });
}
