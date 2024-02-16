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

void modal_aero_turb_drydep_velocity(Ensemble *ensemble) {

  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    EKAT_REQUIRE_MSG(input.has_array("moment"), "Required name: moment");
    EKAT_REQUIRE_MSG(input.has_array("fraction_landuse"),
                     "Required name: fraction_landuse");
    EKAT_REQUIRE_MSG(input.has_array("radius_max"),
                     "Required name: radius_max");
    EKAT_REQUIRE_MSG(input.has_array("tair"), "Required name: tair");
    EKAT_REQUIRE_MSG(input.has_array("pmid"), "Required name: pmid");
    EKAT_REQUIRE_MSG(input.has_array("radius_part"),
                     "Required name: radius_part");
    EKAT_REQUIRE_MSG(input.has_array("density_part"),
                     "Required name: density_part");
    EKAT_REQUIRE_MSG(input.has_array("sig_part"), "Required name: sig_part");
    EKAT_REQUIRE_MSG(input.has_array("fricvel"), "Required name: fricvel");
    EKAT_REQUIRE_MSG(input.has_array("ram1"), "Required name: ram1");
    EKAT_REQUIRE_MSG(input.has_array("vlc_grv"), "Required name: vlc_grv");

    EKAT_REQUIRE_MSG(11 == input.get_array("fraction_landuse").size(),
                     "fraction_landuse array should be exactly 1 entry long.");

    for (std::string s :
         {"moment", "radius_max", "tair", "pmid", "radius_part", "density_part",
          "sig_part", "fricvel", "ram1", "vlc_grv"}) {
      EKAT_REQUIRE_MSG(1 == input.get_array(s).size(),
                       s + " array should be exactly 1 entry long.");
    }

    auto to_dev = [](const std::vector<Real> &vec) {
      ColumnView dev = mam4::validation::create_column_view(vec.size());
      auto host = Kokkos::create_mirror_view(dev);
      for (int i = 0; i < vec.size(); ++i)
        host[i] = vec[i];
      Kokkos::deep_copy(dev, host);
      return dev;
    };
    const int moment = int(input.get_array("moment").front());
    auto fraction_land_use = to_dev(input.get_array("fraction_landuse"));
    const Real radius_max = input.get_array("radius_max").front();
    const Real tair = input.get_array("tair").front();
    const Real pmid = input.get_array("pmid").front();
    const Real radius_part = input.get_array("radius_part").front();
    const Real density_part = input.get_array("density_part").front();
    const Real sig_part = input.get_array("sig_part").front();
    const Real fricvel = input.get_array("fricvel").front();
    const Real ram1 = input.get_array("ram1").front();
    const Real vlc_grv = input.get_array("vlc_grv").front();

    typedef typename Kokkos::MinMax<Real>::value_type DryTrb;
    DryTrb dry_trb;
    Kokkos::parallel_reduce(
        1,
        KOKKOS_LAMBDA(const int, DryTrb &vlc) {
          Real vlc_trb, vlc_dry;
          drydep::modal_aero_turb_drydep_velocity(
              moment, fraction_land_use.data(), radius_max, tair, pmid,
              radius_part, density_part, sig_part, fricvel, ram1, vlc_grv,
              vlc_trb, vlc_dry);
          vlc.min_val = vlc_dry;
          vlc.max_val = vlc_trb;
        },
        Kokkos::MinMax<Real>(dry_trb));

    output.set("vlc_dry", dry_trb.min_val);
    output.set("vlc_trb", dry_trb.max_val);
  });
}
