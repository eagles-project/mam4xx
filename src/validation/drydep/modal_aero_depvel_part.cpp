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

void modal_aero_depvel_part(Ensemble *ensemble) {

  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    const int nlev = 72;
    const Real fraction_landuse[DryDeposition::n_land_type] = {
        0.20918898065265040e-02, 0.10112323792561469e+00,
        0.19104123086831826e+00, 0.56703179010502225e+00,
        0.00000000000000000e+00, 0.42019237748858657e-01,
        0.85693761223933115e-01, 0.66234294754917442e-02,
        0.00000000000000000e+00, 0.00000000000000000e+00,
        0.43754228462347953e-02};
    EKAT_REQUIRE_MSG(input.has_array("moment"), "Required name: moment");
    EKAT_REQUIRE_MSG(input.has_array("tair"), "Required name: tair");
    EKAT_REQUIRE_MSG(input.has_array("pmid"), "Required name: pmid");
    EKAT_REQUIRE_MSG(input.has_array("radius_part"),
                     "Required name: radius_part");
    EKAT_REQUIRE_MSG(input.has_array("density_part"),
                     "Required name: density_part");
    EKAT_REQUIRE_MSG(input.has_array("sig_part"), "Required name: sig_part");
    EKAT_REQUIRE_MSG(input.has_array("fricvel"), "Required name: fricvel");
    EKAT_REQUIRE_MSG(input.has_array("ram1"), "Required name: ram1");

    for (std::string s : {"moment", "fricvel", "ram1"}) {
      EKAT_REQUIRE_MSG(1 == input.get_array(s).size(),
                       s + " array should be exactly 1 entry long.");
    }
    for (std::string s :
         {"tair", "pmid", "radius_part", "density_part", "sig_part"}) {
      EKAT_REQUIRE_MSG(nlev == input.get_array(s).size(),
                       s + " array should be exactly 72 entries long.");
    }

    const int moment = int(input.get_array("moment").front());
    const Real fricvel = input.get_array("fricvel").front();
    const Real ram1 = input.get_array("ram1").front();

    auto to_dev = [](const std::vector<Real> &vec) {
      ColumnView dev = mam4::validation::create_column_view(vec.size());
      auto host = Kokkos::create_mirror_view(dev);
      for (int i = 0; i < vec.size(); ++i)
        host[i] = vec[i];
      Kokkos::deep_copy(dev, host);
      return dev;
    };
    auto tair = to_dev(input.get_array("tair"));
    auto pmid = to_dev(input.get_array("pmid"));
    auto radius_part = to_dev(input.get_array("radius_part"));
    auto density_part = to_dev(input.get_array("density_part"));
    auto sig_part = to_dev(input.get_array("sig_part"));

    ColumnView vlc_trb = mam4::validation::create_column_view(nlev);
    ColumnView vlc_dry = mam4::validation::create_column_view(nlev);
    ColumnView vlc_grv = mam4::validation::create_column_view(nlev);
    Kokkos::parallel_for(
        "modal_aero_depvel_part", nlev, KOKKOS_LAMBDA(int lev) {
          const bool lowest_model_layer = (lev == nlev - 1);
          drydep::modal_aero_depvel_part(
              lowest_model_layer, fraction_landuse, tair[lev], pmid[lev], ram1,
              fricvel, radius_part[lev], density_part[lev], sig_part[lev],
              moment, vlc_dry[lev], vlc_trb[lev], vlc_grv[lev]);
        });
    auto to_host = [](ColumnView dev) {
      auto host = Kokkos::create_mirror_view(dev);
      Kokkos::deep_copy(host, dev);
      std::vector<Real> vec(host.size());
      for (int i = 0; i < vec.size(); ++i)
        vec[i] = host[i];
      return vec;
    };
    output.set("vlc_grv", to_host(vlc_grv));
    output.set("vlc_dry", to_host(vlc_dry));
    output.set("vlc_trb", to_host(vlc_trb).back());
  });
}
