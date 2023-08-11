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
using namespace mo_photo;
void calc_sum_wght(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    // validation test from standalone mo_photo.
    const auto dels = input.get_array("dels");
    const auto wrk0 = input.get_array("wrk0")[0];
    const int iz = int(input.get_array("iz")[0]) - 1;
    const int is = int(input.get_array("is_")[0]) - 1;
    const int iv = int(input.get_array("iv")[0]) - 1;
    const int ial = int(input.get_array("ial")[0]) - 1;
    const auto rsf_tab_1d = input.get_array("rsf_tab");

    auto shape_rsf_tab = input.get_array("shape_rsf_tab");

    const int nw = 67;
    const int nump = 15;
    const int numsza = 10;
    const int numcolo3 = 10;
    const int numalb = 10;

    View5D rsf_tab("rsf_tab", nw, nump, numsza, numcolo3, numalb);
    auto rsf_tab_1 = Kokkos::subview(rsf_tab, Kokkos::ALL(), 1, Kokkos::ALL(),
                                     Kokkos::ALL(), Kokkos::ALL());

    auto rsf_tab_2 = Kokkos::subview(rsf_tab, Kokkos::ALL(), Kokkos::ALL(), 6,
                                     Kokkos::ALL(), Kokkos::ALL());

    auto rsf_tab_3 = Kokkos::subview(rsf_tab, Kokkos::ALL(), Kokkos::ALL(),
                                     Kokkos::ALL(), 7, Kokkos::ALL());

    auto rsf_tab_4 = Kokkos::subview(rsf_tab, Kokkos::ALL(), Kokkos::ALL(),
                                     Kokkos::ALL(), Kokkos::ALL(), 3);

    auto rsf_tab_5 = Kokkos::subview(rsf_tab, 0, Kokkos::ALL(), Kokkos::ALL(),
                                     Kokkos::ALL(), Kokkos::ALL());

    auto rsf_tab_6 = Kokkos::subview(rsf_tab, 9, Kokkos::ALL(), Kokkos::ALL(),
                                     Kokkos::ALL(), Kokkos::ALL());

    Kokkos::deep_copy(rsf_tab, 0.1);
    Kokkos::deep_copy(rsf_tab_1, 2.0);
    Kokkos::deep_copy(rsf_tab_2, 3.0);
    Kokkos::deep_copy(rsf_tab_3, 1.0);
    Kokkos::deep_copy(rsf_tab_4, 0.8);
    Kokkos::deep_copy(rsf_tab_5, 6.0);
    Kokkos::deep_copy(rsf_tab_6, 1e-2);

    const Real zero = 0;
    std::vector<Real> psum(nw, zero);

    calc_sum_wght(dels.data(), wrk0, // in
                  iz, is, iv, ial,   // in
                  rsf_tab,           // in
                  nw,                //
                  psum.data());

    // C++ indexing to fortran indexing
    output.set("psum", psum);
  });
}
