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

void twmo(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    using View1DHost = typename HostType::view_1d<Real>;
    using ColumnView = haero::ColumnView;
    constexpr int pver = mam4::nlev;

    const auto pmid1d_in = input.get_array("pmid1d");
    const auto temp1d_in = input.get_array("temp1d");
    const Real plimu = input.get_array("plimu")[0];
    const Real pliml = input.get_array("pliml")[0];
    const Real gam = input.get_array("gam")[0];
    Real trp = input.get_array("trp")[0];
    //const Real cnst_kap = input.get_array("cnst_kap")[0];
    //const Real cnst_ka1 = input.get_array("cnst_ka1")[0];
    //const Real cnst_faktor = input.get_array("cnst_faktor")[0];

    ColumnView pmid1d, temp1d;
    auto pmid1d_host = View1DHost((Real *)pmid1d_in.data(), pver); 
    auto temp1d_host = View1DHost((Real *)temp1d_in.data(), pver); 
    pmid1d = haero::testing::create_column_view(pver);
    temp1d = haero::testing::create_column_view(pver);
    Kokkos::deep_copy(pmid1d, pmid1d_host);
    Kokkos::deep_copy(temp1d, temp1d_host);

    std::cout << "trp: " << trp << std::endl;

    tropopause::twmo(temp1d, pmid1d, plimu, pliml, gam, trp);

    output.set("trp", trp);
  });
}
