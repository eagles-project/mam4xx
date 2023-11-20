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
void setsox_test_nlev(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    // Ensemble parameters
    // Declare array of strings for input names
    std::string input_variables[] = {"dt"};

    std::string input_arrays[] = {
        "ncol", "loffset", "dtime",  "press", "pdel", "tfld", "mbar",
        "lwc",  "cldfrc",  "cldnum", "xhnm",  "qcw",  "qin"};

    // Iterate over input_variables and error if not in input
    for (std::string name : input_variables) {
      if (!input.has(name.c_str())) {
        std::cerr << "Required name for variable: " << name << std::endl;
        exit(1);
      }
    }
    // Iterate over input_arrays and error if not in input
    for (std::string name : input_arrays) {
      if (!input.has_array(name.c_str())) {
        std::cerr << "Required name for array: " << name << std::endl;
        exit(1);
      }
    }

    // Parse input
    const Real dt = input.get_array("dtime")[0];
    const int loffset = input.get_array("loffset")[0];
    const Real press_in = input.get_array("press")[0];
    const Real pdel_in = input.get_array("pdel")[0];
    const Real tfld_in = input.get_array("tfld")[0];
    const Real mbar_in = input.get_array("mbar")[0];
    const Real lwc_in = input.get_array("lwc")[0];
    const Real cldfrc_in = input.get_array("cldfrc")[0];
    const Real cldnum_in = input.get_array("cldnum")[0];
    const Real xhnm_in = input.get_array("xhnm")[0];

    auto qcw_in = input.get_array("qcw");
    auto qin_in = input.get_array("qin");

    const int nlev = mam4::nlev;
    const int nspec = AeroConfig::num_gas_phase_species();

    // This is a sort hacky way of testing the column dispatch, but should still
    // be reasonably robust.
    // Since the column dispatch pretty much just calls the single-cell version,
    // we'll use the same validation data for every level in a column.
    // then we'll make sure every level gets the same answer and then write the
    // first entry as the final output
    ColumnView press = haero::testing::create_column_view(nlev);
    auto press_h = Kokkos::create_mirror_view(press);
    ColumnView pdel = haero::testing::create_column_view(nlev);
    auto pdel_h = Kokkos::create_mirror_view(pdel);
    ColumnView tfld = haero::testing::create_column_view(nlev);
    auto tfld_h = Kokkos::create_mirror_view(tfld);
    ColumnView mbar = haero::testing::create_column_view(nlev);
    auto mbar_h = Kokkos::create_mirror_view(mbar);
    ColumnView lwc = haero::testing::create_column_view(nlev);
    auto lwc_h = Kokkos::create_mirror_view(lwc);
    ColumnView cldfrc = haero::testing::create_column_view(nlev);
    auto cldfrc_h = Kokkos::create_mirror_view(cldfrc);
    ColumnView cldnum = haero::testing::create_column_view(nlev);
    auto cldnum_h = Kokkos::create_mirror_view(cldnum);
    ColumnView xhnm = haero::testing::create_column_view(nlev);
    auto xhnm_h = Kokkos::create_mirror_view(xhnm);

    using View1DHost = typename HostType::view_1d<Real>;
    ColumnView qcw[nspec];
    View1DHost qcw_h[nspec];
    ColumnView qin[nspec];
    View1DHost qin_h[nspec];

    for (int i = 0; i < nspec; ++i) {
      qcw[i] = haero::testing::create_column_view(nlev);
      qcw_h[i] = Kokkos::create_mirror_view(qcw[i]);
      qin[i] = haero::testing::create_column_view(nlev);
      qin_h[i] = Kokkos::create_mirror_view(qin[i]);
    }

    for (int k = 0; k < nlev; ++k) {
      press_h(k) = press_in;
      pdel_h(k) = pdel_in;
      tfld_h(k) = tfld_in;
      mbar_h(k) = mbar_in;
      lwc_h(k) = lwc_in;
      cldfrc_h(k) = cldfrc_in;
      cldnum_h(k) = cldnum_in;
      xhnm_h(k) = xhnm_in;
      for (int i = 0; i < nspec; ++i) {
        qcw_h[i](k) = qcw_in[i];
        qin_h[i](k) = qin_in[i];
      }
    }

    Kokkos::deep_copy(press, press_h);
    Kokkos::deep_copy(pdel, pdel_h);
    Kokkos::deep_copy(tfld, tfld_h);
    Kokkos::deep_copy(mbar, mbar_h);
    Kokkos::deep_copy(lwc, lwc_h);
    Kokkos::deep_copy(cldfrc, cldfrc_h);
    Kokkos::deep_copy(cldnum, cldnum_h);
    Kokkos::deep_copy(xhnm, xhnm_h);

    for (int i = 0; i < nspec; ++i) {
      Kokkos::deep_copy(qcw[i], qcw_h[i]);
      Kokkos::deep_copy(qin[i], qin_h[i]);
    }

    // Single-column dispatch.
    auto team_policy = ThreadTeamPolicy(1u, Kokkos::AUTO);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
          mam4::mo_setsox::setsox(team, loffset, dt, press, pdel, tfld, mbar,
                                  lwc, cldfrc, cldnum, xhnm, qcw, qin);
        });

    std::vector<Real> qcw_out(nspec);
    std::vector<Real> qin_out(nspec);
    for (int i = 0; i < nspec; ++i) {
      Kokkos::deep_copy(qcw_h[i], qcw[i]);
      qcw_out[i] = qcw_h[i](0);
      Kokkos::deep_copy(qin_h[i], qin[i]);
      qin_out[i] = qin_h[i](0);
      for (int k = 0; k < nlev; ++k) {
        // make sure every level got the same answer
        EKAT_ASSERT(qcw_h[i](0) == qcw_h[i](k));
        EKAT_ASSERT(qin_h[i](0) == qin_h[i](k));
      }
    }

    output.set("qcw", qcw_out);
    output.set("qin", qin_out);
  });
}
