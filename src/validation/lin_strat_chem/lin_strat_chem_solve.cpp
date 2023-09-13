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
using namespace lin_strat_chem;
void lin_strat_chem_solve(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    const Real o3col = input.get_array("o3col")[0];
    const Real temperature = input.get_array("temp")[0];
    const Real sza = input.get_array("sza")[0];
    const Real pmid = input.get_array("pmid")[0];
    const Real delta_t = input.get_array("delta_t")[0];
    const Real rlats = input.get_array("rlats")[0];
    const Real linoz_o3_clim = input.get_array("linoz_o3_clim")[0];
    const Real linoz_t_clim = input.get_array("linoz_t_clim")[0];
    const Real linoz_o3col_clim = input.get_array("linoz_o3col_clim")[0];
    const Real linoz_PmL_clim = input.get_array("linoz_PmL_clim")[0];
    const Real linoz_dPmL_dO3 = input.get_array("linoz_dPmL_dO3")[0];
    const Real linoz_dPmL_dO3col = input.get_array("linoz_dPmL_dO3col")[0];
    const Real linoz_cariolle_psc = input.get_array("linoz_cariolle_psc")[0];
    const Real chlorine_loading = input.get_array("chlorine_loading")[0];
    const Real linoz_dPmL_dT = input.get_array("linoz_dPmL_dT")[0];
    const Real psc_T = input.get_array("psc_T")[0];

    Real o3_vmr = input.get_array("o3_vmr")[0];

    constexpr Real zero = 0;
    Real ss_o3 = zero;
    Real do3_linoz = zero;
    Real do3_linoz_psc = zero;
    Real o3col_du_diag = zero;
    Real o3clim_linoz_diag = zero;
    Real sza_degrees = zero;

    lin_strat_chem_solve_kk(o3col, temperature, sza, pmid, delta_t, rlats,
                            // ltrop, & !in
                            linoz_o3_clim, linoz_t_clim, linoz_o3col_clim,
                            linoz_PmL_clim, linoz_dPmL_dO3,
                            linoz_dPmL_dT, // in
                            linoz_dPmL_dO3col,
                            linoz_cariolle_psc, // in
                            //
                            chlorine_loading,
                            psc_T, // PSC ozone loss T (K) threshold
                            o3_vmr,
                            // diagnostic variables outputs
                            do3_linoz, do3_linoz_psc, ss_o3, o3col_du_diag,
                            o3clim_linoz_diag, sza_degrees);

    output.set("o3_vmr", std::vector<Real>(1, o3_vmr));
    // ask for the following outputs
    output.set("do3_linoz", std::vector<Real>(1, do3_linoz));
    output.set("do3_linoz_psc", std::vector<Real>(1, do3_linoz_psc));
    output.set("ss_o3", std::vector<Real>(1, ss_o3));
    output.set("o3col_du_diag", std::vector<Real>(1, o3col_du_diag));
    output.set("o3clim_linoz_diag", std::vector<Real>(1, o3clim_linoz_diag));
    // output.set("sza_degrees", sza_degrees);
  });
}
