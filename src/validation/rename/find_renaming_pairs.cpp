// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/mam4.hpp>

#include <mam4xx/aero_config.hpp>
#include <mam4xx/rename.hpp>
#include <skywalker.hpp>
#include <validation.hpp>

using namespace skywalker;
using namespace mam4;

void find_renaming_pairs(Ensemble *ensemble) {

  ensemble->process([=](const Input &input, Output &output) {
    const int nmodes = AeroConfig::num_modes();

    // int dest_mode_of_mode = input.get_array("dest_mode_of_mode");
    int dest_mode_of_mode[nmodes] = {-1, 0, -1, -1};

    Real fmode_dist_tail_fac[nmodes];
    Real v2n_lo_rlx[nmodes];
    Real v2n_hi_rlx[nmodes];
    Real ln_diameter_tail_fac[nmodes];
    int num_pairs = 0;
    Real diameter_cutoff[nmodes];
    Real ln_dia_cutoff[nmodes];
    Real diameter_threshold[nmodes];
    Real mean_std_dev[nmodes];

    rename::find_renaming_pairs(dest_mode_of_mode,    // in
                                mean_std_dev,         // out
                                fmode_dist_tail_fac,  // out
                                v2n_lo_rlx,           // out
                                v2n_hi_rlx,           // out
                                ln_diameter_tail_fac, // out
                                num_pairs,            // out
                                diameter_cutoff,      // out
                                ln_dia_cutoff, diameter_threshold);

    // We do not use sz_factor in rename-mam4xx; however, rename-mam4 does.
    // We only computes sz_factor for validation proposes
    Real sz_factor[nmodes] = {0};
    sz_factor[1] =
        Constants::pi_sixth * exp(4.5 * square(log(modes(1).mean_std_dev)));
    sz_factor[0] =
        Constants::pi_sixth * exp(4.5 * square(log(modes(0).mean_std_dev)));

    std::vector<Real> sz_factor_out(nmodes, 0);
    validation::convert_modal_array_to_vector(sz_factor, sz_factor_out);
    output.set("sz_factor", sz_factor_out);

    std::vector<Real> fmode_dist_tail_fac_out(nmodes, 0);
    validation::convert_modal_array_to_vector(fmode_dist_tail_fac,
                                              fmode_dist_tail_fac_out);
    output.set("fmode_dist_tail_fac", fmode_dist_tail_fac_out);

    std::vector<Real> v2n_lo_rlx_out(nmodes, 0);
    validation::convert_modal_array_to_vector(v2n_lo_rlx, v2n_lo_rlx_out);
    output.set("v2n_lo_rlx", v2n_lo_rlx_out);

    std::vector<Real> v2n_hi_rlx_out(nmodes, 0);
    validation::convert_modal_array_to_vector(v2n_hi_rlx, v2n_hi_rlx_out);
    output.set("v2n_hi_rlx", v2n_hi_rlx_out);

    std::vector<Real> ln_diameter_tail_fac_out(nmodes, 0);
    validation::convert_modal_array_to_vector(ln_diameter_tail_fac,
                                              ln_diameter_tail_fac_out);
    output.set("ln_diameter_tail_fac", ln_diameter_tail_fac_out);

    std::vector<Real> diameter_cutoff_out(nmodes, 0);
    validation::convert_modal_array_to_vector(diameter_cutoff,
                                              diameter_cutoff_out);
    output.set("diameter_cutoff", diameter_cutoff_out);

    output.set("num_pairs", num_pairs);

    std::vector<Real> ln_dia_cutoff_out(nmodes, 0);
    validation::convert_modal_array_to_vector(ln_dia_cutoff, ln_dia_cutoff_out);
    output.set("ln_dia_cutoff", ln_dia_cutoff_out);

    std::vector<Real> diameter_threshold_out(nmodes, 0);
    validation::convert_modal_array_to_vector(diameter_threshold,
                                              diameter_threshold_out);
    output.set("diameter_threshold", diameter_threshold_out);
  });
}
