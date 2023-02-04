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
    const int naerosol_species = AeroConfig::num_aerosol_ids();

    // int dest_mode_of_mode = input.get_array("dest_mode_of_mode");
    int dest_mode_of_mode[nmodes] = {0, 1, 0, 0};

    Real sz_factor[nmodes];
    Real fmode_dist_tail_fac[nmodes];
    Real v2n_lo_rlx[nmodes];
    Real v2n_hi_rlx[nmodes];
    Real ln_diameter_tail_fac[nmodes];
    int num_pairs = 0;
    Real diameter_cutoff[nmodes];
    Real ln_dia_cutoff[nmodes];
    Real diameter_threshold[nmodes];
    Real mass_2_vol[naerosol_species];

    rename::find_renaming_pairs(dest_mode_of_mode,    // in
                                sz_factor,            // out
                                fmode_dist_tail_fac,  // out
                                v2n_lo_rlx,           // out
                                v2n_hi_rlx,           // out
                                ln_diameter_tail_fac, // out
                                num_pairs,            // out
                                diameter_cutoff,      // out
                                ln_dia_cutoff,
                                diameter_threshold);

    auto save_mode_values = [](const Real values[nmodes],
                               std::vector<Real> &values_vector) {
      for (int i = 0; i < nmodes; ++i)
        values_vector[i] = values[i];
    };

    Real molecular_weight_rename[naerosol_species] = {
        150, 115, 150, 12, 58.5, 135, 250092}; // [kg/kmol]
    for (int iaero = 0; iaero < naerosol_species; ++iaero) {
      mass_2_vol[iaero] =
          molecular_weight_rename[iaero] / aero_species(iaero).density;
    }

    std::vector<Real> sz_factor_out(nmodes, 0);
    save_mode_values(sz_factor, sz_factor_out);
    output.set("sz_factor", sz_factor_out);

    std::vector<Real> fmode_dist_tail_fac_out(nmodes, 0);
    save_mode_values(fmode_dist_tail_fac, fmode_dist_tail_fac_out);
    output.set("fmode_dist_tail_fac", fmode_dist_tail_fac_out);

    std::vector<Real> v2n_lo_rlx_out(nmodes, 0);
    save_mode_values(v2n_lo_rlx, v2n_lo_rlx_out);
    output.set("v2n_lo_rlx", v2n_lo_rlx_out);

    std::vector<Real> v2n_hi_rlx_out(nmodes, 0);
    save_mode_values(v2n_hi_rlx, v2n_hi_rlx_out);
    output.set("v2n_hi_rlx", v2n_hi_rlx_out);

    std::vector<Real> ln_diameter_tail_fac_out(nmodes, 0);
    save_mode_values(ln_diameter_tail_fac, ln_diameter_tail_fac_out);
    output.set("ln_diameter_tail_fac", ln_diameter_tail_fac_out);

    std::vector<Real> diameter_cutoff_out(nmodes, 0);
    save_mode_values(diameter_cutoff, diameter_cutoff_out);
    output.set("diameter_cutoff", diameter_cutoff_out);

    output.set("num_pairs", num_pairs);

    std::vector<Real> ln_dia_cutoff_out(nmodes, 0);
    save_mode_values(ln_dia_cutoff, ln_dia_cutoff_out);
    output.set("ln_dia_cutoff", ln_dia_cutoff_out);

    std::vector<Real> diameter_threshold_out(nmodes, 0);
    save_mode_values(diameter_threshold, diameter_threshold_out);
    output.set("diameter_threshold", diameter_threshold_out);


    // std::vector<Real> _out(nmodes, 0);
    // save_mode_values(, _out);
    // output.set("", _out);



                                

                                
  });
}