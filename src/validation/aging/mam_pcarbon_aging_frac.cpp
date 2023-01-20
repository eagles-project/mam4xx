#include <mam4xx/aging.hpp>
#include <skywalker.hpp>
#include <validation.hpp>

using namespace skywalker;
using namespace mam4;

void mam_pcarbon_aging_frac(Ensemble *ensemble) {

  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    int nlev = 1;
    mam4::Prognostics progs(nlev);

    auto nsrc = input.get_array("nsrc");
    auto dgn_a_f = input.get_array("dgn_a");
    auto qaer_cur_f = input.get_array("qaer_cur");
    auto qaer_del_cond_f = input.get_array("qaer_del_cond");
    auto qaer_del_coag_in_f = input.get_array("qaer_del_coag_in");

    const int num_modes = AeroConfig::num_modes();
    const int num_aero = AeroConfig::num_aerosol_ids();

    Real qaer_cur_c[num_aero][num_modes];
    Real qaer_del_cond_c[num_aero][num_modes];
    Real qaer_del_coag_in_c[num_aero][num_modes];

    int n = 0;
    for (int a = 0; a < num_aero; ++a) {
      for (int m = 0; m < num_modes; ++m) {
        qaer_cur_c[a][m] = qaer_cur_f[n];
        qaer_del_cond_c[a][m] = qaer_del_cond_f[n];
        qaer_del_coag_in_c[a][m] = qaer_del_coag_in_f[n];
        n += 1;
      }
    }

    Real xferfrac_pcage;
    Real frac_cond;
    Real frac_coag;

    aging::mam_pcarbon_aging_frac(nsrc[0], dgn_a_f.data(), qaer_cur_c,
                                  qaer_del_cond_c, qaer_del_coag_in_c,
                                  xferfrac_pcage, frac_cond, frac_coag);

    n = 0;
    for (int a = 0; a < num_aero; ++a) {
      for (int m = 0; m < num_modes; ++m) {
        qaer_cur_f[n] = qaer_cur_c[a][m];
        qaer_del_cond_f[n] = qaer_del_cond_c[a][m];
        qaer_del_coag_in_f[n] = qaer_del_coag_in_c[a][m];
        n += 1;
      }
    }

    output.set("qaer_cur", qaer_cur_f);
    output.set("qaer_del_cond", qaer_del_cond_f);
    output.set("qaer_del_coag_in", qaer_del_coag_in_f);
    output.set("xferfrac_pcage", xferfrac_pcage);
    output.set("frac_cond", frac_cond);
    output.set("frac_coag", frac_coag);
  });
}