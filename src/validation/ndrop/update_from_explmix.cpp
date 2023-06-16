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

void update_from_explmix(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    // number of vertical points.
    const int top_lev = input.get_array("top_lev")[0] -1;
    const int pver = input.get_array("pver")[0];
    const int ntot_amode = input.get_array("ntot_amode")[0];
    const auto mam_idx_db = input.get_array("mam_idx");
    const auto nspec_amode_db = input.get_array("nspec_amode");

    int nnew = input.get_array("nnew")[0];
    int nsav = input.get_array("nsav")[0];
    nnew -= 1;
    nsav -= 1;
     

    const auto raercol_1 = input.get_array("raercol_1");
    const auto raercol_cw_1 = input.get_array("raercol_cw_1");
    const auto raercol_2 = input.get_array("raercol_2");
    const auto raercol_cw_2 = input.get_array("raercol_cw_2");
    
    auto qcld = input.get_array("qcld");
    auto mact = input.get_array("mact");
    auto nact = input.get_array("nact");
    const auto ekd = input.get_array("ekd");
    const auto zs = input.get_array("zs");
    const auto zn = input.get_array("zn");
    const auto cldn_col = input.get_array("cldn_col");
    const auto csbot = input.get_array("csbot");
    
    const Real dt = input.get("dt");
    const Real dtmicro = input.get_array("dtmicro")[0];

    const Real zero = 0.0;
    const int nmodes = AeroConfig::num_modes();
    const int ncnst_tot = 25;
    const int nspec_max = 8;
    int raer_len = pver * ncnst_tot;
    int act_len = pver * nmodes;

    //std::vector<std::vector<Real>> nact_out(pver, std::vector<Real>(nmodes));
    std::vector<Real> nact_out(act_len, 0.0);
    std::vector<Real> mact_out(act_len, 0.0);
    std::vector<std::vector<Real>> _nact(pver, std::vector<Real>(nmodes));
    std::vector<std::vector<Real>> _mact(pver, std::vector<Real>(nmodes));
    

    std::vector<Real> qcld_out(pver);
    
    std::vector<Real> raercol_1_out(raer_len, 0.0);
    std::vector<Real> raercol_2_out(raer_len, 0.0);
    std::vector<Real> raercol_cw_1_out(raer_len, 0.0);
    std::vector<Real> raercol_cw_2_out(raer_len, 0.0);
    //std::vector<std::vector<Real>> (pver, std::vector<Real>(ncnst_tot));
    
    std::vector<Real> nnew_out(1);
    std::vector<Real> nsav_out(1);
    int counter = 0;

    Real raercol[pver][ncnst_tot][2];
    Real raercol_cw[pver][ncnst_tot][2];

    int nspec_amode[nmodes];
    int mam_idx[nmodes][nspec_max];
    for(int m = 0; m < nmodes; m++) {
      nspec_amode[m] = nspec_amode_db[m];
      
    }
    counter = 0;
    for(int n = 0; n < nspec_max; n++) {
      for(int m = 0; m < nmodes; m++) {
        mam_idx[m][n] = mam_idx_db[counter];
        counter++;
      }
    }

    counter = 0; 
    for(int n = 0; n < ncnst_tot; n++) {
       for (int k = 0; k < pver; k++){
          raercol[k][n][0] = raercol_1[counter];
          raercol_cw[k][n][0] = raercol_cw_1[counter];

          raercol[k][n][1] = raercol_2[counter];
          raercol_cw[k][n][1] = raercol_cw_2[counter];
          counter++;

      }
    } 

    counter = 0;
    for(int m = 0; m < nmodes; m++) {
        for (int k = 0; k < pver; k++) {
            _nact[k][m] = nact[counter];
            _mact[k][m] = mact[counter];
            counter++;
        }
    }  
    
    for (int k = top_lev; k < pver; k++) {
      // add logic for km1 and kp1 from fortran
      int kp1 = haero::min(k + 1, pver);
      int km1 = haero::max(k - 1, top_lev);

      Real csbot_k = csbot[k];
      Real csbot_km1 = csbot[km1];
      Real cldn_k = cldn_col[k];
      Real cldn_km1 = cldn_col[km1];
      Real cldn_kp1 = cldn_col[kp1];

      Real zn_k = zn[k];
      Real zs_k = zs[k];
      Real zs_km1 = zs[km1];
      Real ekd_k = ekd[k];
      Real ekd_km1 = ekd[km1];

      auto nact_k = _nact[k].data();
      auto mact_k = _mact[k].data();

      Real qcld_k = qcld[k];
      Real qcld_km1 = qcld[km1];
      Real qcld_kp1 = qcld[kp1];

      
      ndrop::update_from_explmix(dtmicro, k, pver, csbot_k, csbot_km1, cldn_k, cldn_km1, cldn_kp1, 
                    zn_k, zs_k, zs_km1, ekd_k, ekd_km1, nact_k, mact_k, 
                    qcld_k, qcld_km1,  qcld_kp1, raercol[k], raercol[km1], raercol[kp1], 
                    raercol_cw[k], raercol_cw[km1], raercol_cw[kp1], 
                    nsav, nnew, nspec_amode, mam_idx);

      for(int m = 0; m < nmodes; m++) {
        _nact[k][m] = nact_k[m];
        _mact[k][m] = mact_k[m];
      }

      qcld[k] = qcld_k;
      qcld[km1] = qcld_km1;
      qcld[kp1] = qcld_kp1;

    }

    //nnew += 1;
    //nsav += 1;
    nnew_out[0] = nnew + 1;
    nsav_out[0] = nsav + 1; 
    counter = 0;
    for(int n = 0; n < ncnst_tot; n++) {
       for (int k = 0; k < pver; k++){
          raercol_1_out[counter] = raercol[k][n][0];
          raercol_cw_1_out[counter] = raercol_cw[k][n][0];

          raercol_2_out[counter] = raercol[k][n][1];
          raercol_cw_2_out[counter] = raercol_cw[k][n][1];
          counter++;
      }
    } 
    for (int k = 0; k < pver; k++) {
      qcld_out[k] = qcld[k]; 
    }
    counter = 0;
    for(int m = 0; m < nmodes; m++) {
        for (int k = 0; k < pver; k++) {
            nact[counter] = _nact[k][m];
            mact[counter] = _mact[k][m];
            counter++;
        }
    }  
    
    output.set("nact", nact_out);
    output.set("mact", mact_out);
    output.set("qcld", qcld_out);

    output.set("raercol_1", raercol_1_out);
    output.set("raercol_cw_1", raercol_cw_1_out);
    output.set("raercol_2", raercol_2_out);
    output.set("raercol_cw_2", raercol_cw_2_out);

    output.set("nnew", nnew_out);
    output.set("nsav", nsav_out);
  });
}
