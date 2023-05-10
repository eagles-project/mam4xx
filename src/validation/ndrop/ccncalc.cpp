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

void ccncalc(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {

    // number of vertical points. 
    const Real zero = 0;
    const int maxd_aspectype = 14;
    const int ntot_amode = 4;
    const int nvars = 40;
    const int psat = ndrop_od::psat;

    const int pver = input.get_array("pver")[0];
    const auto state_q_db = input.get_array("state_q");
     
    const auto tair_db = input.get_array("temp");
    const auto pmid_db = input.get_array("pmid");

    const int top_lev = 6;

    ColumnView state_q[nvars];
    
    for (int i = 0; i < nvars; ++i)
     {
       state_q[i] = haero::testing::create_column_view(pver);
     } 
    
    int count=0;
    for (int i = 0; i < nvars; ++i)
    {
      for (int kk = 0; kk < pver; ++kk)
      {
        state_q[i](kk) = state_q_db[count];
        count++;
      }
    }

    ColumnView tair;
    ColumnView pmid;
    tair = haero::testing::create_column_view(pver);
    pmid = haero::testing::create_column_view(pver);
    // // FIXME. Find a better way:
    for (int kk = 0; kk < pver; ++kk)
    {
       tair(kk) = tair_db[kk];
       pmid(kk) = pmid_db[kk];
    }  
    
    Real qcldbrn[maxd_aspectype][ntot_amode] = {{zero}}; 
    Real qcldbrn_num[ntot_amode] ={zero};

    const auto lspectype_amode_db = input.get_array("lspectype_amode"); 
    int lspectype_amode[maxd_aspectype][ntot_amode] = {};

    const auto lmassptr_amode_db = input.get_array("lmassptr_amode");
    int lmassptr_amode[maxd_aspectype][ntot_amode] = {};

    const auto specdens_amode_db = input.get_array("specdens_amode");
    const auto spechygro_db = input.get_array("spechygro");

    count=0;
    for (int i = 0; i < ntot_amode; ++i)
    {
      for (int j = 0; j < maxd_aspectype; ++j)
      {
        lspectype_amode[j][i] = lspectype_amode_db[count];
        lmassptr_amode[j][i] = lmassptr_amode_db[count];
        count++;
      }
    }

    const auto voltonumbhi_amode = input.get_array("voltonumbhi_amode");
    const auto voltonumblo_amode = input.get_array("voltonumblo_amode");
    const auto numptr_amode_db = input.get_array("numptr_amode");
    const auto nspec_amode_db  = input.get_array("nspec_amode");

    int numptr_amode[ntot_amode];
    int nspec_amode[ntot_amode];
    for (int i = 0; i < ntot_amode; ++i)
    {
      numptr_amode[i] = numptr_amode_db[i];
      nspec_amode[i] = nspec_amode_db[i];
    }

    ColumnView ccn[psat];

    for (int i = 0; i < psat; ++i)
    {
      ccn[i] = haero::testing::create_column_view(pver);
    }

    // FIXME: use a Kokkos:parallel_for which requires to transfer data from host to device. 
    for (int kk = top_lev; kk < pver; ++kk)
    {

      Real state_q_kk[nvars] = {zero}; 
      for (int i = 0; i < nvars; ++i)
      {
        state_q_kk[i] = state_q[i](kk);
      }

      Real air_density =
      conversions::density_of_ideal_gas(tair_db[kk], pmid_db[kk] );

      Real ccn_kk[psat] = {zero};

      ndrop_od::ccncalc(state_q_kk,
            tair(kk),
            qcldbrn,
            qcldbrn_num,
            air_density,
            lspectype_amode,
            specdens_amode_db.data(),
            spechygro_db.data(),
            lmassptr_amode,
            voltonumbhi_amode.data(),
            voltonumblo_amode.data(),
            numptr_amode,
            nspec_amode,
            ccn_kk); 

      

      for (int i = 0; i < psat; ++i)
      {
       ccn[i](kk) = ccn_kk[i]; 
      }

    } // end kk
    
    

    // printf("ccn(%d) 0 %e \n",top_lev, ccn[0][top_lev]);

    // for (int i = 0; i < 6; ++i)
    // {
    //   output.set("ccn_"+std::to_string(i+1), ccn[i]);
    // }
    
  });
}
