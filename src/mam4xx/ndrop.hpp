ifndef MAM4XX_NDROP_HPP
#define MAM4XX_NDROP_HPP

#include <haero/aero_species.hpp>
#include <haero/constants.hpp>
#include <haero/gas_species.hpp>
#include <haero/math.hpp>

#include <mam4xx/aero_config.hpp>

#include "mam4_types.hpp"

#include <iostream>
#include <string>
#include <vector>

namespace mam4 {


//const inputs?
KOKKOS_INLINE_FUNCTION void get_total_aer_mmr_sum(int imode, 
                                                  int nspec, 
                                                  int istart, 
                                                  int istop, 
                                                  Real vaerosolsum[AeroConfig::num_modes()][AeroConfig::num_aerosol_ids()], //out
                                                  Real hygrosum[AeroConfig::num_modes()][AeroConfig::num_aerosol_ids()]     //out
                                                  ) {


    //internal
    //Prognositics::q_aero_i[mode][id]
    //Prognostics::q_aero_c[mode][id]

    Real density_sp;
    Real hygro_sp;
    Real vol;

    int type_idx;
    int density_sp;
    int hygro_sp;
    int spc_idx;

    for(int lspec = 0; lspec < nspec; lspec++) {
        type_idx = lspectype_amode(lspec, imode);
        density_sp  = specdens_amode(type_idx);      //species density
        //hygro_sp    = spechygro(type_idx);           //species hygroscopicity
        hygro_sp    = Diagnostics::hygroscopicity[type_idx];           //species hygroscopicity
        spc_idx   = lmassptr_amode(lspec,imode);     //index of species in state_q array
        for(int icol = istart; icol < istop; icol++) {
            vol = max(Prognositics::q_aero_i[icol][spc_idx] + Prognostics::q_aero_c[icol][spc_idx], 0) / density_sp;       //volume = mmr/density
            
            total_mmr = ((Prognositics::q_aero_i[icol][spc_idx] * Diagnostics::dry_geometric_mean_diameter_i[icol]) + (Prognostics::q_aero_c[icol][spc_idx] * Diagnostics::dry_geometric_mean_diameter_c[icol])) / (Prognositics::q_aero_i[icol][spc_idx] + Prognostics::q_aero_c[icol][spc_idx]);
            
            vaerosolsum[icol] += vol;           //bulk volume
            hygrosum[icol] += vol * hygro_sp;    //bulk hygroscopicity
        }
    }

//Fortran
//  do lspec = 1, nspec
//       type_idx = lspectype_amode(lspec,imode)
//       density_sp  = specdens_amode(type_idx) !species density
//       hygro_sp    = spechygro(type_idx)      !species hygroscopicity
//       spc_idx   = lmassptr_amode(lspec,imode) !index of species in state_q array
//       do icol = istart, istop
//          vol = max(state_q(icol,spc_idx) + qcldbrn1d(icol,lspec), 0._r8)/density_sp !volume = mmr/density
//          vaerosolsum(icol) = vaerosolsum(icol) + vol        !bulk volume
//          hygrosum(icol)    = hygrosum(icol) + vol*hygro_sp !bulk hygroscopicity
//       enddo
//    enddo

}

}