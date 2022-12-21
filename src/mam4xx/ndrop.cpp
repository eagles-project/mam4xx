#include "ndrop.hpp"
#include <string>

namespace mam4 {

//const inputs?
something get_aer_mmr_sum(int imode, int nspec, int istart, int istop, Real state_q, Real qcldbrn1d) {

    //output
    Real vaerosolsum;
    Real hygrosum;

    //internal
    Real density_sp;
    Real hygro_sp;
    Real vol;

    int icol;
    int spc_idx;
    int type_idx;

    for(int lspec = 0; lspec < nspec; lspec++) {
        type_idx = lspectype_amode(lspec, imode);
        //density_sp
        //hygro_sp
        //spc_idx
        for(int icol = istart; icol < istop; icol++) {
            //max
            //vaerosolsum
            //hygrosum
        }
    }

}

}