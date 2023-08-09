#ifndef MAM4XX_MO_CHM_DIAGS_HPP
#define MAM4XX_MO_CHM_DIAGS_HPP

#include <haero/math.hpp>
#include <mam4xx/mam4_types.hpp>
#include <mam4xx/utils.hpp>
#include <mam4xx/gas_chem.hpp>

namespace mam4 {

namespace mo_chm_diags {

//FIXME: bad constants
constexpr Real S_molwgt = 32.066;
// constants for converting O3 mixing ratio to DU
constexpr Real DUfac = 2.687e20;   // 1 DU in molecules per m^2
constexpr int gas_pcnst = gas_chem::gas_pcnst;
// number of vertical levels
constexpr int pver = 72;
constexpr int pverm = pver - 1;

KOKKOS_INLINE_FUNCTION
void het_diags(ColumnView het_rates, ColumnView mmr, ColumnView pdel, int lchnk, int ncol) {

    //real(r8), intent(in)  :: het_rates(ncol,pver,max(1,gas_pcnst))
    //real(r8), intent(in)  :: mmr(ncol,pver,gas_pcnst)
    //real(r8), intent(in)  :: pdel(ncol,pver)

    Real noy_wk[ncol];
    Real sox_wk[ncol];
    Real nhx_wk[ncol];
    Real wrk_wd[ncol];
    Real wght[ncol]; 
    
    //===========
    // output integrated wet deposition field
    //===========
    for(int i = 0; i < ncol; i++) {
        noy_wk[i] = 0;
        sox_wk[i] = 0;
        nhx_wk[i] = 0;
    }

    get_wght_all_p(lchnk, ncol, wght)

    do mm = 1,gas_pcnst
       !
       ! compute vertical integral
       !
       wrk_wd(:ncol) = 0._r8
       do kk = 1,pver
          wrk_wd(:ncol) = wrk_wd(:ncol) + het_rates(:ncol,kk,mm) * mmr(:ncol,kk,mm) * pdel(:ncol,kk) 
       enddo
       !
       wrk_wd(:ncol) = wrk_wd(:ncol) * rgrav * wght(:ncol) * rearth**2
       !

       call outfld( wetdep_name(mm), wrk_wd(:ncol),         ncol, lchnk )
       call outfld( wtrate_name(mm), het_rates(:ncol,:,mm), ncol, lchnk )

       if ( any(sox_species == mm ) ) then
          sox_wk(:ncol) = sox_wk(:ncol) + wrk_wd(:ncol)*S_molwgt/adv_mass(mm)
       endif

    enddo
    
    call outfld( 'WD_NOY', noy_wk(:ncol), ncol, lchnk )
    call outfld( 'WD_SOX', sox_wk(:ncol), ncol, lchnk )
    call outfld( 'WD_NHX', nhx_wk(:ncol), ncol, lchnk )

 } // het_diags

} //namespace mo_chm_diags
} //namespace mam4