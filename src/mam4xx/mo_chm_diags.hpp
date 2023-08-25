#ifndef MAM4XX_MO_CHM_DIAGS_HPP
#define MAM4XX_MO_CHM_DIAGS_HPP

#include <haero/math.hpp>
#include <mam4xx/mam4_types.hpp>
#include <mam4xx/utils.hpp>
#include <mam4xx/gas_chem.hpp>
#include <mam4xx/phys_grid.hpp>

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

    Real sox_species[3];
    id_so2     = get_spc_ndx( 'SO2' )
    id_so4     = get_spc_ndx( 'SO4' )
    id_h2so4   = get_spc_ndx( 'H2SO4' )
    sox_species = (/ id_so2, id_so4, id_h2so4 /)

KOKKOS_INLINE_FUNCTION
void het_diags(Real het_rates[ncol][pver][gas_pcnst], //inout
               Real mmr[ncol][pver][gas_pcnst],
               Real pdel[ncol][pver], 
               int lchnk, 
               int ncol, 
               Real wght[ncol],
               Real wrk_wd[ncol], //output
               Real noy_wk[ncol], //output
               Real sox_wk[ncol], //output
               Real nhx_wk[ncol], //output
               ) {

    //===========
    // output integrated wet deposition field
    //===========
    for(int i = 0; i < ncol; i++) {
        noy_wk[i] = 0; //this isn't actually used in this function?
        sox_wk[i] = 0;
        nhx_wk[i] = 0; //this isn't actually used in this function?
    }

    for(int mm = 0; mm < gas_pcnst; mm++) {
       //
       // compute vertical integral
       //
       for(int i = 0; i < ncol; i++) 
         wrk_wd[i] = 0;

       for(int kk = 1; kk < pver; kk++) {
         for(int i = 0; i < ncol; i++) 
          wrk_wd[i] = het_rates[i][kk][mm] * mmr[i][kk][mm] * pdel[i][kk];
       }
       
       for(int i = 0; i < ncol; i++) 
         wrk_wd[i] *= rgrav * wght[i] * haero::pow(rearth, 2);

       if( any(sox_species == mm ) ) {
         for(int i = 0; i < ncol; i++) 
            sox_wk[i] += wrk_wd[i] * S_molwgt / adv_mass(mm);
       }

    }


 } // het_diags

 //========================================================================
KOKKOS_INLINE_FUNCTION 
void chm_diags(int lchnk, int ncol, Real vmr[ncol][pver][gas_pcnst], 
               Real mmr[ncol][pver][gas_pcnst],
               Real depvel[ncol][gas_pcnst], 
               Real depflx[ncol][gas_pcnst], 
               Real mmr_tend[ncol][pver][gas_pcnst], 
               Real pdel[ncol][pver],
               Real pdeldry[ncol][pver][gas_pcnst],
               Real* pbuf, 
               Real ltrop[pcols] // index of the lowest stratospheric level
               ) {
    
    //--------------------------------------------------------------------
    //	... local variables
    //--------------------------------------------------------------------
    integer     :: icol,kk, mm, nn
    real(r8)    :: ozone_layer(ncol,pver)   // ozone concentration [DU]
    real(r8)    :: ozone_col(ncol)          // vertical integration of ozone [DU]
    real(r8)    :: ozone_trop(ncol)         // vertical integration of ozone in troposphere [DU]
    real(r8)    :: ozone_strat(ncol)        // vertical integration of ozone in stratosphere [DU]
    
    real(r8), dimension(ncol,pver) :: vmr_nox, vmr_noy, vmr_clox, vmr_cloy, vmr_tcly, vmr_brox, vmr_broy, vmr_toth
    real(r8), dimension(ncol,pver) :: mmr_noy, mmr_sox, mmr_nhx, net_chem
    real(r8), dimension(ncol)      :: df_noy, df_sox, df_nhx

    real(r8) :: area(ncol), mass(ncol,pver), drymass(ncol,pver)
    real(r8) :: wgt
    character(len=16) :: spc_name
    real(r8), pointer :: fldcw(:,:)  //working pointer to extract data from pbuf for sum of mass for aerosol classes
    real(r8), dimension(ncol,pver) :: mass_bc, mass_dst, mass_mom, mass_ncl, mass_pom, mass_so4, mass_soa

    logical :: history_aerosol      // output aerosol variables
    logical :: history_verbose      // produce verbose history output

    //-----------------------------------------------------------------------

    call phys_getopts( history_aerosol_out = history_aerosol, &
                       history_verbose_out = history_verbose )
    //--------------------------------------------------------------------
    //	... "diagnostic" groups
    //--------------------------------------------------------------------
    vmr_nox(:ncol,:) = 0._r8
    vmr_noy(:ncol,:) = 0._r8
    vmr_clox(:ncol,:) = 0._r8
    vmr_cloy(:ncol,:) = 0._r8
    vmr_tcly(:ncol,:) = 0._r8
    vmr_brox(:ncol,:) = 0._r8
    vmr_broy(:ncol,:) = 0._r8
    vmr_toth(:ncol,:) = 0._r8
    mmr_noy(:ncol,:) = 0._r8
    mmr_sox(:ncol,:) = 0._r8
    mmr_nhx(:ncol,:) = 0._r8
    df_noy(:ncol) = 0._r8
    df_sox(:ncol) = 0._r8
    df_nhx(:ncol) = 0._r8

    // Save the sum of mass mixing ratios for each class instea of individual
    // species to reduce history file size

    // Mass_bc = bc_a1 + bc_c1 + bc_a3 + bc_c3 + bc_a4 + bc_c4
    // Mass_dst = dst_a1 + dst_c1 + dst_a3 + dst_c3
    // Mass_mom = mom_a1 + mom_c1 + mom_a2 + mom_c2 + mom_a3 + mom_c3 + mom_a4 + mom_c4
    // Mass_ncl = ncl_a1 + ncl_c1 + ncl_a2 + ncl_c2 + ncl_a3 + ncl_c3
    // Mass_pom = pom_a1 + pom_c1 + pom_a3 + pom_c3 + pom_a4 + pom_c4
    // Mass_so4 = so4_a1 + so4_c1 + so4_a2 + so4_c2 + so4_a3 + so4_c3
    // Mass_soa = soa_a1 + soa_c1 + soa_a2 + soa_c2 + soa_a3 + soa_c3

    //initialize the mass arrays
    if (history_aerosol .and. .not. history_verbose) then
       mass_bc(:ncol,:) = 0._r8
       mass_dst(:ncol,:) = 0._r8
       mass_mom(:ncol,:) = 0._r8
       mass_ncl(:ncol,:) = 0._r8
       mass_pom(:ncol,:) = 0._r8
       mass_so4(:ncol,:) = 0._r8
       mass_soa(:ncol,:) = 0._r8
    endif

    call get_area_all_p(lchnk, ncol, area)
    area = area * rearth**2

    do kk = 1,pver
       mass(:ncol,kk) = pdel(:ncol,kk) * area(:ncol) * rgrav
       drymass(:ncol,kk) = pdeldry(:ncol,kk) * area(:ncol) * rgrav
    enddo

    call outfld( 'AREA', area(:ncol),   ncol, lchnk )
    call outfld( 'MASS', mass(:ncol,:), ncol, lchnk )
    call outfld( 'DRYMASS', drymass(:ncol,:), ncol, lchnk )

    // convert ozone from mol/mol (w.r.t. dry air mass) to DU
    ozone_layer(:ncol,:) = pdeldry(:ncol,:)*vmr(:ncol,:,id_o3)*avogadro*rgrav/mwdry/DUfac*1.e3_r8
    // total column ozone
    ozone_col(:) = 0._r8
    ozone_trop(:) = 0._r8
    ozone_strat(:) = 0._r8
    do icol = 1,ncol
       do kk = 1,pver
          ozone_col(icol) = ozone_col(icol) + ozone_layer(icol,kk)
          if (kk <= ltrop(icol)) then
             ozone_strat(icol) = ozone_strat(icol) + ozone_layer(icol,kk)
          else
             ozone_trop(icol) = ozone_trop(icol) + ozone_layer(icol,kk)
          endif
       enddo
    enddo
    call outfld( 'TOZ', ozone_col, ncol, lchnk )
    // stratospheric column ozone
    call outfld( 'SCO', ozone_strat, ncol, lchnk )
    // tropospheric column ozone
    call outfld( 'TCO', ozone_trop, ncol, lchnk )

    do mm = 1,gas_pcnst

      // other options of species are not used, only use weight=1
       wgt = 1._r8

       if ( any( sox_species == mm ) ) then
          mmr_sox(:ncol,:) = mmr_sox(:ncol,:) +  wgt * mmr(:ncol,:,mm)
       endif
       
       if ( any( aer_species == mm ) ) then
          call outfld( solsym(mm), mmr(:ncol,:,mm), ncol ,lchnk )
          call outfld( trim(solsym(mm))//'_SRF', mmr(:ncol,pver,mm), ncol ,lchnk )
          if (history_aerosol .and. .not. history_verbose) then
             select case (trim(solsym(mm)))
             case ('bc_a1','bc_a3','bc_a4')
                  mass_bc(:ncol,:) = mass_bc(:ncol,:) + mmr(:ncol,:,mm)
             case ('dst_a1','dst_a3')
                  mass_dst(:ncol,:) = mass_dst(:ncol,:) + mmr(:ncol,:,mm)
             case ('mom_a1','mom_a2','mom_a3','mom_a4')
                  mass_mom(:ncol,:) = mass_mom(:ncol,:) + mmr(:ncol,:,mm)
             case ('ncl_a1','ncl_a2','ncl_a3')
                  mass_ncl(:ncol,:) = mass_ncl(:ncol,:) + mmr(:ncol,:,mm)
             case ('pom_a1','pom_a3','pom_a4')
                  mass_pom(:ncol,:) = mass_pom(:ncol,:) + mmr(:ncol,:,mm)
             case ('so4_a1','so4_a2','so4_a3')
                  mass_so4(:ncol,:) = mass_so4(:ncol,:) + mmr(:ncol,:,mm)
             case ('soa_a1','soa_a2','soa_a3')
                  mass_soa(:ncol,:) = mass_soa(:ncol,:) + mmr(:ncol,:,mm)
             endselect
          endif
       else
          call outfld( solsym(mm), vmr(:ncol,:,mm), ncol ,lchnk )
          call outfld( trim(solsym(mm))//'_SRF', vmr(:ncol,pver,mm), ncol ,lchnk )
       endif

       call outfld( depvel_name(mm), depvel(:ncol,mm), ncol ,lchnk )
       call outfld( depflx_name(mm), depflx(:ncol,mm), ncol ,lchnk )

       if ( any( sox_species == mm ) ) then
          df_sox(:ncol) = df_sox(:ncol) +  wgt * depflx(:ncol,mm)*S_molwgt/adv_mass(mm)
       endif

       do kk=1,pver
          do icol=1,ncol
             net_chem(icol,kk) = mmr_tend(icol,kk,mm) * mass(icol,kk) 
          enddo
       enddo
       call outfld( dtchem_name(mm), net_chem(:ncol,:), ncol, lchnk )

    enddo

    // diagnostics for cloud-borne aerosols, then add to corresponding mass accumulators
    if (history_aerosol .and. .not. history_verbose) then

       do nn = 1,pcnst
          fldcw => qqcw_get_field(pbuf,nn,lchnk,errorhandle=.true.)
          if(associated(fldcw)) then
             select case (trim(cnst_name_cw(nn)))
                case ('bc_c1','bc_c3','bc_c4')
                     mass_bc(:ncol,:) = mass_bc(:ncol,:) + fldcw(:ncol,:)
                case ('dst_c1','dst_c3')
                     mass_dst(:ncol,:) = mass_dst(:ncol,:) + fldcw(:ncol,:)
                case ('mom_c1','mom_c2','mom_c3','mom_c4')
                     mass_mom(:ncol,:) = mass_mom(:ncol,:) + fldcw(:ncol,:)
                case ('ncl_c1','ncl_c2','ncl_c3')
                     mass_ncl(:ncol,:) = mass_ncl(:ncol,:) + fldcw(:ncol,:)
                case ('pom_c1','pom_c3','pom_c4')
                     mass_pom(:ncol,:) = mass_pom(:ncol,:) + fldcw(:ncol,:)
                case ('so4_c1','so4_c2','so4_c3')
                     mass_so4(:ncol,:) = mass_so4(:ncol,:) + fldcw(:ncol,:)
                case ('soa_c1','soa_c2','soa_c3')
                     mass_soa(:ncol,:) = mass_soa(:ncol,:) + fldcw(:ncol,:)
             endselect
          endif
       enddo
       call outfld( 'Mass_bc', mass_bc(:ncol,:),ncol,lchnk)
       call outfld( 'Mass_dst', mass_dst(:ncol,:),ncol,lchnk)
       call outfld( 'Mass_mom', mass_mom(:ncol,:),ncol,lchnk)
       call outfld( 'Mass_ncl', mass_ncl(:ncol,:),ncol,lchnk)
       call outfld( 'Mass_pom', mass_pom(:ncol,:),ncol,lchnk)
       call outfld( 'Mass_so4', mass_so4(:ncol,:),ncol,lchnk)
       call outfld( 'Mass_soa', mass_soa(:ncol,:),ncol,lchnk)
    endif

    call outfld( 'NOX',  vmr_nox(:ncol,:),  ncol, lchnk )
    call outfld( 'NOY',  vmr_noy(:ncol,:),  ncol, lchnk )
    call outfld( 'CLOX', vmr_clox(:ncol,:), ncol, lchnk )
    call outfld( 'CLOY', vmr_cloy(:ncol,:), ncol, lchnk )
    call outfld( 'BROX', vmr_brox(:ncol,:), ncol, lchnk )
    call outfld( 'BROY', vmr_broy(:ncol,:), ncol, lchnk )
    call outfld( 'TCLY', vmr_tcly(:ncol,:), ncol, lchnk )
    call outfld( 'NOY_mmr', mmr_noy(:ncol,:), ncol ,lchnk )
    call outfld( 'SOX_mmr', mmr_sox(:ncol,:), ncol ,lchnk )
    call outfld( 'NHX_mmr', mmr_nhx(:ncol,:), ncol ,lchnk )
    call outfld( 'DF_NOY', df_noy(:ncol), ncol ,lchnk )
    call outfld( 'DF_SOX', df_sox(:ncol), ncol ,lchnk )
    call outfld( 'DF_NHX', df_nhx(:ncol), ncol ,lchnk )


  }  //chm_diags

} //namespace mo_chm_diags
} //namespace mam4