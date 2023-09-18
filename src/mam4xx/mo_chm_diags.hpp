#ifndef MAM4XX_MO_CHM_DIAGS_HPP
#define MAM4XX_MO_CHM_DIAGS_HPP

#include <haero/math.hpp>
#include <mam4xx/gas_chem.hpp>
#include <mam4xx/mam4_types.hpp>
#include <mam4xx/utils.hpp>

namespace mam4 {

namespace mo_chm_diags {

using Real = haero::Real;
using View1D = DeviceType::view_1d<Real>;

// FIXME: bad constants
constexpr Real S_molwgt = 32.066;
// constants for converting O3 mixing ratio to DU
constexpr Real DUfac = 2.687e20; // 1 DU in molecules per m^2
constexpr Real rearth = 6.37122e6;
constexpr Real rgrav =
    1.0 / 9.80616; // reciprocal of acceleration of gravity ~ m/s^2
constexpr Real avogadro = haero::Constants::avogadro;
constexpr const int gas_pcnst = gas_chemistry::gas_pcnst;
/*char solsymmmm[gas_pcnst][17] = {"O3              ","H2O2            ","H2SO4
   ","SO2             ","DMS             ", "SOAG            ","so4_a1 ","pom_a1
   ","soa_a1          ","bc_a1           ", "dst_a1          ","ncl_a1 ","mom_a1
   ","num_a1          ","so4_a2          ", "soa_a2          ","ncl_a2 ","mom_a2
   ","num_a2          ","dst_a3          ", "ncl_a3          ","so4_a3 ","bc_a3
   ","pom_a3          ","soa_a3          ", "mom_a3          ","num_a3 ","pom_a4
   ","bc_a4           ","mom_a4          ", "num_a4          "}; //solution
   system
*/
// number of vertical levels
constexpr const int pver = mam4::nlev;

/* will be ported from set_sox
    Real sox_species[3];
    id_so2     = get_spc_ndx( 'SO2' )
    id_so4     = get_spc_ndx( 'SO4' )
    id_h2so4   = get_spc_ndx( 'H2SO4' )
    sox_species = (/ id_so2, id_so4, id_h2so4 /)
*/

KOKKOS_INLINE_FUNCTION
void het_diags(
    const ThreadTeam &team,
    const ColumnView het_rates[gas_pcnst], //[pver][gas_pcnst], //in
    const ColumnView mmr[gas_pcnst],       //[pver][gas_pcnst],
    const ColumnView &pdel,                //[pver],
    const Real &wght,
    View1D wrk_wd, //[gas_pcnst], //output
    // Real noy_wk, //output //this isn't actually used in this function?
    View1D sox_wk, // output
    // Real nhx_wk, //output //this isn't actually used in this function?
    const Real adv_mass[gas_pcnst], // constant from elsewhere
    const Real sox_species[3]) {
  // change to pass values for a single col
  //===========
  // output integrated wet deposition field
  //===========

  sox_wk[0] = 0;
  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team, gas_pcnst), KOKKOS_LAMBDA(int mm) {
        //
        // compute vertical integral
        //
        wrk_wd(mm) = 0;

        for (int kk = 0; kk < pver; kk++) {
          wrk_wd(mm) += het_rates[mm](kk) * mmr[mm](kk) *
                        pdel(kk); // parallel_reduce in the future?
        }

        wrk_wd(mm) *= rgrav * wght * haero::square(rearth);
      });

  for (int mm = 0; mm < gas_pcnst; mm++) {
    for (int i = 0; i < 3; i++) { // FIXME: bad constant (len of sox species)
      if (sox_species[i] == mm)
        sox_wk[0] += wrk_wd[mm] * S_molwgt / adv_mass[mm];
    }
  }
} // het_diags

/*
 //========================================================================
KOKKOS_INLINE_FUNCTION
void chm_diags(int lchnk, int ncol,
               const ColumnView vmr[gas_pcnst], //[pver][gas_pcnst],
               const ColumnView mmr[gas_pcnst], //[pver][gas_pcnst],
               Real depvel[gas_pcnst],
               Real depflx[gas_pcnst],
               const ColumnView mmr_tend[gas_pcnst], //[pver][gas_pcnst],
               const ColumnView &pdel, //[pver],
               const ColumnView &pdeldry, //[pver]
               Real* pbuf,
               Real ltrop[pcols], // index of the lowest stratospheric level
               Real area,  // NEW input from host model (and output)
               const Real sox_species[3],
               const Real aer_species[gas_pcnst],
               const Real adv_mass[gas_pcnst], //constant from elsewhere
               //output fields
               const ColumnView &mass,
               const ColumnView &drymass,
               const ColumnView &ozone_layer, //[pver], ozone concentration [DU]
               Real &ozone_col,          // vertical integration of ozone [DU]
               Real &ozone_trop,         // vertical integration of ozone in
troposphere [DU] Real &ozone_strat,        // vertical integration of ozone in
stratosphere [DU] const ColumnView &vmr_nox, //[pver], const ColumnView
&vmr_noy, //[pver], const ColumnView &vmr_clox, //[pver], const ColumnView
&vmr_cloy, //[pver], const ColumnView &vmr_brox, //[pver], const ColumnView
&vmr_broy, //[pver], const ColumnView &vmr_toth, //[pver], const ColumnView
&mmr_noy, //[pver], const ColumnView &mmr_sox, //[pver], const ColumnView
&net_chem, //[pver], Real &df_noy, Real &df_sox, Real &df_nhx, const ColumnView
&mass_bc, //[pver], const ColumnView &mass_dst, //[pver], const ColumnView
&mass_mom, //[pver], const ColumnView &mass_ncl, //[pver], const ColumnView
&mass_pom, //[pver], const ColumnView &mass_so4, //[pver], const ColumnView
&mass_soa, //[pver],

               char solsymm[gas_pcnst][17],
               //Real depvel_name[gas_pcnst], //not needed bc the outfld
depvel_name is just outputting depvel?
               //Real depflx_name[gas_pcnst], //not needed bc the outfld
depflx_name is just outputting depflx?

               ) {

    //--------------------------------------------------------------------
    //	... local variables
    //--------------------------------------------------------------------

    Real wgt;
    char spc_name[16];
    //Real pointer :: fldcw(:,:)  //working pointer to extract data from pbuf
for sum of mass for aerosol classes



//---------------------------not
needed?--------------------------------------------
    //bool history_aerosol;      // output aerosol variables
    //bool history_verbose;      // produce verbose history output

    //call phys_getopts( history_aerosol_out = history_aerosol, &
     //                  history_verbose_out = history_verbose )
//----------------------------------------------------------------------------------

    //--------------------------------------------------------------------
    //	... "diagnostic" groups
    //--------------------------------------------------------------------
   for(int kk = 1; kk < pver; kk++) {

      vmr_nox(kk) = 0;
      vmr_noy(kk) = 0;
      vmr_clox(kk) = 0;
      vmr_cloy(kk) = 0;
      vmr_tcly(kk) = 0;
      vmr_brox(kk) = 0;
      vmr_broy(kk) = 0;
      vmr_toth(kk) = 0;
      mmr_noy(kk) = 0;
      mmr_sox(kk) = 0;
      mmr_nhx(kk) = 0;
   }
   df_noy = 0;
   df_sox = 0;
   df_nhx = 0;

    // Save the sum of mass mixing ratios for each class instea of individual
    // species to reduce history file size

    // Mass_bc = bc_a1 + bc_c1 + bc_a3 + bc_c3 + bc_a4 + bc_c4
    // Mass_dst = dst_a1 + dst_c1 + dst_a3 + dst_c3
    // Mass_mom = mom_a1 + mom_c1 + mom_a2 + mom_c2 + mom_a3 + mom_c3 + mom_a4 +
mom_c4
    // Mass_ncl = ncl_a1 + ncl_c1 + ncl_a2 + ncl_c2 + ncl_a3 + ncl_c3
    // Mass_pom = pom_a1 + pom_c1 + pom_a3 + pom_c3 + pom_a4 + pom_c4
    // Mass_so4 = so4_a1 + so4_c1 + so4_a2 + so4_c2 + so4_a3 + so4_c3
    // Mass_soa = soa_a1 + soa_c1 + soa_a2 + soa_c2 + soa_a3 + soa_c3

    //initialize the mass arrays
    //if (history_aerosol .and. .not. history_verbose) then // what was this
used for? for(int kk = 1; kk < pver; kk++) { mass_bc(kk) = 0; mass_dst(kk) = 0;
      mass_mom(kk) = 0;
      mass_ncl(kk) = 0;
      mass_pom(kk) = 0;
      mass_so4(kk) = 0;
      mass_soa(kk) = 0;
   }
    //endif

   area *= haero::square(rearth);

   for(int kk = 1; kk < pver; kk++) {
      mass(kk) = pdel(kk) * area * rgrav;
      drymass(kk) = pdeldry(kk) * area * rgrav;
   }

   // convert ozone from mol/mol (w.r.t. dry air mass) to DU
   for(int kk = 1; kk < pver; kk++) {
   ozone_layer(kk) = pdeldry(kk)*vmr(kk)[id_o3]*avogadro*rgrav/mwdry/DUfac*1e3;
   }
   // total column ozone
   ozone_col = 0;
   ozone_trop = 0;
   ozone_strat = 0;

   for(int kk = 1; kk < pver; kk++) {
      ozone_col += ozone_layer(kk);
      if (kk <= ltrop) {
         // stratospheric column ozone
         ozone_strat += ozone_layer(kk);
      } else {
         // tropospheric column ozone
         ozone_trop += ozone_layer(kk);
      }
   }

   for(int mm = 0; mm < gas_pcnst; mm++) {
      // other options of species are not used, only use weight=1
       wgt = 1;

       for(int i = 0; i < 3; i++) { //FIXME: bad constant (len of sox species)
         if(sox_species[i] == mm) {
            for(int kk = 1; kk < pver; kk++) {
               mmr_sox(kk) = mmr_sox(kk) +  wgt * mmr[mm](kk);
            }
         }
      }

      if(aer_species[mm] == mm) {
         if (history_aerosol .and. .not. history_verbose) then
            switch (trim(solsym(mm))) {
            case 'bc_a1','bc_a3','bc_a4':
               for(int kk = 1; kk < pver; kk++) {
                  mass_bc(kk) += mmr[mm](kk);
               }
               break;
            case 'dst_a1','dst_a3':
               for(int kk = 1; kk < pver; kk++) {
                  mass_dst(kk) += mmr[mm](kk);
               }
               break;
            case 'mom_a1','mom_a2','mom_a3','mom_a4':
               for(int kk = 1; kk < pver; kk++) {
                  mass_mom(kk) += mmr[mm](kk);
               }
               break;
            case 'ncl_a1','ncl_a2','ncl_a3':
               for(int kk = 1; kk < pver; kk++) {
                  mass_ncl(kk) += mmr[mm](kk);
               }
               break;
            case 'pom_a1','pom_a3','pom_a4':
               for(int kk = 1; kk < pver; kk++) {
                  mass_pom(kk) += mmr[mm](kk);
               }
               break;
            case 'so4_a1','so4_a2','so4_a3':
               for(int kk = 1; kk < pver; kk++) {
                  mass_so4(kk) += mmr[mm](kk);
               }
               break;
            case 'soa_a1','soa_a2','soa_a3':
               for(int kk = 1; kk < pver; kk++) {
                  mass_soa(kk) += mmr[mm](kk);
               }
               break;
            }
         endif
      }

      for(int i = 0; i < 3; i++) { //FIXME: bad constant (len of sox species)
         if(sox_species[i] == mm) {
            df_sox += wgt * depflx[mm]*S_molwgt/adv_mass(mm);
         }
      }

      for(int kk = 1; kk < pver; kk++) {
         net_chem(kk) = mmr_tend[mm](kk) * mass(kk);
      }

   }

    // diagnostics for cloud-borne aerosols, then add to corresponding mass
accumulators if (history_aerosol .and. .not. history_verbose) then

      for(int nn = 0; nn < pcnst; nn++) {
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
      }

    endif

  }  //chm_diags
*/
} // namespace mo_chm_diags
} // namespace mam4
#endif