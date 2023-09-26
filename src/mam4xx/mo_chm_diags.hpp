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
constexpr Real mwdry = haero::Constants::molec_weight_dry_air *
                       1e3; //     ! molecular weight dry air ~ kg/kmole;//!
                            //     molecular weight dry air
// constants for converting O3 mixing ratio to DU
constexpr Real DUfac = 2.687e20; // 1 DU in molecules per m^2
constexpr Real rearth = 6.37122e6;
constexpr Real rgrav =
    1.0 / 9.80616; // reciprocal of acceleration of gravity ~ m/s^2
constexpr Real avogadro = haero::Constants::avogadro;
constexpr const int gas_pcnst = gas_chemistry::gas_pcnst;
constexpr const int pcnst = 80; // FIXME, 80 is the only value I found for this
                                // in the fortran, but using 41 in the test
// number of vertical levels
constexpr const int pver = mam4::nlev;

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

namespace {

KOKKOS_INLINE_FUNCTION
size_t gpu_safe_strlen(const char *s) {
  size_t l = 0;
  while (s[l])
    ++l;
  return l;
}

// this helper function returns true if the name string contains the (literal)
// pattern string, false if not
KOKKOS_INLINE_FUNCTION
bool name_matches(const char *name, const char *pattern) {
  size_t name_len = gpu_safe_strlen(name),
         pattern_len = gpu_safe_strlen(pattern);
  if (pattern_len > name_len)
    return false;
  for (size_t i = 0; i < name_len; ++i) {
    if (name[i] != pattern[i])
      return false;
  }
  return true;
}

} // namespace

//========================================================================
// TODO: toth and tcly vars not actually used in the function...
KOKKOS_INLINE_FUNCTION
void chm_diags(
    const ThreadTeam &team, int lchnk, int ncol, int id_o3,
    const ColumnView vmr[gas_pcnst],      //[pver][gas_pcnst],
    const ColumnView mmr[gas_pcnst],      //[pver][gas_pcnst],
    const ColumnView &depvel,             //[gas_pcnst],
    const ColumnView &depflx,             //[gas_pcnst],
    const ColumnView mmr_tend[gas_pcnst], //[pver][gas_pcnst],
    const ColumnView &pdel,               //[pver],
    const ColumnView &pdeldry,            //[pver]
    const ColumnView fldcw[pcnst],        //[pver][pcnst],
    const int ltrop,        // index of the lowest stratospheric level
    const ColumnView &area, // [1], input and output
    const Real sox_species[3], const Real aer_species[gas_pcnst],
    const Real adv_mass[gas_pcnst], // constant from elsewhere
    const char solsym[gas_pcnst][17],
    // output fields
    const ColumnView &mass,        //[pver],
    const ColumnView &drymass,     //[pver],
    const ColumnView &ozone_layer, //[pver], ozone concentration [DU]
    const ColumnView &ozone_col,   // [1], vertical integration of ozone
    const ColumnView
        &ozone_trop, // [1], vertical integration of ozone in troposphere [DU]
    const ColumnView
        &ozone_strat, // [1], vertical integration of ozone instratosphere [DU]
    const ColumnView &vmr_nox,  //[pver],
    const ColumnView &vmr_noy,  //[pver],
    const ColumnView &vmr_clox, //[pver],
    const ColumnView &vmr_cloy, //[pver],
    const ColumnView &vmr_brox, //[pver],
    const ColumnView &vmr_broy, //[pver],
    const ColumnView &mmr_noy,  //[pver],
    const ColumnView &mmr_sox,  //[pver],
    const ColumnView &mmr_nhx,  //[pver],
    const ColumnView &net_chem, //[pver],
    const ColumnView &df_noy,   //[1],
    const ColumnView &df_sox,   //[1],
    const ColumnView &df_nhx,   //[1],
    const ColumnView &mass_bc,  //[pver],
    const ColumnView &mass_dst, //[pver],
    const ColumnView &mass_mom, //[pver],
    const ColumnView &mass_ncl, //[pver],
    const ColumnView &mass_pom, //[pver],
    const ColumnView &mass_so4, //[pver],
    const ColumnView &mass_soa  //[pver],
) {

  //--------------------------------------------------------------------
  //	... local variables
  //--------------------------------------------------------------------

  Real wgt;
  // Real pointer :: fldcw(:,:)  //working pointer to extract data from pbuf for
  // sum of mass for aerosol classes

  //---------------------------not
  // needed?-------------------------------------------- bool history_aerosol;
  // //
  // output aerosol variables bool history_verbose;      // produce verbose
  // history output

  // call phys_getopts( history_aerosol_out = history_aerosol, &
  //                   history_verbose_out = history_verbose )
  //----------------------------------------------------------------------------------

  //--------------------------------------------------------------------
  //	... "diagnostic" groups
  //--------------------------------------------------------------------
  for (int kk = 0; kk < pver; kk++) {
    vmr_nox(kk) = 0;
    vmr_noy(kk) = 0;
    vmr_clox(kk) = 0;
    vmr_cloy(kk) = 0;
    vmr_brox(kk) = 0;
    vmr_broy(kk) = 0;
    mmr_noy(kk) = 0;
    mmr_sox(kk) = 0;
    mmr_nhx(kk) = 0;
  }
  df_noy(0) = 0;
  df_sox(0) = 0;
  df_nhx(0) = 0;

  // Save the sum of mass mixing ratios for each class instea of individual
  // species to reduce history file size

  // Mass_bc = bc_a1 + bc_c1 + bc_a3 + bc_c3 + bc_a4 + bc_c4
  // Mass_dst = dst_a1 + dst_c1 + dst_a3 + dst_c3
  // Mass_mom = mom_a1 + mom_c1 + mom_a2 + mom_c2 + mom_a3 + mom_c3 + mom_a4 +
  // mom_c4 Mass_ncl = ncl_a1 + ncl_c1 + ncl_a2 + ncl_c2 + ncl_a3 + ncl_c3
  // Mass_pom = pom_a1 + pom_c1 + pom_a3 + pom_c3 + pom_a4 + pom_c4
  // Mass_so4 = so4_a1 + so4_c1 + so4_a2 + so4_c2 + so4_a3 + so4_c3
  // Mass_soa = soa_a1 + soa_c1 + soa_a2 + soa_c2 + soa_a3 + soa_c3

  // initialize the mass arrays
  // if (history_aerosol .and. .not. history_verbose) then // what was this used
  // for?
  for (int kk = 0; kk < pver; kk++) {
    mass_bc(kk) = 0;
    mass_dst(kk) = 0;
    mass_mom(kk) = 0;
    mass_ncl(kk) = 0;
    mass_pom(kk) = 0;
    mass_so4(kk) = 0;
    mass_soa(kk) = 0;
  }

  area(0) *= haero::square(rearth);

  for (int kk = 0; kk < pver; kk++) {
    mass(kk) = pdel(kk) * area(0) * rgrav;
    drymass(kk) = pdeldry(kk) * area(0) * rgrav;
  }

  // convert ozone from mol/mol (w.r.t. dry air mass) to DU
  for (int kk = 0; kk < pver; kk++) {
    ozone_layer(kk) =
        pdeldry(kk) * vmr[id_o3](kk) * avogadro * rgrav / mwdry / DUfac * 1e3;
  }
  // total column ozone
  ozone_col(0) = 0;
  ozone_trop(0) = 0;
  ozone_strat(0) = 0;

  for (int kk = 0; kk < pver; kk++) {
    ozone_col(0) += ozone_layer(kk);
    if (kk <= ltrop) {
      // stratospheric column ozone
      ozone_strat(0) += ozone_layer(kk);
    } else {
      // tropospheric column ozone
      ozone_trop(0) += ozone_layer(kk);
    }
  }

  for (int mm = 0; mm < gas_pcnst; mm++) {
    // other options of species are not used, only use weight=1
    wgt = 1;

    for (int i = 0; i < 3; i++) { // FIXME: bad constant (len of sox species)
      if (sox_species[i] == mm) {
        for (int kk = 0; kk < pver; kk++) {
          mmr_sox(kk) = mmr_sox(kk) + wgt * mmr[mm](kk);
        }
      }
    }

    if (aer_species[mm] == mm) {
      const char *symbol = solsym[mm];
      if (name_matches(symbol, "bc_a")) {
        for (int kk = 0; kk < pver; kk++) {
          mass_bc(kk) += mmr[mm](kk);
        }
      } else if (name_matches(symbol, "dst_a")) {
        for (int kk = 0; kk < pver; kk++) {
          mass_dst(kk) += mmr[mm](kk);
        }
      } else if (name_matches(symbol, "mom_a")) {
        for (int kk = 0; kk < pver; kk++) {
          mass_mom(kk) += mmr[mm](kk);
        }
      } else if (name_matches(symbol, "ncl_a")) {
        for (int kk = 0; kk < pver; kk++) {
          mass_ncl(kk) += mmr[mm](kk);
        }
      } else if (name_matches(symbol, "pom_a")) {
        for (int kk = 0; kk < pver; kk++) {
          mass_pom(kk) += mmr[mm](kk);
        }
      } else if (name_matches(symbol, "so4_a")) {
        for (int kk = 0; kk < pver; kk++) {
          mass_so4(kk) += mmr[mm](kk);
        }
      } else if (name_matches(symbol, "soa_a")) {
        for (int kk = 0; kk < pver; kk++) {
          mass_soa(kk) += mmr[mm](kk);
        }
      }
    }

    for (int i = 0; i < 3; i++) { // FIXME: bad constant (len of sox species)
      if (sox_species[i] == mm) {
        df_sox(0) += wgt * depflx[mm] * S_molwgt / adv_mass[mm];
      }
    }

    for (int kk = 0; kk < pver; kk++) {
      net_chem(kk) = mmr_tend[mm](kk) * mass(kk);
    }
  }

  // diagnostics for cloud-borne aerosols, then add to corresponding mass
  // accumulators
  // if (history_aerosol.and..not .history_verbose) then

  for (int nn = lchnk; nn < pcnst; nn++) {
    // fldcw = > qqcw_get_field(pbuf, nn, lchnk, errorhandle =.true.)
    // if(associated(fldcw)) then
    // NOTE: The "cloud-water" constituent name are the same as their "aerosol"
    // NOTE: counterparts with "_a" (and "_A") replaced by "_c" (and "_C").
    // NOTE: See initaermodes_set_cnstnamecw in
    // NOTE: eam/src/chemistry/modal_aero/modal_aero_initialize_data.F90.
    const char *symbol = solsym[nn];
    char symbol_cw[17];
    {
      size_t symbol_len = gpu_safe_strlen(symbol);
      bool change_next = false;
      for (size_t i = 0; i < symbol_len; ++i) {
        if (symbol[i] == 'a' && change_next) {
          symbol_cw[i] = 'c';
        } else if (symbol[i] == '_') {
          change_next = true;
        } else {
          symbol_cw[i] = symbol[i];
        }
      }
    }
    if (name_matches(symbol_cw, "bc_c")) {
      for (int kk = 0; kk < pver; kk++) {
        mass_bc(kk) += fldcw[nn](kk);
      }
    } else if (name_matches(symbol_cw, "dst_c")) {
      for (int kk = 0; kk < pver; kk++) {
        mass_dst(kk) += fldcw[nn](kk);
      }
    } else if (name_matches(symbol_cw, "mom_c")) {
      for (int kk = 0; kk < pver; kk++) {
        mass_mom(kk) += fldcw[nn](kk);
      }
    } else if (name_matches(symbol_cw, "ncl_c")) {
      for (int kk = 0; kk < pver; kk++) {
        mass_ncl(kk) += fldcw[nn](kk);
      }
    } else if (name_matches(symbol_cw, "pom_c")) {
      for (int kk = 0; kk < pver; kk++) {
        mass_pom(kk) += fldcw[nn](kk);
      }
    } else if (name_matches(symbol_cw, "so4_c")) {
      for (int kk = 0; kk < pver; kk++) {
        mass_so4(kk) += fldcw[nn](kk);
      }
    } else if (name_matches(symbol_cw, "soa_c")) {
      for (int kk = 0; kk < pver; kk++) {
        mass_soa(kk) += fldcw[nn](kk);
      }
    }
  }
} // chm_diags
} // namespace mo_chm_diags
} // namespace mam4
#endif
