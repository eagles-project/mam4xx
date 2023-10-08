// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MAM4XX_MODAL_AERO_CALCSIZE_HPP
#define MAM4XX_MODAL_AERO_CALCSIZE_HPP

#include <haero/atmosphere.hpp>
#include <haero/math.hpp>
#include <haero/surface.hpp>

#include <mam4xx/aero_config.hpp>
#include <mam4xx/conversions.hpp>
#include <mam4xx/mam4_types.hpp>
#include <mam4xx/utils.hpp>

namespace mam4 {
namespace modal_aero_calcsize {

constexpr int maxd_aspectype = ndrop::maxd_aspectype;

// size_adjustment(list_idx_local, top_lev, ncol, lchnk, imode, dryvol_a,
// state_q,
//                 &!input dryvol_c, pdel, do_adjust, update_mmr,
//                 do_aitacc_transfer, deltatinv, fracadj, qqcw, &!input
//                 dgncur_a, dgncur_c, v2ncur_a, v2ncur_c, &!output drv_a_accsv,
//                 drv_c_accsv, drv_a_aitsv, drv_c_aitsv, drv_a_sv, drv_c_sv,
//                 &!output num_a_accsv, num_c_accsv, num_a_aitsv, num_c_aitsv,
//                 num_a_sv, num_c_sv, &!output dotend, dotendqqcw, dqdt,
//                 dqqcwdt, qsrflx)

KOKKOS_INLINE_FUNCTION void size_adjustment(const int imode, 
	const Real dryvol_i, const Real n_i_imode, const Real dryvol_c, 
	 // pdel
	const bool do_adjust, const bool update_mmr, const bool do_aitacc_transfer,  
	const Real adj_tscale_inv,
	const Real dt,  const Real n_c_imode, 
	// additional parameters
	const Real num2vol_ratio_min_imode,
    const Real num2vol_ratio_max_imode, 
    const Real dgnmin_nmodes_imode,
    const Real dgnmax_nmodes_imode,
    const Real mean_std_dev_nmodes_imode, 
    const int accumulation_idx, 
    const int aitken_idx,
    // outputs
	Real &dgncur_i_imode, Real &dgncur_c_imode, Real & num2vol_ratio_cur_i,
	Real &num2vol_ratio_cur_c,
	Real &dryvol_i_accsv , Real &dryvol_c_accsv, 
	Real &dryvol_i_aitsv, Real &dryvol_c_aitsv,
	Real &drv_i_sv, Real &drv_c_sv,
	Real &num_i_k_accsv,
	Real &num_c_k_accsv, 
	Real& num_i_k_aitsv, 
	Real& num_c_k_aitsv,
	Real &num_i_sv,
	Real &num_c_sv, 
	Real & dnidt_imode, Real& dncdt_imode
	) {

	constexpr Real zero=0;

      /*-----------------------------------------------------------------------------
      !Purpose: Do the aerosol size adjustment if needed
      !
      !Called by: modal_aero_calcsize_sub
      !Calls    : compute_dgn_vol_limits, adjust_num_sizes, update_dgn_voltonum
      !
      !Author: Richard Easter (Refactored by Balwinder Singh)
      !-----------------------------------------------------------------------------*/

      /*find state q array number mode indices for interstitial and cloud borne
      aerosols !Both num_mode_idx and num_cldbrn_mode_idx should be exactly same
      and should be same !for both prognostic and diagnostic radiation lists
    */
      // const int  num_mode_idx        = numptr_amode[imode];
      // const int num_cldbrn_mode_idx = numptrcw_amode[imode];

      // const int mam_ait = modeptr_aitken; //!aitken mode number in this mam
      // package const int mam_acc = modeptr_accum;  // !accumulation mode
      // number in this mam package

  // dnidt_imode => dqdt
  // dncdt_imode => dqqcwdt

  const Real dgnmin = dgnmin_nmodes_imode;
  const Real dgnmax = dgnmax_nmodes_imode;
  const Real mean_std_dev = mean_std_dev_nmodes_imode;

  // initial value of num interstitial for this Real and mode
  //NOTE: number mixing ratios are always present for diagnostic
  // calls, so it is okay to use "state_q" instead of rad_cnst calls
  auto init_num_i = n_i_imode; // num_a0 = state_q(icol,klev,num_mode_idx)

  // `adjust_num_sizes` will use the initial value, but other
  // calculations require this to be nonzero.
  // Make it non-negative
  auto num_i_k = init_num_i < 0 ? zero : init_num_i;

  auto init_num_c = n_c_imode; // fldcw(icol,klev)
  // Make it non-negative
  auto num_c_k = init_num_c < 0 ? zero : init_num_c;

  const auto is_aitken_or_accumulation =
      imode == accumulation_idx || imode == aitken_idx;
  const auto do_adjust_aitken_or_accum =
      is_aitken_or_accumulation && do_aitacc_transfer;
  if (do_adjust) {
    /*------------------------------------------------------------------
     *  Do number adjustment for interstitial and activated particles
     *------------------------------------------------------------------
     * Adjustments that are applied over time-scale deltat
     * (model time step in seconds):
     *
     *   1. make numbers non-negative or
     *   2. make numbers zero when volume is zero
     *
     *
     * Adjustments that are applied over time-scale of a day (in
     *seconds)
     *   3. bring numbers to within specified bounds
     *
     * (Adjustment details are explained in the process)
     *------------------------------------------------------------------*/

    // number tendencies to be updated by adjust_num_sizes subroutine

    auto &interstitial_tend = dnidt_imode;
    auto &cloudborne_tend = dncdt_imode;

    /*NOTE: Only number tendencies (NOT mass mixing ratios) are
     updated in adjust_num_sizes Effect of these adjustment will be
     reflected in the particle diameters (via
     "update_diameter_and_vol2num" subroutine call below) */
    // Thus, when we are NOT doing the diameter/vol adjustments below,
    // then we DO the num adjustment here
    if (!do_adjust_aitken_or_accum) {
      calcsize::adjust_num_sizes(
          dryvol_i, dryvol_c, init_num_i, init_num_c, dt, // in
          num2vol_ratio_min_imode, num2vol_ratio_max_imode,
          adj_tscale_inv,                      // in
          num_i_k, num_c_k,                    // out
          interstitial_tend, cloudborne_tend); // out
    }
  }

  // update diameters and volume to num ratios for interstitial
  // aerosols


  calcsize::update_diameter_and_vol2num(
      dryvol_i, num_i_k, num2vol_ratio_min_imode, num2vol_ratio_max_imode,
      dgnmin, dgnmax, mean_std_dev, dgncur_i_imode, num2vol_ratio_cur_i);

  // update diameters and volume to num ratios for cloudborne aerosols
  calcsize::update_diameter_and_vol2num(
      dryvol_c, num_c_k, num2vol_ratio_min_imode, num2vol_ratio_max_imode,
      dgnmin, dgnmax, mean_std_dev, dgncur_c_imode, num2vol_ratio_cur_c);

  // save number concentrations and dry volumes for explicit
  // aitken <--> accum mode transfer, which is the next step in
  // the calcSize process
  if (do_aitacc_transfer) {
    if (imode == aitken_idx) {
      // TODO: determine if we need to save these--i.e., is drv_i ever
      // changed before the max() calculation in
      // aitken_accum_exchange() if yes, maybe better to skip the
      // logic and do it, regardless?
      dryvol_i_aitsv = dryvol_i;
      num_i_k_aitsv = num_i_k;
      dryvol_c_aitsv = dryvol_c;
      num_c_k_aitsv = num_c_k;
    } else if (imode == accumulation_idx) {
      dryvol_i_accsv = dryvol_i;
      num_i_k_accsv = num_i_k;
      dryvol_c_accsv = dryvol_c;
      num_c_k_accsv = num_c_k;
    }
  }
  // these were work variables that seem to not be used
  drv_i_sv = dryvol_i;
  num_i_sv = num_i_k;
  drv_c_sv = dryvol_c;
  num_c_sv = num_c_k;

} /// size_adjustment

KOKKOS_INLINE_FUNCTION
void compute_dry_volume(
    int imode,           // in
    const Real *state_q, // in
    const Real *qqcw,    // in
    const Real inv_density[AeroConfig::num_modes()]
                          [AeroConfig::num_aerosol_ids()], // in
    const int lmassptr_amode[maxd_aspectype][AeroConfig::num_modes()],
    Real &dryvol_i, // out
    Real &dryvol_c) // out
{
  constexpr Real zero = 0.0;
  dryvol_i = zero;
  dryvol_c = zero;

  const auto n_spec = num_species_mode(imode);
  for (int ispec = 0; ispec < n_spec; ispec++) {
    const int idx = lmassptr_amode[ispec][imode];
    dryvol_i += max(zero, state_q[idx]) * inv_density[imode][ispec];
    dryvol_c += max(zero, qqcw[idx]) * inv_density[imode][ispec];
  } // end ispec

} // end

KOKKOS_INLINE_FUNCTION
void modal_aero_calcsize_sub(
    const Real *state_q, // in
    const Real *qqcw,    // in
    const Real dt,
    const bool do_adjust,
    const bool do_aitacc_transfer,
    const bool update_mmr,
    const Real inv_density[AeroConfig::num_modes()]
                          [AeroConfig::num_aerosol_ids()], // in

    const Real num2vol_ratio_min[AeroConfig::num_modes()],
    const Real num2vol_ratio_max[AeroConfig::num_modes()],                      
    const Real num2vol_ratio_nom_nmodes[AeroConfig::num_modes()],
    const Real dgnmin_nmodes[AeroConfig::num_modes()],
    const Real dgnmax_nmodes[AeroConfig::num_modes()],
    const Real dgnnom_nmodes[AeroConfig::num_modes()],
    const Real mean_std_dev_nmodes[AeroConfig::num_modes()],
    const int lmassptr_amode[maxd_aspectype][AeroConfig::num_modes()],
    int numptr_amode[AeroConfig::num_modes()],
    Real dgncur_i[AeroConfig::num_modes()],
    Real dgncur_c[AeroConfig::num_modes()]
    // ncol, lchnk, state_q, pdel, deltat, qqcw, ptend, do_adjust_in, &
    // do_aitacc_transfer_in, list_idx_in, update_mmr_in, dgnumdry_m
) {
	// dgncur_a => dgnumdry_m(:,:,:)
#if 1
	const Real zero =0.0;
	const int aitken_idx = int(ModeIndex::Aitken);
    const int accumulation_idx = int(ModeIndex::Accumulation);
  constexpr int nmodes = AeroConfig::num_modes();
  const Real seconds_in_a_day = 86400.0; // BAD_CONSTANT!!
  /*-----------------------------------------------------------------------
     !
     ! Calculates aerosol size distribution parameters
     !    mprognum_amode >  0
     !       calculate Dgnum from mass, number, and fixed sigmag
     !    mprognum_amode <= 0
     !       calculate number from mass, fixed Dgnum, and fixed sigmag
     !
     ! Also (optionally) adjusts prognostic number to
     !    be within bounds determined by mass, Dgnum bounds, and sigma bounds
     !
     ! Author: R. Easter
     ! 09/2020: Refacotred by Balwinder Singh
     !
     !-----------------------------------------------------------------------*/
  // time scale for number adjustment
  /*----------------------------------------------------------------------------
  ! tadj = adjustment time scale for number, surface when they are prognosed
  !----------------------------------------------------------------------------*/

  // tadj => adj_tscale
  const Real adj_tscale = max(seconds_in_a_day, dt);

  // inverse of the adjustment time scale
  // tadjinv = 1.0/(tadj*close_to_one)
  const Real adj_tscale_inv = FloatingPoint<Real>::safe_denominator(adj_tscale);

  // fracadj = max( 0.0_r8, min( 1.0_r8, deltat*tadjinv ) )
  // Now compute dry diameter for both interstitial and cloud borne aerosols
  Real dryvol_i = zero;
  Real dryvol_c = zero;

  //
  Real dryvol_i_accsv = zero;
    Real dryvol_c_accsv = zero; 
	Real dryvol_i_aitsv = zero;
	Real dryvol_c_aitsv = zero;
	Real drv_i_sv[nmodes] ={};  
	Real drv_c_sv[nmodes] ={};
	Real num_i_k_accsv = zero;
	Real num_c_k_accsv = zero; 
	Real num_i_k_aitsv = zero; 
	Real num_c_k_aitsv = zero;
	Real num_i_sv[nmodes] ={};
	Real num_c_sv[nmodes] ={};
	Real dnidt[nmodes] ={};
	Real  dncdt[nmodes] ={};
  for (int imode = 0; imode < nmodes; imode++) {
    /*----------------------------------------------------------------------
   Initialize all parameters to the default values for the mode
   ----------------------------------------------------------------------*/
    // interstitial
    //  ----------------------------------------------------------------------
    //  Algorithm to compute dry aerosol diameter:
    //  calculate aerosol diameter volume, volume is computed from mass
    //  and density
    //  ----------------------------------------------------------------------

    // FIXME: get rid of these unused bits and related comments?
    // find start and end index of species in this mode in the
    // "population" array The indices are same for interstitial and
    // cloudborne species s_spec_ind = population_offsets(imode) start
    // index e_spec_ind = population_offsets(imode+1) - 1 end index of
    // species for all (modes expect the last mode)

    // if(imode.eq.nmodes) then  for last mode
    //    e_spec_ind = num_populations if imode==nmodes, end index is
    //    the total number of species
    // endif

    // nspec = num_mode_species(imode) total number of species in mode
    // "imode"

    // capture densities for each species in this mode
    // density(1:max_nspec) = huge(density) initialize the whole array
    // to a huge value [FIXME: NaN would be better than huge]
    // density(1:nspec) = spec_density(imode, 1:nspec) assign density
    // till nspec (as nspec can be different for each mode)

    // Initialize diameter(dgnum), volume to number
    // ratios(num2vol_ratio_cur) and dry volume (dryvol) for both
    // interstitial and cloudborne aerosols we did not implement
    // set_initial_sz_and_volumes
    // interstitial
    dgncur_i[imode] = dgnnom_nmodes[imode]; // diameter [m]
    Real num2vol_ratio_cur_i =
        num2vol_ratio_nom_nmodes[imode]; // volume to number
    // cloud borne
    dgncur_c[imode] = dgnnom_nmodes[imode]; // diameter [m]
    Real num2vol_ratio_cur_c =
        num2vol_ratio_nom_nmodes[imode]; // volume to number

    // dry volume is set to zero inside compute_dry_volume_k
    //----------------------------------------------------------------------
    // Compute dry volume mixratios (aerosol diameter)
    // Current default: number mmr is prognosed
    //       Algorithm:calculate aerosol diameter from mass, number, and
    //       fixed sigmag
    //
    // sigmag ("sigma g") is "geometric standard deviation for aerosol
    // mode"
    //
    // Volume = sum_over_components{ component_mass mixratio / density }
    //----------------------------------------------------------------------
    // dryvol_i, dryvol_c are set to zero inside compute_dry_volume_k
    compute_dry_volume(imode,       // in
                       state_q,     // in
                       qqcw,        // in
                       inv_density, // in
                       lmassptr_amode,
                       dryvol_i, // out
                       dryvol_c);

    // do size adjustment based on computed dry diameter values and update the
    // diameters
    //find state q array number mode indices for interstitial and cloud borne aerosols
    // Both num_mode_idx and num_cldbrn_mode_idx should be exactly same and should be same
    // for both prognostic and diagnostic radiation lists

    const int num_mode_idx = numptr_amode[imode];
    const int num_cldbrn_mode_idx = numptr_amode[imode];//numptrcw_amode[imode];
    const Real n_i_imode = state_q[num_mode_idx]; // from state_q
    const Real n_c_imode = qqcw[num_cldbrn_mode_idx];// from qqcw
    // const bool update_mmr
   
    size_adjustment(imode, dryvol_i, n_i_imode, dryvol_c, 
	 // pdel
	do_adjust, update_mmr, do_aitacc_transfer,  
	adj_tscale_inv,
	dt,  n_c_imode, 
	// additional parameters
	num2vol_ratio_min[imode],
    num2vol_ratio_max[imode], 
    dgnmin_nmodes[imode],
    dgnmax_nmodes[imode],
    mean_std_dev_nmodes[imode], 
    accumulation_idx, 
    aitken_idx,
    // outputs
	dgncur_i[imode], dgncur_c[imode], num2vol_ratio_cur_i,
	num2vol_ratio_cur_c,
	dryvol_i_accsv ,
	dryvol_c_accsv, 
	dryvol_i_aitsv,
	dryvol_c_aitsv,
	drv_i_sv[imode],
	drv_c_sv[imode],
	num_i_k_accsv,
	num_c_k_accsv, 
	num_i_k_aitsv, 
	num_c_k_aitsv,
	num_i_sv[imode],
	num_c_sv[imode], 
	dnidt[imode],
	dncdt[imode]
	);

  } // imode
#endif
} // modal_aero_calcsize_sub

} // namespace modal_aero_calcsize

} // namespace mam4

#endif
