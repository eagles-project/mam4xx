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
#include <mam4xx/calcsize.hpp>
#include <mam4xx/conversions.hpp>
#include <mam4xx/mam4_types.hpp>
#include <mam4xx/ndrop.hpp>
#include <mam4xx/utils.hpp>

namespace mam4 {
namespace modal_aero_calcsize {
using haero::max;
using haero::min;
using haero::sqrt;

constexpr int maxd_aspectype = ndrop::maxd_aspectype;

KOKKOS_INLINE_FUNCTION
void init_calcsize(
    Real inv_density[AeroConfig::num_modes()][AeroConfig::num_aerosol_ids()],
    Real num2vol_ratio_min[AeroConfig::num_modes()],
    Real num2vol_ratio_max[AeroConfig::num_modes()],
    Real num2vol_ratio_max_nmodes[AeroConfig::num_modes()],
    Real num2vol_ratio_min_nmodes[AeroConfig::num_modes()],
    Real num2vol_ratio_nom_nmodes[AeroConfig::num_modes()],
    Real dgnmin_nmodes[AeroConfig::num_modes()],
    Real dgnmax_nmodes[AeroConfig::num_modes()],
    Real dgnnom_nmodes[AeroConfig::num_modes()],
    Real mean_std_dev_nmodes[AeroConfig::num_modes()],
    // outputs
    bool noxf_acc2ait[AeroConfig::num_aerosol_ids()],
    int &n_common_species_ait_accum,
    int ait_spec_in_acc[AeroConfig::num_aerosol_ids()],
    int acc_spec_in_ait[AeroConfig::num_aerosol_ids()]) {

  const Real one = 1.0;

  // find aerosol species in accumulation that can be transfer to aitken mode
  const int accum_idx = int(ModeIndex::Accumulation);
  const int aitken_idx = int(ModeIndex::Aitken);

  // check if accumulation species exists in aitken mode
  // also save idx for transfer
  int count = 0;
  for (int isp = 0; isp < num_species_mode(accum_idx); ++isp) {
    // assume species can not be transfer.
    noxf_acc2ait[isp] = true;
    AeroId sp_accum = mode_aero_species(accum_idx, isp);

    for (int jsp = 0; jsp < num_species_mode(aitken_idx); ++jsp) {
      AeroId sp_aitken = mode_aero_species(aitken_idx, jsp);
      if (sp_accum == sp_aitken) {
        // false : can be transfer.
        noxf_acc2ait[isp] = false;
        // save index for transfer from accumulation to aitken mode
        // adding offset because we are using this index for state_q
        // Note: we assuimg accum mode is the first mode
        acc_spec_in_ait[count] = isp + utils::aero_start_ind();
        // save index for transfer from aitken to accumulation mode
        // adding offset because we are using this index for state_q
        // offset: aero_start + num of spec accum + 1 (number concentration)
        // Note: we assumig Aitken mode is the second mode
        ait_spec_in_acc[count] =
            jsp + utils::aero_start_ind() + num_species_mode(accum_idx) + 1;
        count++;
        break;
      }
    } // end aitken foor
  }   // end accumulation for
  n_common_species_ait_accum = count;

  // Set mode parameters.
  for (int m = 0; m < AeroConfig::num_modes(); ++m) {
    // FIXME: There is a comment in modal_aero_newnuc.F90 that Dick Easter
    // FIXME: thinks that dgnum_aer isn't used in MAM4, but it is actually
    // FIXME: used in this nucleation parameterization. So we will have to
    // FIXME: figure this out.
    dgnnom_nmodes[m] = modes(m).nom_diameter;
    dgnmin_nmodes[m] = modes(m).min_diameter;
    dgnmax_nmodes[m] = modes(m).max_diameter;
    mean_std_dev_nmodes[m] = modes(m).mean_std_dev;
    num2vol_ratio_nom_nmodes[m] =
        one / conversions::mean_particle_volume_from_diameter(
                  dgnnom_nmodes[m], modes(m).mean_std_dev);
    num2vol_ratio_min_nmodes[m] =
        one / conversions::mean_particle_volume_from_diameter(
                  dgnmax_nmodes[m], modes(m).mean_std_dev);
    num2vol_ratio_max_nmodes[m] =
        one / conversions::mean_particle_volume_from_diameter(
                  dgnmin_nmodes[m], modes(m).mean_std_dev);

    // compute inv density; density is constant, so we can compute in init.
    const auto n_spec = num_species_mode(m);
    for (int ispec = 0; ispec < n_spec; ispec++) {
      const int aero_id = int(mode_aero_species(m, ispec));
      inv_density[m][ispec] = Real(1.0) / aero_species(aero_id).density;
    } // for(ispec)
    // FIXME: do we need to update num2vol_ratio_min_nmodes and
    // num2vol_ratio_max_nmodes as well?
    num2vol_ratio_min[m] = num2vol_ratio_min_nmodes[m];
    num2vol_ratio_max[m] = num2vol_ratio_max_nmodes[m];

  } // for(m)

} // init_calcsize

struct CalcsizeData {

  int nspec_amode[AeroConfig::num_modes()];
  int lspectype_amode[ndrop::maxd_aspectype][AeroConfig::num_modes()];
  Real specdens_amode[ndrop::maxd_aspectype];
  int lmassptr_amode[ndrop::maxd_aspectype][AeroConfig::num_modes()];
  Real spechygro[ndrop::maxd_aspectype];
  Real mean_std_dev_nmodes[AeroConfig::num_modes()];

  int numptr_amode[AeroConfig::num_modes()];
  int mam_idx[AeroConfig::num_modes()][ndrop::nspec_max];
  int mam_cnst_idx[AeroConfig::num_modes()][ndrop::nspec_max];

  // FIXME: inv_density: we have different order of species in mam4xx.
  Real inv_density[AeroConfig::num_modes()][AeroConfig::num_aerosol_ids()] = {};
  Real num2vol_ratio_min[AeroConfig::num_modes()] = {};
  Real num2vol_ratio_max[AeroConfig::num_modes()] = {};
  Real num2vol_ratio_max_nmodes[AeroConfig::num_modes()] = {};
  Real num2vol_ratio_min_nmodes[AeroConfig::num_modes()] = {};
  Real num2vol_ratio_nom_nmodes[AeroConfig::num_modes()] = {};
  Real dgnmin_nmodes[AeroConfig::num_modes()] = {};
  Real dgnmax_nmodes[AeroConfig::num_modes()] = {};
  Real dgnnom_nmodes[AeroConfig::num_modes()] = {};
  bool noxf_acc2ait[AeroConfig::num_aerosol_ids()] = {};
  int n_common_species_ait_accum = {};
  int ait_spec_in_acc[AeroConfig::num_aerosol_ids()] = {};
  int acc_spec_in_ait[AeroConfig::num_aerosol_ids()] = {};

  const bool do_adjust = true;
  const bool do_aitacc_transfer = true;
  bool update_mmr = false;

  void initialize() {

    ndrop::get_e3sm_parameters(nspec_amode, lspectype_amode, lmassptr_amode,
                               numptr_amode, specdens_amode, spechygro, mam_idx,
                               mam_cnst_idx);

    init_calcsize(inv_density, num2vol_ratio_min, num2vol_ratio_max,
                  num2vol_ratio_max_nmodes, num2vol_ratio_min_nmodes,
                  num2vol_ratio_nom_nmodes, dgnmin_nmodes, dgnmax_nmodes,
                  dgnnom_nmodes, mean_std_dev_nmodes,
                  // outputs
                  noxf_acc2ait, n_common_species_ait_accum, ait_spec_in_acc,
                  acc_spec_in_ait);
  }

  void set_update_mmr(const bool update_mmr_in) { update_mmr = update_mmr_in; }
};

// NOTE: this version uses state_q and qqcw variables using format from e3sm
template<typename VectorType>
KOKKOS_INLINE_FUNCTION
void compute_coef_acc_ait_transfer(
    int iacc, const Real num2vol_ratio_geomean, const Real adj_tscale_inv,
    const VectorType&state_q, const VectorType&qqcw, const Real drv_i_accsv,
    const Real drv_c_accsv, const Real num_i_accsv, const Real num_c_accsv,
    const bool noxf_acc2ait[AeroConfig::num_aerosol_ids()],
    const Real voltonum_ait,
    const Real inv_density[AeroConfig::num_modes()]
                          [AeroConfig::num_aerosol_ids()],
    const Real num2vol_ratio_max_nmodes[AeroConfig::num_modes()],
    // additional parameters
    const int lmassptr_amode[maxd_aspectype][AeroConfig::num_modes()],
    Real &drv_i_noxf, Real &drv_c_noxf, int &acc2_ait_index,
    Real &xfercoef_num_acc2ait, Real &xfercoef_vol_acc2ait,
    Real xfertend_num[2][2]) {

  const Real zero = 0.0, one = 1.0;

  Real drv_t_noxf = zero, num_t0 = zero;
  Real num_t_noxf = zero;
  Real xferfrac_num_acc2ait = zero, xferfrac_vol_acc2ait = zero;

  acc2_ait_index = 0;
  xfercoef_num_acc2ait = zero;
  xfercoef_vol_acc2ait = zero;

  Real drv_t = drv_i_accsv + drv_c_accsv;
  Real num_t = num_i_accsv + num_c_accsv;
  drv_i_noxf = zero;
  drv_c_noxf = zero;
  const auto n_spec = num_species_mode(iacc); // number of species in iacc mode

  if (drv_t > zero) {
    // if number is larger than the mean, it means we have small particles
    // (keeping volume constant drv_t), we need to move particles to aitken
    // mode
    if (num_t > drv_t * num2vol_ratio_geomean) {
      // As there may be more species in the accumulation mode which are not
      // present in the aitken mode, we need to compute the num and volume only
      // for the species which can be transferred

      // In mam4xx q_i and q_c have a different structure that mam4, i.e,
      // q_i[nmode][nspecies](k). So, we need to modify the following for loop
      for (int ispec = 0; ispec < n_spec; ++ispec) {
        if (noxf_acc2ait[ispec]) { // then species which can't be
                                   // transferred
          // need qmass*invdens = (kg/kg-air) * [1/(kg/m3)] = m3/kg-air
          ; // !get mmr
          // Fortran to C++ indexing
          const int idx = lmassptr_amode[ispec][iacc] - 1;
          drv_i_noxf += max(zero, state_q[idx]) * inv_density[iacc][ispec];
          drv_c_noxf += max(zero, qqcw[idx]) * inv_density[iacc][ispec];
        } // end if
      }   // end ispec
      drv_t_noxf =
          drv_i_noxf +
          drv_c_noxf; // total volume that can't be moved to the aitken mode
      // Note: voltonumblo is equivalent to num2vol_ratio_max_nmodes
      num_t_noxf = drv_t_noxf *
                   num2vol_ratio_max_nmodes[iacc]; // total number that can't be
                                                   // moved to the aitken mode
      num_t0 = num_t;
      num_t = max(zero, num_t - num_t_noxf);
      drv_t = max(zero, drv_t - drv_t_noxf);
    } // end if (num_t > drv_t*num2vol_ratio_geomean)
  }   // end if (drv_t)

  if (drv_t > zero) {
    // Find out if we need to transfer based on the new num_t
    if (num_t > drv_t * num2vol_ratio_geomean) {
      acc2_ait_index = 1;
      if (num_t > drv_t * voltonum_ait) { // if number of larger than the
                                          // aitken mean, move all particles
        xferfrac_num_acc2ait = one;
        xferfrac_vol_acc2ait = one;
      } else { // scale the transfer
        xferfrac_vol_acc2ait = ((num_t / drv_t) - num2vol_ratio_geomean) /
                               (voltonum_ait - num2vol_ratio_geomean);
        xferfrac_num_acc2ait =
            xferfrac_vol_acc2ait * (drv_t * voltonum_ait / num_t);
        // bound the transfer coefficients between 0 and 1
        if ((xferfrac_num_acc2ait <= zero) || (xferfrac_vol_acc2ait <= zero)) {
          xferfrac_num_acc2ait = zero;
          xferfrac_vol_acc2ait = zero;
        } else if ((xferfrac_num_acc2ait >= one) ||
                   (xferfrac_vol_acc2ait >= one)) {
          xferfrac_num_acc2ait = one;
          xferfrac_vol_acc2ait = one;
        }
      }
      xferfrac_num_acc2ait = xferfrac_num_acc2ait * num_t *
                             FloatingPoint<Real>::safe_denominator(num_t0);
      xfercoef_num_acc2ait = xferfrac_num_acc2ait * adj_tscale_inv;
      xfercoef_vol_acc2ait = xferfrac_vol_acc2ait * adj_tscale_inv;
      xfertend_num[1][0] = num_i_accsv * xfercoef_num_acc2ait;
      xfertend_num[1][1] = num_c_accsv * xfercoef_num_acc2ait;
    } // end if (num_t > drv_t*num2vol_ratio_geomean)
  }

} // end compute_coef_acc_ait_transfer
template<typename VectorType>
KOKKOS_INLINE_FUNCTION void size_adjustment(
    const int imode, const Real dryvol_i, const Real n_i_imode,
    const Real dryvol_c,
    // pdel
    const bool do_adjust, const bool update_mmr, const bool do_aitacc_transfer,
    const Real adj_tscale_inv, const Real dt, const Real n_c_imode,
    // additional parameters
    const Real num2vol_ratio_min_imode, const Real num2vol_ratio_max_imode,
    const Real dgnmin_nmodes_imode, const Real dgnmax_nmodes_imode,
    const Real mean_std_dev_nmodes_imode, const int accumulation_idx,
    const int aitken_idx,
    // outputs
    Real &dgncur_i_imode, Real &dgncur_c_imode, Real &num2vol_ratio_cur_i,
    Real &num2vol_ratio_cur_c, Real &dryvol_i_accsv, Real &dryvol_c_accsv,
    Real &dryvol_i_aitsv, Real &dryvol_c_aitsv, Real &drv_i_sv, Real &drv_c_sv,
    Real &num_i_k_accsv, Real &num_c_k_accsv, Real &num_i_k_aitsv,
    Real &num_c_k_aitsv, Real &num_i_sv, Real &num_c_sv,  VectorType &dnidt_imode,
    Real &dncdt_imode) {

  constexpr Real zero = 0;

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
  // NOTE: number mixing ratios are always present for diagnostic
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
          dnidt_imode, dncdt_imode); // out
    }
  }

  // update diameters and volume to num ratios for interstitial
  // aerosols

  // FIXME: Need to add this adjustment with szadj_block_fac to match values
  // from Fortran code. for n=nait, divide v2nmin by 1.0e6 to effectively turn
  // off the
  //         adjustment when number is too small (size is too big)
  // BAD CONSTANT
  constexpr Real szadj_block_fac = 1.0e6;
  auto v2nmin = num2vol_ratio_min_imode;
  auto v2nmax = num2vol_ratio_max_imode;
  if (imode == aitken_idx)
    v2nmin /= szadj_block_fac;

  // for n=nacc, multiply v2nmax by 1.0e6 to effectively turn off the
  //          adjustment when number is too big (size is too small)
  if (imode == accumulation_idx)
    v2nmax *= szadj_block_fac;

  calcsize::update_diameter_and_vol2num(dryvol_i, num_i_k, v2nmin, v2nmax,
                                        dgnmin, dgnmax, mean_std_dev,
                                        dgncur_i_imode, num2vol_ratio_cur_i);

  // update diameters and volume to num ratios for cloudborne aerosols
  calcsize::update_diameter_and_vol2num(dryvol_c, num_c_k, v2nmin, v2nmax,
                                        dgnmin, dgnmax, mean_std_dev,
                                        dgncur_c_imode, num2vol_ratio_cur_c);

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
template<typename VectorType>
KOKKOS_INLINE_FUNCTION
void compute_dry_volume(
    int imode,           // in
    const VectorType& state_q, // in
    const VectorType& qqcw,    // in
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
    // Fortran to C++ indexing
    const int idx = lmassptr_amode[ispec][imode] - 1;
    dryvol_i += max(zero, state_q[idx]) * inv_density[imode][ispec];
    dryvol_c += max(zero, qqcw[idx]) * inv_density[imode][ispec];
  } // end ispec

} // end

//------------------------------------------------------------------------------------------------
template<typename VectorType>
KOKKOS_INLINE_FUNCTION
void update_tends_flx(const int jmode,         // in
                      const int src_mode_ixd,  // in
                      const int dest_mode_ixd, // in
                      const int n_common_species_ait_accum,
                      const int *src_species_idx, //
                      const int *dest_species_idx,
                      const Real xfertend_num[2][2], const Real xfercoef,
                      const VectorType& state_q, const VectorType& qqcw, VectorType& ptend,
                      Real *dqqcwdt) {

  // NOTES on arrays and indices:
  // jmode==0 is aitken->accumulation transfer;
  //     ==1 is accumulation->aitken transfer;

  // xfertend_num(jmode,0) contains how much to transfer for interstitial
  // aerosols xfertend_num(jmode,1) contains how much to transfer for cloudborne
  // aerosols

  const Real zero = 0;

  // interstiatial species
  Real &dqdt_src_i = ptend[src_mode_ixd];
  Real &dqdt_dest_i = ptend[dest_mode_ixd];
  const int aer_interstiatial = 0;
  calcsize::update_num_tends(jmode, aer_interstiatial, dqdt_src_i, dqdt_dest_i,
                             xfertend_num);

  // cloud borne apecies
  const int aer_cloud_borne = 1;
  Real &dqdt_src_c = dqqcwdt[src_mode_ixd];
  Real &dqdt_dest_c = dqqcwdt[dest_mode_ixd];

  calcsize::update_num_tends(jmode, aer_cloud_borne, dqdt_src_c, dqdt_dest_c,
                             xfertend_num);

  for (int i = 0; i < n_common_species_ait_accum; ++i) {
    const int ispec_src = src_species_idx[i];
    const int ispec_dest = dest_species_idx[i];
    // interstitial species
    const Real xfertend_i = haero::max(zero, state_q[ispec_src]) * xfercoef;
    ptend[ispec_src] -= xfertend_i;
    ptend[ispec_dest] += xfertend_i;

    // cloud borne species
    const Real xfertend_c = haero::max(zero, qqcw[ispec_src]) * xfercoef;
    dqqcwdt[ispec_src] -= xfertend_c;
    dqqcwdt[ispec_dest] += xfertend_c;
  }

} // end update_tends_flx
template<typename VectorType>
KOKKOS_INLINE_FUNCTION
void aitken_accum_exchange(
    const VectorType& state_q, const VectorType& qqcw, const int &aitken_idx,
    const int &accum_idx, const CalcsizeData &calcsizedata,
    const Real &adj_tscale_inv, const Real &dt, const Real &drv_i_aitsv,
    const Real &num_i_aitsv, const Real &drv_c_aitsv, const Real &num_c_aitsv,
    const Real &drv_i_accsv, const Real &num_i_accsv, const Real &drv_c_accsv,
    const Real &num_c_accsv, Real &dgncur_i_aitken, Real &dgncur_i_accum,
    Real &dgncur_c_aitken, Real &dgncur_c_accum,  VectorType& ptend, Real *dqqcwdt) {

  // FIXME: This version of does not include 	update_mmr=true, i.e tendencies
  // are not updated.

  // -----------------------------------------------------------------------------
  // Purpose: Exchange aerosols between aitken and accumulation modes based on
  // new  sizes
  //
  // Called by: modal_aero_calcsize_sub
  // Calls    : endrun
  //
  // Author: Richard Easter (Refactored by Balwinder Singh)
  // Ported to C++/Kokkos by: Oscar Diaz-Ibarra and Michael Schmidt
  // -----------------------------------------------------------------------------

  const Real zero = 0;
  Real num2vol_ratio_cur_c_accum = zero;
  Real num2vol_ratio_cur_i_accum = zero;

  Real num2vol_ratio_cur_c_aitken = zero;
  Real num2vol_ratio_cur_i_aitken = zero;

  const Real voltonum_ait =
      calcsizedata.num2vol_ratio_nom_nmodes[aitken_idx]; // volume to number for
                                                         // aitken mode
  const Real voltonum_acc =
      calcsizedata.num2vol_ratio_nom_nmodes[accum_idx]; // volume to number for
                                                        // accumulation mode
  int ait2acc_index = 0,
      acc2_ait_index = 0; // indices for transfer between modes
  Real xfertend_num[2][2] = {{0, 0}, {0, 0}}; // tendency for number transfer

  Real xfercoef_num_ait2acc = 0;
  Real xfercoef_vol_ait2acc =
      0; // volume and number transfer coefficients (ait->acc)
  Real xfercoef_num_acc2ait = 0;
  Real xfercoef_vol_acc2ait =
      0; // volume and number transfer coefficients (acc->ait)

  Real drv_i_noxf = 0, drv_c_noxf = 0;
  // "noxf" stands for "no transfer"

  // ------------------------------------------------------------------------
  //  Compute geometric mean of aitken and accumulation
  //  modes vol to num. This value is used to decide whether or not particles
  //  are transfered between these modes
  // ------------------------------------------------------------------------

  // num2vol_ratio_geomean is the geometric mean num2vol_ratio values
  // between the aitken and accum modes
  // const auto num2vol_ratio_geomean =
  // haero::sqrt(voltonum_ait*voltonum_acc);
  // voltonum_ait and voltonum_acc are O(10^22) and O(10^20), respectively,
  // and their multiplication overflows single precision, and
  // the square root ends up NaN. Thus,we compute sqrt individually
  const auto num2vol_ratio_geomean =
      haero::sqrt(voltonum_ait) * haero::sqrt(voltonum_acc);

  // Compute aitken -> accumulation transfer
  calcsize::compute_coef_ait_acc_transfer(
      accum_idx, num2vol_ratio_geomean, adj_tscale_inv, drv_i_aitsv,
      drv_c_aitsv, num_i_aitsv, num_c_aitsv, voltonum_acc, ait2acc_index,
      xfercoef_num_ait2acc, xfercoef_vol_ait2acc, xfertend_num);
  //  ----------------------------------------------------------------------------------------
  //   compute accum --> aitken transfer rates
  //
  //   accum may have some species (seasalt, dust, poa etc.) that are
  //   not in aitken mode.
  //   so first divide the accum dry volume & number into not-transferred (using
  //   noxf_acc2ait) species and transferred species, and use the
  //   transferred-species portion in what follows
  //  ----------------------------------------------------------------------------------------

  compute_coef_acc_ait_transfer(
      accum_idx, num2vol_ratio_geomean, adj_tscale_inv, state_q, qqcw,
      drv_i_accsv, drv_c_accsv, num_i_accsv, num_c_accsv,
      calcsizedata.noxf_acc2ait, voltonum_ait, calcsizedata.inv_density,
      calcsizedata.num2vol_ratio_max_nmodes, calcsizedata.lmassptr_amode,
      drv_i_noxf, drv_c_noxf, acc2_ait_index, xfercoef_num_acc2ait,
      xfercoef_vol_acc2ait, xfertend_num);

  // jump to end of loop if no transfer is needed
  if (ait2acc_index + acc2_ait_index > 0) {

    // compute new dgncur & num2vol_ratio_cur for aitken & accum modes

    // interstitial species
    const Real num_diff_i =
        (xfertend_num[0][0] - xfertend_num[1][0]) *
        dt; // diff in num from  ait -> accum and accum -> ait transfer
    const Real num_i = max(
        zero, num_i_aitsv - num_diff_i); // num removed/added from aitken mode
    const Real num_i_acc =
        max(zero,
            num_i_accsv + num_diff_i); // num added/removed to accumulation mode

    const Real vol_diff_i =
        (drv_i_aitsv * xfercoef_vol_ait2acc -
         (drv_i_accsv - drv_i_noxf) * xfercoef_vol_acc2ait) *
        dt; // diff in volume transfer fomr ait -> accum and accum -> ait
            // transfer
    const Real drv_i = max(
        zero, drv_i_aitsv - vol_diff_i); // drv removed/added from aitken mode

    const Real drv_i_acc =
        max(zero,
            drv_i_accsv + vol_diff_i); // drv added/removed to accumulation mode

    // cloud borne species
    const Real num_diff_c = (xfertend_num[0][1] - xfertend_num[1][1]) *
                            dt; // same as above for cloud borne aerosols

    const Real num_c = max(zero, num_c_aitsv - num_diff_c);
    const Real num_c_acc = max(zero, num_c_accsv + num_diff_c);
    const Real vol_diff_c =
        (drv_c_aitsv * xfercoef_vol_ait2acc -
         (drv_c_accsv - drv_c_noxf) * xfercoef_vol_acc2ait) *
        dt;
    const Real drv_c = max(zero, drv_c_aitsv - vol_diff_c);
    const Real drv_c_acc = max(zero, drv_c_accsv + vol_diff_c);

    // NOTE: CHECK original function does not have num2vol_ratio_max and dgnmax
    // as inputs. interstitial species (aitken mode)
    calcsize::compute_new_sz_after_transfer(
        drv_i, // in
        num_i, // in
        calcsizedata
            .num2vol_ratio_min_nmodes[aitken_idx], // corresponds to
                                                   // num2vol_ratio_hi because
                                                   // it is computed with
                                                   // dgnumhi
        calcsizedata
            .num2vol_ratio_max_nmodes[aitken_idx], // corresponds to
                                                   // num2vol_ratio_lo because
                                                   // it is computed with
                                                   // dgnumlo
        calcsizedata.num2vol_ratio_nom_nmodes[aitken_idx],
        calcsizedata.dgnmax_nmodes[aitken_idx],
        calcsizedata.dgnmin_nmodes[aitken_idx],
        calcsizedata.dgnnom_nmodes[aitken_idx],
        calcsizedata.mean_std_dev_nmodes[aitken_idx], dgncur_i_aitken,
        num2vol_ratio_cur_i_aitken);

    // cloud borne species (aitken mode)
    calcsize::compute_new_sz_after_transfer(
        drv_c,                                             // in
        num_c,                                             // in
        calcsizedata.num2vol_ratio_min_nmodes[aitken_idx], // corresponds to
                                                           // num2vol_ratio_hi
        calcsizedata.num2vol_ratio_max_nmodes[aitken_idx], // corresponds to
                                                           // num2vol_ratio_lo
        calcsizedata.num2vol_ratio_nom_nmodes[aitken_idx],
        calcsizedata.dgnmax_nmodes[aitken_idx],
        calcsizedata.dgnmin_nmodes[aitken_idx],
        calcsizedata.dgnnom_nmodes[aitken_idx],
        calcsizedata.mean_std_dev_nmodes[aitken_idx], dgncur_c_aitken,
        num2vol_ratio_cur_c_aitken);

    // interstitial species (accumulation mode)
    calcsize::compute_new_sz_after_transfer(
        drv_i_acc, // in
        num_i_acc, // in
        calcsizedata
            .num2vol_ratio_min_nmodes[accum_idx], // corresponds to
                                                  // num2vol_ratio_hi because it
                                                  // is computed with dgnumhi
        calcsizedata
            .num2vol_ratio_max_nmodes[accum_idx], // corresponds to
                                                  // num2vol_ratio_lo because it
                                                  // is computed with dgnumlo
        calcsizedata.num2vol_ratio_nom_nmodes[accum_idx],
        calcsizedata.dgnmax_nmodes[accum_idx],
        calcsizedata.dgnmin_nmodes[accum_idx],
        calcsizedata.dgnnom_nmodes[accum_idx],
        calcsizedata.mean_std_dev_nmodes[accum_idx], dgncur_i_accum,
        num2vol_ratio_cur_i_accum);

    // cloud borne species (accumulation mode)
    calcsize::compute_new_sz_after_transfer(
        drv_c_acc, // in
        num_c_acc, // in
        calcsizedata
            .num2vol_ratio_min_nmodes[accum_idx], // corresponds to
                                                  // num2vol_ratio_hi because it
                                                  // is computed with dgnumlo
        calcsizedata
            .num2vol_ratio_max_nmodes[accum_idx], // corresponds to
                                                  // num2vol_ratio_lo because it
                                                  // is computed with dgnumhi
        calcsizedata.num2vol_ratio_nom_nmodes[accum_idx],
        calcsizedata.dgnmax_nmodes[accum_idx],
        calcsizedata.dgnmin_nmodes[accum_idx],
        calcsizedata.dgnnom_nmodes[accum_idx],
        calcsizedata.mean_std_dev_nmodes[accum_idx], dgncur_c_accum,
        num2vol_ratio_cur_c_accum);

    //------------------------------------------------------------------
    // compute tendency amounts for aitken <--> accum transfer
    //------------------------------------------------------------------
    // jmode = 0 does aitken --> accum
    // index of accum and aitken mode for num concetration in state_q
    const int accum_idx_q =
        utils::aero_start_ind() + num_species_mode(accum_idx);
    const int aitken_idx_q = accum_idx_q + 1 + num_species_mode(aitken_idx);
    if (ait2acc_index > 0) {
      const int jmode = 0;
      // Since jmode = 0, source mode = aitken and destination mode accumulation
      update_tends_flx(
          jmode,        // in
          aitken_idx_q, // in src => aitken
          accum_idx_q,  // in dest => accumulation
          calcsizedata.n_common_species_ait_accum,
          calcsizedata.ait_spec_in_acc, // defined in aero_modes - src => aitken
          calcsizedata
              .acc_spec_in_ait, // defined in aero_modes - src => accumulation
          xfertend_num, xfercoef_vol_ait2acc, state_q, qqcw, ptend, dqqcwdt);
    } // end if (ait2acc_index)

    // jmode = 1 does accum --> aitken
    if (acc2_ait_index > 0) {
      const int jmode = 1;
      // Same suboutine as above (update_tends_flx) is called but source
      // and destination has been swapped so that transfer happens from
      // accumulation to aitken mode xfercoef_vol_acc2ait is used instead of
      // xfercoef_vol_ait2acc in this call as we are doing accum -> aitken
      // transfer
      update_tends_flx(
          jmode,        // in
          accum_idx_q,  // in src=> accumulation
          aitken_idx_q, // in dest => aitken
          calcsizedata.n_common_species_ait_accum,
          calcsizedata
              .acc_spec_in_ait, // defined in aero_modes - src => accumulation
          calcsizedata.ait_spec_in_acc, // defined in aero_modes - src => aitken
          xfertend_num, xfercoef_vol_acc2ait, state_q, qqcw, ptend, dqqcwdt);
    } // end if (acc2_ait_index)
  }   // end if (ait2acc_index+acc2_ait_index > 0)

} // aitken_accum_exchange

template<typename VectorType>
KOKKOS_INLINE_FUNCTION
void modal_aero_calcsize_sub(const VectorType& state_q, // in
                             const VectorType& qqcw,    // in
                             const Real dt, const CalcsizeData &calcsizedata,
                             Real dgncur_i[AeroConfig::num_modes()],
                             Real dgncur_c[AeroConfig::num_modes()],
                             VectorType& ptend, Real *dqqcwdt) {

  const Real zero = 0.0;
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
  Real drv_i_sv[nmodes] = {};
  Real drv_c_sv[nmodes] = {};
  Real num_i_k_accsv = zero;
  Real num_c_k_accsv = zero;
  Real num_i_k_aitsv = zero;
  Real num_c_k_aitsv = zero;
  Real num_i_sv[nmodes] = {};
  Real num_c_sv[nmodes] = {};

  // get index of num concentration in state_q
  int num_idx_state_q[nmodes] = {};
  utils::get_num_idx_in_state_q(num_idx_state_q);

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
    dgncur_i[imode] = calcsizedata.dgnnom_nmodes[imode]; // diameter [m]
    Real num2vol_ratio_cur_i =
        calcsizedata.num2vol_ratio_nom_nmodes[imode]; // volume to number
    // cloud borne
    dgncur_c[imode] = calcsizedata.dgnnom_nmodes[imode]; // diameter [m]
    Real num2vol_ratio_cur_c =
        calcsizedata.num2vol_ratio_nom_nmodes[imode]; // volume to number

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
    compute_dry_volume(imode,                    // in
                       state_q,                  // in
                       qqcw,                     // in
                       calcsizedata.inv_density, // in
                       calcsizedata.lmassptr_amode,
                       dryvol_i, // out
                       dryvol_c);

    // do size adjustment based on computed dry diameter values and update the
    // diameters
    // find state q array number mode indices for interstitial and cloud borne
    // aerosols
    // Both num_mode_idx and num_cldbrn_mode_idx should be exactly same and
    // should be same for both prognostic and diagnostic radiation lists
    // Fortran to C++ indexing
    const int num_mode_idx = calcsizedata.numptr_amode[imode] - 1;
    // Fortran to C++ indexing
    const int num_cldbrn_mode_idx =
        calcsizedata.numptr_amode[imode] - 1;         // numptrcw_amode[imode];
    const Real n_i_imode = state_q[num_mode_idx];     // from state_q
    const Real n_c_imode = qqcw[num_cldbrn_mode_idx]; // from qqcw
    // const bool update_mmr

    size_adjustment(
        imode, dryvol_i, n_i_imode, dryvol_c,
        // pdel
        calcsizedata.do_adjust, calcsizedata.update_mmr,
        calcsizedata.do_aitacc_transfer, adj_tscale_inv, dt, n_c_imode,
        // additional parameters
        calcsizedata.num2vol_ratio_min[imode],
        calcsizedata.num2vol_ratio_max[imode],
        calcsizedata.dgnmin_nmodes[imode], calcsizedata.dgnmax_nmodes[imode],
        calcsizedata.mean_std_dev_nmodes[imode], accumulation_idx, aitken_idx,
        // outputs
        dgncur_i[imode], dgncur_c[imode], num2vol_ratio_cur_i,
        num2vol_ratio_cur_c, dryvol_i_accsv, dryvol_c_accsv, dryvol_i_aitsv,
        dryvol_c_aitsv, drv_i_sv[imode], drv_c_sv[imode], num_i_k_accsv,
        num_c_k_accsv, num_i_k_aitsv, num_c_k_aitsv, num_i_sv[imode],
        num_c_sv[imode], ptend[num_idx_state_q[imode]],
        dqqcwdt[num_idx_state_q[imode]]);
  } // imode

  /*------------------------------------------------------------------------------
  ! when the aitken mode mean size is too big, the largest
  !    aitken particles are transferred into the accum mode
  !    to reduce the aitken mode mean size
  ! when the accum mode mean size is too small, the smallest
  !    accum particles are transferred into the aitken mode
  !    to increase the accum mode mean size
  !------------------------------------------------------------------------------*/

  if (calcsizedata.do_aitacc_transfer) {
    aitken_accum_exchange(
        state_q, qqcw, aitken_idx, accumulation_idx, calcsizedata,
        adj_tscale_inv, dt, dryvol_i_aitsv, num_i_k_aitsv, dryvol_c_aitsv,
        num_c_k_aitsv, dryvol_i_accsv, num_i_k_accsv, dryvol_c_accsv,
        num_c_k_accsv, dgncur_i[aitken_idx], dgncur_i[accumulation_idx],
        dgncur_c[aitken_idx], dgncur_c[accumulation_idx], ptend, dqqcwdt);
  }
} // modal_aero_calcsize_sub

} // namespace modal_aero_calcsize

} // namespace mam4

#endif
