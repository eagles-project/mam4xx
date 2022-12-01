#ifndef MAM4XX_CALCSIZE_HPP
#define MAM4XX_CALCSIZE_HPP

#include <haero/atmosphere.hpp>
#include <haero/math.hpp>

#include <mam4xx/aero_config.hpp>
#include <mam4xx/conversions.hpp>

namespace mam4 {

using Atmosphere = haero::Atmosphere;
using Constants = haero::Constants;
using Real = haero::Real;
using ThreadTeam = haero::ThreadTeam;

using ColumnView = haero::ColumnView;

using haero::max;
using haero::min;
using haero::sqrt;

namespace calcsize {

/*-----------------------------------------------------------------------------
Compute initial dry volume based on bulk mass mixing ratio (mmr) and species
density  volume = mmr/density
 -----------------------------------------------------------------------------*/

KOKKOS_INLINE_FUNCTION
void compute_dry_volume_k(int k, int imode, const Real inv_density[4][7],
                          const Prognostics& prognostics, // in
                          Real& dryvol_i,                 // out
                          Real& dryvol_c)                 // out
{
  const auto q_i = prognostics.q_aero_i;
  const auto q_c = prognostics.q_aero_c;
  dryvol_i = 0;
  dryvol_c = 0;
  const auto n_spec = num_species_mode(imode);
  for (int ispec = 0; ispec < n_spec; ispec++) {
    dryvol_i += max(0.0, q_i[imode][ispec](k)) * inv_density[imode][ispec];
    dryvol_c += max(0.0, q_c[imode][ispec](k)) * inv_density[imode][ispec];
  } // end ispec

} // end

/*
 * \brief Get relaxed limits for volume_to_num (we use relaxed limits for
 * aerosol number "adjustment" calculations via "adjust_num_sizes" subroutine.
 * Note: The relaxed limits will be artificially inflated (or deflated) for the
 * aitken and accumulation modes if "do_aitacc_transfer" flag is true to
 * effectively shut-off aerosol number "adjustment" calculations for these
 * modes because we do the explicit transfer (via "aitken_accum_exchange"
 * subroutine) from one mode to another instead of adjustments for these
 * modes).
 *
 * \note v2nmin and v2nmax are only updated for aitken and accumulation modes.
 */
KOKKOS_INLINE_FUNCTION
void get_relaxed_v2n_limits(const bool do_aitacc_transfer,
                            const bool is_aitken_mode, const bool is_accum_mode,
                            Real& v2nmin,   // in
                            Real& v2nmax,   // in
                            Real& v2nminrl, // out
                            Real& v2nmaxrl) // out
{
  /*
   * Relaxation factor is currently assumed to be a factor of 3 in diameter
   * which makes it 3**3=27 for volume.  i.e. dgnumlo_relaxed = dgnumlo/3 and
   * dgnumhi_relaxed = dgnumhi*3; therefore we use 3**3=27 as a relaxation
   * factor for volume.
   *
   * \see get_relaxed_v2n_limits
   */

  const Real relax_factor = 27.0; // BAD_CONSTANT!!

  // factor to artificially inflate or deflate v2nmin and v2nmax
  const Real szadj_block_fac = 1.0e6; // BAD_CONSTANT!!

  // default relaxation:
  v2nminrl = v2nmin / relax_factor;
  v2nmaxrl = v2nmax * relax_factor;
  // if do_aitacc_transfer is turned on, we will do the ait<->acc transfer
  // separately in aitken_accum_exchange subroutine, so we are effectively
  // turning OFF the size adjustment for these two modes here by artificially
  // inflating (or deflating) v2min and v2nmax using "szadj_block_fac" and then
  // computing v2minrl and v2nmaxrl based on newly computed v2min and v2nmax.

  if (do_aitacc_transfer) {
    // for aitken mode, divide v2nmin by 1.0e6 to effectively turn off the
    //          adjustment when number is too small (size is too big)
    if (is_aitken_mode)
      v2nmin /= szadj_block_fac;
    // for accumulation, multiply v2nmax by 1.0e6 to effectively turn off the
    //          adjustment when number is too big (size is too small)
    if (is_accum_mode)
      v2nmax *= szadj_block_fac;

    // Also change the v2nmaxrl/v2nminrl so that
    // the interstitial<-->activated number adjustment is effectively turned
    // off
    v2nminrl = v2nmin / relax_factor;
    v2nmaxrl = v2nmax * relax_factor;
  }
}

/*----------------------------------------------------------------------------
 * Compute particle diameter and volume to number ratios using dry bulk volume
 * (drv)
 *--------------------------------------------------------------------------*/
KOKKOS_INLINE_FUNCTION
void update_diameter_and_vol2num(const Real& drv, const Real& num, Real v2nmin,
                                 Real v2nmax, Real dgnmin, Real dgnmax,
                                 Real cmn_factor, Real& dgncur, Real& v2ncur) {
  const auto drv_gt_0 = drv > 0.0;
  if (!drv_gt_0)
    return;

  const auto drv_mul_v2nmin = drv * v2nmin;
  const auto drv_mul_v2nmax = drv * v2nmax;

  if (num <= drv_mul_v2nmin) {
    dgncur = dgnmax; // FIXME: e3sm uses dgnxx => hi
    v2ncur = v2nmin; // set to minimum vol2num ratio for this mode
  } else if (num >= drv_mul_v2nmax) {
    dgncur = dgnmin; // FIXME: e3sm uses dgnyy => lo
    v2ncur = v2nmax; // set to maximum vol2num ratio for this mode
  } else {
    dgncur = pow((drv / (cmn_factor * num)),
                 (1.0 / 3.0)); // compute diameter based on dry volume (drv)
    v2ncur = num / drv;
  }
}

KOKKOS_INLINE_FUNCTION
// rename to match ported fortran version
static Real update_num_adj_tends(const Real& num, const Real& num0,
                                 const Real& dt_inverse) {
  return (num - num0) * dt_inverse;
}

KOKKOS_INLINE_FUNCTION
static Real min_max_bounded(const Real& drv, const Real& v2nmin,
                            const Real& v2nmax, const Real& num) {
  return max(drv * v2nmin, min(drv * v2nmax, num));
}

/*
 * \brief number adjustment routine. See the implementation for more detailed
 * comments.
 */
KOKKOS_INLINE_FUNCTION
void adjust_num_sizes(const Real& drv_i, const Real& drv_c,
                      const Real& init_num_i, const Real& init_num_c,
                      const Real& dt, const Real& v2nmin, const Real& v2nmax,
                      const Real& v2nminrl, const Real& v2nmaxrl,
                      const Real& adj_tscale_inv, const Real& close_to_one,
                      Real& num_i, Real& num_c, Real& dqdt, Real& dqqcwdt) {

  /*
   *
   * The logic behind the number adjustment is described in detail in the
   * "else" section of the following "if" condition.
   *
   * We accomplish number adjustments in 3 steps:
   *
   *   1. Ensure that number mixing ratios are either zero or positive to
   * begin with. If both of them are zero (or less), we make them zero and
   * update tendencies accordingly (logic in the first "if" block")
   *   2. In this step, we use "relaxed" bounds for bringing number mixing
   *      ratios in their bounds. This is accomplished in three sub-steps
   * [(a), (b) and (c)] described in "Step 2" below.
   *   3. In this step, we use the actual bounds for bringing number mixing
   *      ratios in their bounds. This is also accomplished in three sub-steps
   *      [(a), (b) and (c)] described in "Step 3" below.
   *
   * If the number mixing ratio in a mode is out of mode's min/max range, we
   * re-balance interstitial and cloud borne aerosols such that the number
   * mixing ratio comes within the range. Time step for such an operation is
   * assumed to be one day (in seconds). That is, it is assumed that number
   * mixing ratios will be within range in a day. "adj_tscale" represents that
   * time scale
   *
   */

  // fraction of adj_tscale covered in the current time step "dt"
  const auto frac_adj_in_dt = max(0.0, min(1.0, dt * adj_tscale_inv));

  // inverse of time step
  const auto dtinv = 1.0 / (dt * close_to_one);

  /*
   * The masks below represent four if-else conditions in the original fortran
   * code. The masks represent whether a given branch should be traversed for
   * a given element of the Real, and this Real is passed to the function
   * invocations.
   */
  //  FIXME: I think this comment no longer applies(?)

  const auto drva_le_zero = drv_i <= 0.0;
  const auto drvc_le_zero = drv_c <= 0.0;

  const auto drv_i_c_le_zero = drva_le_zero && drvc_le_zero;
  const Real zero = 0;

  // If both interstitial (drv_i) and cloud borne (drv_c) dry volumes are zero
  // (or less) adjust numbers(num_a and num_c respectively) for both of them to
  // be zero for this mode and level
  if (drv_i_c_le_zero) {
    num_i = zero;
    num_c = zero;
    dqdt = update_num_adj_tends(num_i, init_num_i, dtinv);
    dqqcwdt = update_num_adj_tends(num_c, init_num_c, dtinv);
  } else if (drvc_le_zero) {
    // if cloud borne dry volume (drv_c) is zero(or less), the interstitial
    // number/volume == total/combined apply step 1 and 3, but skip the relaxed
    // adjustment (step 2, see below)
    num_c = zero;
    const auto numbnd = min_max_bounded(drv_i, v2nmin, v2nmax, num_i);
    num_i = num_i + (numbnd - num_i) * frac_adj_in_dt;
  } else if (drva_le_zero) {
    // interstitial volume is zero, treat similar to above
    const auto numbnd = min_max_bounded(drv_c, v2nmin, v2nmax, num_c);
    num_c = num_c + (numbnd - num_c) * frac_adj_in_dt;
    num_i = zero;
  } else {
    // The number adjustment is done in 3 steps:
    // Step 1: assumes that num_a and num_c are non-negative (nothing to be done
    // here)
    const auto num_i_stp1 = num_i;
    const auto num_c_stp1 = num_c;

    // Step 2 [Apply relaxed bounds] has 3 parts (a), (b) and (c)
    //  Step 2: (a)Apply relaxed bounds to bound num_a and num_c within
    //  "relaxed" bounds.
    auto numbnd = min_max_bounded(drv_i, v2nminrl, v2nmaxrl,
                                  num_i_stp1); // bounded to relaxed min and max
    /* (b)Ideally, num_* should be in range. If they are not, we assume that
       they will reach their maximum (or minimum) for this mode within a day
       (time scale). We then compute how much num_* will change in a time step
       by multiplying the difference between num_* and its maximum(or minimum)
       with "frac_adj_in_dt".
    */

    const auto delta_num_i_stp2 = (numbnd - num_i_stp1) * frac_adj_in_dt;
    // change in num_i in one time step
    auto num_i_stp2 = num_i_stp1 + delta_num_i_stp2;

    // bounded to relaxed min and max
    numbnd = min_max_bounded(drv_c, v2nminrl, v2nmaxrl, num_c_stp1);
    const auto delta_num_c_stp2 = (numbnd - num_c_stp1) * frac_adj_in_dt;

    // change in num_i in one time step
    auto num_c_stp2 = num_c_stp1 + delta_num_c_stp2;

    /* (c)We now also need to balance num_* incase only one among the
      interstitial or cloud-borne is changing. If interstitial stayed the same
      (i.e. it is within range) but cloud-borne is predicted to reach its
      maximum (or minimum), we modify interstitial number (num_i), so as to
      accommodate change in the cloud-borne aerosols (and vice-versa). We try to
      balance these by moving the num_* in the opposite direction as much as
      possible to conserve num_i + num_c (such that num_i + num_c stays close to
      its original value)
    */
    const auto delta_num_i_stp2_eq0 = delta_num_i_stp2 == 0.0;
    const auto delta_num_c_stp2_eq0 = delta_num_c_stp2 == 0.0;

    if (delta_num_i_stp2_eq0 && !delta_num_c_stp2_eq0) {
      num_i_stp2 = min_max_bounded(drv_i, v2nminrl, v2nmaxrl,
                                   num_i_stp1 - delta_num_c_stp2);
    } else if (delta_num_c_stp2_eq0 && !delta_num_i_stp2_eq0) {
      num_c_stp2 = min_max_bounded(drv_c, v2nminrl, v2nmaxrl,
                                   num_c_stp1 - delta_num_i_stp2);
    } else {
      // nothing here
      // for i = 1, N; do ~panic;
    } // end if

    /* Step3[apply stricter bounds] has 3 parts (a), (b) and (c)
       Step 3:(a) compute combined total of num_i and num_c
    */
    const auto total_drv = drv_i + drv_c;
    const auto total_num = num_i_stp2 + num_c_stp2;

    /*
     * 3(b) We now compute amount of num_* to change if total_num
     *     is out of range. If total_num is within range, we don't do anything
     * (i.e. delta_numi3 and delta_num_c_stp3 remain zero)
     */
    auto delta_num_i_stp3 = zero;
    auto delta_num_c_stp3 = zero;

    /*
     * "total_drv*v2nmin" represents minimum number for this mode, and
     * "total_drv*v2nmxn" represents maximum number for this mode
     */
    const auto min_number_bound = total_drv * v2nmin;
    const auto max_number_bound = total_drv * v2nmax;

    const auto total_lt_lowerbound = total_num < min_number_bound;
    const auto total_gt_upperbound = total_num > max_number_bound;
    if (total_lt_lowerbound) {
      // change in total_num in one time step
      const auto delta_num_t3 = (min_number_bound - total_num) * frac_adj_in_dt;

      /*
       * Now we need to decide how to distribute "delta_num" (change in
       * number) for num_i and num_c.
       */
      const auto do_dist_delta_num =
          (num_i_stp2 < drv_i * v2nmin) && (num_c_stp2 < drv_c * v2nmin);
      if (do_dist_delta_num) {
        /* if both num_i and num_c are less than the lower bound distribute
         * "delta_num" using weighted ratios
         */
        delta_num_i_stp3 = delta_num_t3 * (num_i_stp2 / total_num);
        delta_num_c_stp3 = delta_num_t3 * (num_c_stp2 / total_num);
      } else if (num_c_stp2 < drv_c * v2nmin) {
        // if only num_c is less than lower bound, assign total change to num_c
        delta_num_c_stp3 = delta_num_t3;
      } else if (num_i_stp2 < drv_i * v2nmin) {
        // if only num_i is less than lower bound, assign total change to num_i
        delta_num_i_stp3 = delta_num_t3;
      } else {
        // nothing here
      } // end if (do_dist_delta_num)

    } else if (total_gt_upperbound) {
      // change in total_num in one time step
      const auto delta_num_t3 = (max_number_bound - total_num) * frac_adj_in_dt;

      // decide how to distribute "delta_num"(change in number) for num_i and
      // num_c
      const auto do_dist_delta_num =
          (num_i_stp2 > drv_i * v2nmax) && (num_c_stp2 > drv_c * v2nmax);
      if (do_dist_delta_num) {
        /*
         * if both num_i and num_c are more than the upper bound distribute
         * "delta_num" using weighted ratios
         */
        delta_num_i_stp3 = delta_num_t3 * (num_i_stp2 / total_num);
        delta_num_c_stp3 = delta_num_t3 * (num_c_stp2 / total_num);

      } else if (num_c_stp2 > drv_c * v2nmax) {
        // if only num_c is more than the upper bound, assign total change to
        // num_c
        delta_num_c_stp3 = delta_num_t3;

      } else if (num_i_stp2 > drv_i * v2nmax) {
        // if only num_i is more than the upper bound, assign total change to
        // num_i
        delta_num_i_stp3 = delta_num_t3;
      } else {
        // nothing here
      } // end if (do_dist_delta_num)

    } // end if (total_lt_lowerbound)

    num_i = num_i_stp2 + delta_num_i_stp3;
    num_c = num_c_stp2 + delta_num_c_stp3;

  } // end if (drv_i_c_le_zero)

  // Update tendencies
  dqdt = update_num_adj_tends(num_i, init_num_i, dtinv);
  dqqcwdt = update_num_adj_tends(num_c, init_num_c, dtinv);
}

KOKKOS_INLINE_FUNCTION
void compute_coef_ait_acc_transfer(const int iacc,             // in
                                   const Real v2n_geomean,     // in
                                   const Real adj_tscale_inv,  // in
                                   const Real drv_i_aitsv,     // in
                                   const Real drv_c_aitsv,     // in
                                   const Real num_i_aitsv,     // in
                                   const Real num_c_aitsv,     // in
                                   const Real voltonum_acc,    // in
                                   int& ait2acc_index,         // out
                                   Real& xfercoef_num_ait2acc, // out
                                   Real& xfercoef_vol_ait2acc, // out
                                   Real xfertend_num[2][2]     // out
) {

  // ------------------------------------------------------------
  //  Purpose: Computes coefficients for transfer from aitken to
  //     accumulation mode
  //
  //  Author: Richard Easter (Refactored by Balwinder Singh)
  // ------------------------------------------------------------

  const Real zero = 0, one = 1;

  // initialize
  ait2acc_index = 0;
  Real xferfrac_num_ait2acc = zero;
  Real xferfrac_vol_ait2acc = zero;

  // compute aitken --> accum transfer rates

  const Real drv_t = drv_i_aitsv + drv_c_aitsv;
  const Real num_t = num_i_aitsv + num_c_aitsv;
  if (drv_t > zero) {
    // if num is less than the mean value, we have large particles (keeping
    //  volume constant drv_t), which needs to be moved to accumulation mode
    if (num_t < drv_t * v2n_geomean) {
      ait2acc_index = 1;
      if (num_t < drv_t * voltonum_acc) { // then  move all particles if number
        // is smaller than the acc mean
        xferfrac_num_ait2acc = one;
        xferfrac_vol_ait2acc = one;
      } else { // otherwise scale the transfer
        xferfrac_vol_ait2acc =
            ((num_t / drv_t) - v2n_geomean) / (voltonum_acc - v2n_geomean);
        xferfrac_num_ait2acc =
            xferfrac_vol_ait2acc * (drv_t * voltonum_acc / num_t);
        // bound the transfer coefficients between 0 and 1
        if ((xferfrac_num_ait2acc <= zero) || (xferfrac_vol_ait2acc <= zero)) {
          xferfrac_num_ait2acc = zero;
          xferfrac_vol_ait2acc = zero;
        } else if ((xferfrac_num_ait2acc >= one) ||
                   (xferfrac_vol_ait2acc >= one)) {
          xferfrac_num_ait2acc = one;
          xferfrac_vol_ait2acc = one;
        }

      } // end if (num_t < drv_t*voltonum_acc)
      xfercoef_num_ait2acc = xferfrac_num_ait2acc * adj_tscale_inv;
      xfercoef_vol_ait2acc = xferfrac_vol_ait2acc * adj_tscale_inv;
      xfertend_num[0][0] = num_i_aitsv * xfercoef_num_ait2acc;
      xfertend_num[0][1] = num_c_aitsv * xfercoef_num_ait2acc;
    } // end if (num_t < drv_t*v2n_geomean)
  }   // end if (drv_t > zero)

} // end compute_coef_ait_acc_transfer

KOKKOS_INLINE_FUNCTION
void compute_coef_acc_ait_transfer(
    int iacc, int klev, const Real v2n_geomean, const Real adj_tscale_inv,
    const Prognostics& prognostics, const Real drv_i_accsv,
    const Real drv_c_accsv, const Real num_i_accsv, const Real num_c_accsv,
    const bool no_transfer_acc2ait[7], const Real voltonum_ait,
    const Real inv_density[4][7], const Real v2nmin_nmodes[4], Real& drv_i_noxf,
    Real& drv_c_noxf, int& acc2_ait_index, Real& xfercoef_num_acc2ait,
    Real& xfercoef_vol_acc2ait, Real xfertend_num[2][2]) {

  const auto q_i = prognostics.q_aero_i;
  const auto q_c = prognostics.q_aero_c;

  Real drv_t_noxf, num_t0;
  Real num_t_noxf;
  Real xferfrac_num_acc2ait, xferfrac_vol_acc2ait;
  const Real zero_div_fac =
      1.0e-37; // BAD_CONSTANT!! This is not a physical constant, but it could
               // impact numerical errors.
  const Real zero = 0.0, one = 1.0;

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
    if (num_t > drv_t * v2n_geomean) {
      // As there may be more species in the accumulation mode which are not
      // present in the aitken mode, we need to compute the num and volume only
      // for the species which can be transferred

      // In mam4xx q_i and q_c have a different structure that mam4, i.e,
      // q_i[nmode][nspecies](k). So, we need to modify the following for loop
      for (int ispec = 0; ispec < n_spec; ++ispec) {
        if (no_transfer_acc2ait[ispec]) { // then species which can't be
                                          // transferred
          // need qmass*invdens = (kg/kg-air) * [1/(kg/m3)] = m3/kg-air
          drv_i_noxf +=
              max(zero, q_i[iacc][ispec](klev)) * inv_density[iacc][ispec];
          drv_c_noxf +=
              max(zero, q_c[iacc][ispec](klev)) * inv_density[iacc][ispec];
        } // end if
      }   // end ispec
      drv_t_noxf =
          drv_i_noxf +
          drv_c_noxf; // total volume that can't be moved to the aitken mode
      num_t_noxf =
          drv_t_noxf * v2nmin_nmodes[iacc]; // total number that can't be
                                            // moved to the aitken mode
      num_t0 = num_t;
      num_t = max(zero, num_t - num_t_noxf);
      drv_t = max(zero, drv_t - drv_t_noxf);
    } // end if (num_t > drv_t*v2n_geomean)
  }   // end if (drv_t)

  if (drv_t > zero) {
    // Find out if we need to transfer based on the new num_t
    if (num_t > drv_t * v2n_geomean) {
      acc2_ait_index = 1;
      if (num_t > drv_t * voltonum_ait) { // if number of larger than the
                                          // aitken mean, move all particles
        xferfrac_num_acc2ait = one;
        xferfrac_vol_acc2ait = one;
      } else { // scale the transfer
        xferfrac_vol_acc2ait =
            ((num_t / drv_t) - v2n_geomean) / (voltonum_ait - v2n_geomean);
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
      xferfrac_num_acc2ait =
          xferfrac_num_acc2ait * num_t / max(zero_div_fac, num_t0);
      xfercoef_num_acc2ait = xferfrac_num_acc2ait * adj_tscale_inv;
      xfercoef_vol_acc2ait = xferfrac_vol_acc2ait * adj_tscale_inv;
      xfertend_num[1][0] = num_i_accsv * xfercoef_num_acc2ait;
      xfertend_num[1][1] = num_c_accsv * xfercoef_num_acc2ait;
    } // end if (num_t > drv_t*v2n_geomean)
  }

} // end compute_coef_acc_ait_transfer

KOKKOS_INLINE_FUNCTION
void compute_new_sz_after_transfer(
    const Real drv,                     // in
    const Real num,                     // in
    const Real voltonumbhi2,            // in v2nmax_nmodes(imode) FIXME: stale?
    const Real voltonumblo2,            // in v2nmin_nmodes(imode) FIXME: stale?
    const Real voltonumb,               // in v2nnom_nmodes(imode)
    const Real dgn_nmodes_hi,           // in dgnmax_nmodes(imode)
    const Real dgn_nmodes_lo,           // in dgnmin_nmodes(imode)
    const Real dgn_nmodes_nom,          // in dgnnom_nmodes(imode)
    const Real cmn_factor_nmodes_imode, // in cmn_factor_nmodes(imode)
    Real& dgncur, Real& v2ncur) {

  // FIXME: E3Sm uses the following expressions for voltonumbhi and voltonumblo.
  // Note that voltonumbhi uses dgn_nmodes_hi and voltonumblo uses dgn_nmodes_lo
  // In other part of e3sm voltonumbhi is computed with dgn_nmodes_lo and
  // voltonumblo with dgn_nmodes_hi.

  const Real voltonumbhi =
      1. / pow(cmn_factor_nmodes_imode * dgn_nmodes_hi, 3.0);
  const Real voltonumblo =
      1. / pow(cmn_factor_nmodes_imode * dgn_nmodes_lo, 3.0);

  const Real zero = 0;
  const Real third =
      1.0 / 3.0; // BAD_CONSTANT!! it is not a physical constant. change name?
  if (drv > zero) {
    if (num <= drv * voltonumbhi) {
      dgncur = dgn_nmodes_hi;
      v2ncur = voltonumbhi;
    } else if (num >= drv * voltonumblo) {
      dgncur = dgn_nmodes_lo;
      v2ncur = voltonumblo;
    } else {
      dgncur = pow(drv / (cmn_factor_nmodes_imode * num), third);
      v2ncur = num / drv;
    } // end if (num <= drv*voltonumbhi)
  } else {
    dgncur = dgn_nmodes_nom;
    v2ncur = voltonumb;
  } // end if (drv > zero)

} // end compute_new_sz_after_transfer

KOKKOS_INLINE_FUNCTION
void update_num_tends(const int jmode, const int aer_type, Real& dqdt_src,
                      Real& dqdt_dest, const Real xfertend_num[2][2]) {
  const Real xfertend = xfertend_num[jmode][aer_type];
  dqdt_src -= xfertend;
  dqdt_dest += xfertend;

} // end update_num_tends

//------------------------------------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION
void update_tends_flx(const int klev,          // in
                      const int jmode,         // in
                      const int src_mode_ixd,  // in
                      const int dest_mode_ixd, // in
                      const int n_common_species_ait_accum,
                      const int* src_species_idx, //
                      const int* dest_species_idx,
                      const Real xfertend_num[2][2], const Real xfercoef,
                      const Prognostics& prognostics,
                      const Tendencies& tendencies) {

  // NOTES on arrays and indices:
  // jmode==1 is aitken->accumulation transfer;
  //     ==2 is accumulation->aitken transfer;

  // xfertend_num(jmode,1) contains how much to transfer for interstitial
  // aerosols xfertend_num(jmode,2) contains how much to transfer for cloudborne
  // aerosols

  const auto q_i = prognostics.q_aero_i;
  const auto q_c = prognostics.q_aero_c;
  const auto didt = tendencies.q_aero_i;
  const auto dnidt = tendencies.n_mode_i;
  const auto dcdt = tendencies.q_aero_c;
  const auto dncdt = tendencies.n_mode_c;

  const Real zero = 0;

  // interstiatial species
  Real& dqdt_src_i = dnidt[src_mode_ixd](klev);
  Real& dqdt_dest_i = dnidt[dest_mode_ixd](klev);
  const int aer_interstiatial = 0;
  update_num_tends(jmode, aer_interstiatial, dqdt_src_i, dqdt_dest_i,
                   xfertend_num);

  // cloud borne apecies
  const int aer_cloud_borne = 1;
  Real& dqdt_src_c = dncdt[src_mode_ixd](klev);
  Real& dqdt_dest_c = dncdt[dest_mode_ixd](klev);

  update_num_tends(jmode, aer_cloud_borne, dqdt_src_c, dqdt_dest_c,
                   xfertend_num);

  for (int i = 0; i < n_common_species_ait_accum; ++i) {
    const int ispec_src = src_species_idx[i];
    const int ispec_dest = dest_species_idx[i];
    // interstitial species
    const Real xfertend_i =
        max(zero, q_i[src_mode_ixd][ispec_src](klev)) * xfercoef;
    didt[src_mode_ixd][ispec_src](klev) -= xfertend_i;
    didt[dest_mode_ixd][ispec_dest](klev) += xfertend_i;

    // cloud borne species
    const Real xfertend_c =
        max(zero, q_c[src_mode_ixd][ispec_src](klev)) * xfercoef;
    dcdt[src_mode_ixd][ispec_src](klev) -= xfertend_c;
    dcdt[dest_mode_ixd][ispec_dest](klev) += xfertend_c;
  }

} // end update_tends_flx

/*
 * \brief Exchange aerosols between aitken and accumulation modes based on new
    sizes.
 */
KOKKOS_INLINE_FUNCTION
void aitken_accum_exchange(
    const int& k, const int& aitken_idx, const int& accum_idx,
    const bool no_transfer_acc2ait[7], const int n_common_species_ait_accum,
    const int* ait_spec_in_acc, const int* acc_spec_in_ait,
    const Real v2nmax_nmodes[4], const Real v2nmin_nmodes[4],
    const Real v2nnom_nmodes[4], const Real dgnmax_nmodes[4],
    const Real dgnmin_nmodes[4], const Real dgnnom_nmodes[4],
    const Real cmn_factor_nmodes[4], const Real inv_density[4][7],
    const Real& adj_tscale_inv, const Real& dt, const Prognostics& prognostics,
    const Real& drv_i_aitsv, const Real& num_i_aitsv, const Real& drv_c_aitsv,
    const Real& num_c_aitsv, const Real& drv_i_accsv, const Real& num_i_accsv,
    const Real& drv_c_accsv, const Real& num_c_accsv,
    const Diagnostics& diagnostics, const Tendencies& tendencies) {

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

  Real& dgncur_i_aitken = diagnostics.dgncur_i[aitken_idx](k);
  Real& dgncur_i_accum = diagnostics.dgncur_i[accum_idx](k);

  Real& dgncur_c_aitken = diagnostics.dgncur_c[aitken_idx](k);
  Real& dgncur_c_accum = diagnostics.dgncur_c[accum_idx](k);

  Real& v2ncur_c_accum = diagnostics.v2ncur_c[accum_idx](k);
  Real& v2ncur_i_accum = diagnostics.v2ncur_i[accum_idx](k);

  Real& v2ncur_c_aitken = diagnostics.v2ncur_c[aitken_idx](k);
  Real& v2ncur_i_aitken = diagnostics.v2ncur_i[aitken_idx](k);

  const Real voltonum_ait =
      v2nnom_nmodes[aitken_idx]; // volume to number for aitken mode
  const Real voltonum_acc =
      v2nnom_nmodes[accum_idx]; // volume to number for accumulation mode
  int ait2acc_index = 0,
      acc2_ait_index = 0; // indices for transfer between modes
  Real xfertend_num[2][2] = {{0, 0}, {0, 0}}; // tendency for number transfer
  const Real zero = 0;

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

  // v2n_geomean is the geometric mean vol2num values
  // between the aitken and accum modes
  auto v2n_geomean = haero::sqrt(voltonum_ait * voltonum_acc);

  // Compute aitken -> accumulation transfer
  compute_coef_ait_acc_transfer(
      accum_idx, v2n_geomean, adj_tscale_inv, drv_i_aitsv, drv_c_aitsv,
      num_i_aitsv, num_c_aitsv, voltonum_acc, ait2acc_index,
      xfercoef_num_ait2acc, xfercoef_vol_ait2acc, xfertend_num);

  //  ----------------------------------------------------------------------------------------
  //   compute accum --> aitken transfer rates
  //
  //   accum may have some species (seasalt, dust, poa etc.) that are
  //   not in aitken mode.
  //   so first divide the accum dry volume & number into not-transferred (using
  //   no_transfer_acc2ait) species and transferred species, and use the
  //   transferred-species portion in what follows
  //  ----------------------------------------------------------------------------------------

  compute_coef_acc_ait_transfer(
      accum_idx, k, v2n_geomean, adj_tscale_inv, prognostics, drv_i_accsv,
      drv_c_accsv, num_i_accsv, num_c_accsv, no_transfer_acc2ait, voltonum_ait,
      inv_density, v2nmin_nmodes, drv_i_noxf, drv_c_noxf, acc2_ait_index,
      xfercoef_num_acc2ait, xfercoef_vol_acc2ait, xfertend_num);

  // jump to end of loop if no transfer is needed
  if (ait2acc_index + acc2_ait_index > 0) {

    // compute new dgncur & v2ncur for aitken & accum modes

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

    // NOTE: CHECK orginal function does not have v2nmax.. and dgnmax as inputs.
    // interstitial species (aitken mode)
    compute_new_sz_after_transfer(
        drv_i, // in
        num_i, // in
        v2nmax_nmodes[aitken_idx], v2nmin_nmodes[aitken_idx],
        v2nnom_nmodes[aitken_idx], dgnmax_nmodes[aitken_idx],
        dgnmin_nmodes[aitken_idx], dgnnom_nmodes[aitken_idx],
        cmn_factor_nmodes[aitken_idx], dgncur_i_aitken, v2ncur_i_aitken);

    // cloud borne species (aitken mode)
    compute_new_sz_after_transfer(
        drv_c, // in
        num_c, // in
        v2nmax_nmodes[aitken_idx], v2nmin_nmodes[aitken_idx],
        v2nnom_nmodes[aitken_idx], dgnmax_nmodes[aitken_idx],
        dgnmin_nmodes[aitken_idx], dgnnom_nmodes[aitken_idx],
        cmn_factor_nmodes[aitken_idx], dgncur_c_aitken, v2ncur_c_aitken);

    // interstitial species (accumulation mode)
    compute_new_sz_after_transfer(
        drv_i_acc, // in
        num_i_acc, // in
        v2nmax_nmodes[accum_idx], v2nmin_nmodes[accum_idx],
        v2nnom_nmodes[accum_idx], dgnmax_nmodes[accum_idx],
        dgnmin_nmodes[accum_idx], dgnnom_nmodes[accum_idx],
        cmn_factor_nmodes[accum_idx], dgncur_i_accum, v2ncur_i_accum);

    // cloud borne species (accumulation mode)
    compute_new_sz_after_transfer(
        drv_c_acc, // in
        num_c_acc, // in
        v2nmax_nmodes[accum_idx], v2nmin_nmodes[accum_idx],
        v2nnom_nmodes[accum_idx], dgnmax_nmodes[accum_idx],
        dgnmin_nmodes[accum_idx], dgnnom_nmodes[accum_idx],
        cmn_factor_nmodes[accum_idx], dgncur_c_accum, v2ncur_c_accum);

    //------------------------------------------------------------------
    // compute tendency amounts for aitken <--> accum transfer
    //------------------------------------------------------------------
    // jmode = 1 does aitken --> accum
    if (ait2acc_index > 0) {
      const int jmode = 0;
      // Since jmode = 0, source mode = aitken and destination mode accumulation
      // interstitial and cloudborne aero solver have same struct, so idx
      // in cases are equal NOTE: ??

      update_tends_flx(
          k,          // in
          jmode,      // in
          aitken_idx, // in src => aitken
          accum_idx,  // in dest => accumulation
          n_common_species_ait_accum,
          ait_spec_in_acc, // defined in aero_modes - src => aitken
          acc_spec_in_ait, // defined in aero_modes - src => accumulation
          xfertend_num, xfercoef_vol_ait2acc, prognostics, tendencies);
    } // end if (ait2acc_index)

    // jmode = 2 does accum --> aitken
    if (acc2_ait_index > 0) {
      const int jmode = 1;
      // Same suboutine as above (update_tends_flx) is called but source
      // and destination has been swapped so that transfer happens from
      // accumulation to aitken mode xfercoef_vol_acc2ait is used instead of
      // xfercoef_vol_ait2acc in this call as we are doing accum -> aitken
      // transfer
      update_tends_flx(
          k,          // in
          jmode,      // in
          accum_idx,  // in src=> accumulation
          aitken_idx, // in dest => aitken
          n_common_species_ait_accum,
          acc_spec_in_ait, // defined in aero_modes - src => accumulation
          ait_spec_in_acc, // defined in aero_modes - src => aitken
          xfertend_num, xfercoef_vol_acc2ait, prognostics, tendencies);
    } // end if (acc2_ait_index)
  }   // end if (ait2acc_index+acc2_ait_index > 0)

} // aitken_accum_exchange

} // namespace calcsize

/// @class CalcSize
/// This class implements MAM4's CalcSize parameterization.
class CalcSize {
public:
  // nucleation-specific configuration
  struct Config {

    bool do_aitacc_transfer;
    bool do_adjust;

    // default constructor -- sets default values for parameters
    Config() : do_aitacc_transfer(true), do_adjust(true) {}

    Config(const Config&) = default;
    ~Config() = default;
    Config& operator=(const Config&) = default;
  };

private:
  Config config_;

  Real v2nmin_nmodes[AeroConfig::num_modes()],
      v2nmax_nmodes[AeroConfig::num_modes()],
      v2nnom_nmodes[AeroConfig::num_modes()];
  // Mode parameters
  Real dgnmin_nmodes[AeroConfig::num_modes()], // min geometric number diameter
      dgnmax_nmodes[AeroConfig::num_modes()],  // max geometric number diameter
      dgnnom_nmodes[AeroConfig::num_modes()];  // mean geometric number diameter

  // There is a common factor calculated over and over in the core loop of this
  // process. This factor has been pulled out so the calculation only has to be
  // performed once.
  Real common_factor_nmodes[AeroConfig::num_modes()];

  Real _inv_density[AeroConfig::num_modes()][AeroConfig::num_aerosol_ids()];

  // is the species present in both accum and aitken modes?
  // boolean view indicating which species will be transferred from
  // accumulation -> aitken indexed to accumulation mode (because it carries
  // more species)
  const bool _no_transfer_acc2ait[7] = {true,  false, true, false,
                                        false, true,  true};
  ;
  // number of common species between accum and aitken modes
  const int _n_common_species_ait_accum = 4;
  // index of aitken species in accum mode.
  const int _ait_spec_in_acc[4] = {0, 1, 2, 3};
  // index of accum species in aitken mode.
  const int _acc_spec_in_ait[4] = {0, 2, 5, 6};

public:
  // name -- unique name of the process implemented by this class
  const char* name() const { return "MAM4 calcsize"; }

  // init -- initializes the implementation with MAM4's configuration and with
  // a process-specific configuration.
  void init(const AeroConfig& aero_config,
            const Config& calcsize_config = Config()) {
    // Set nucleation-specific config parameters.
    config_ = calcsize_config;

    // Set mode parameters.
    for (int m = 0; m < AeroConfig::num_modes(); ++m) {
      // FIXME: There is no mean geometric number diameter in a mode.
      // FIXME: Assume "nominal" diameter for now?
      // FIXME: There is a comment in modal_aero_newnuc.F90 that Dick Easter
      // FIXME: thinks that dgnum_aer isn't used in MAM4, but it is actually
      // FIXME: used in this nucleation parameterization. So we will have to
      // FIXME: figure this out.
      dgnnom_nmodes[m] = modes(m).nom_diameter;
      dgnmin_nmodes[m] = modes(m).min_diameter;
      dgnmax_nmodes[m] = modes(m).max_diameter;
      common_factor_nmodes[m] =
          exp(4.5 * log(modes(m).mean_std_dev) * log(modes(m).mean_std_dev)) *
          Constants::pi_sixth; // A common factor
      v2nnom_nmodes[m] =
          1.0 / (common_factor_nmodes[m] * pow(dgnnom_nmodes[m], 3.0));
      v2nmin_nmodes[m] =
          1.0 / (common_factor_nmodes[m] * pow(dgnmax_nmodes[m], 3.0));
      v2nmax_nmodes[m] =
          1.0 / (common_factor_nmodes[m] * pow(dgnmin_nmodes[m], 3.0));
      // min_vol2num
      // = 1.0_wp/(pi_sixth*(imode%max_diameter**3.0_wp)*exp(4.5_wp*(log(imode%mean_std_dev))**2.0_wp))

      // compute inv density; density is constant, so we can compute in init.
      const auto n_spec = num_species_mode(m);
      for (int ispec = 0; ispec < n_spec; ispec++) {
        const int aero_id = int(mode_aero_species(m, ispec));
        _inv_density[m][ispec] = Real(1.0) / aero_species(aero_id).density;
      } // for(ispec)

    } // for(m)

  } // end(init)

  KOKKOS_INLINE_FUNCTION
  void compute_tendencies(const AeroConfig& config, const ThreadTeam& team,
                          Real t, Real dt, const Atmosphere& atmosphere,
                          const Prognostics& prognostics,
                          const Diagnostics& diagnostics,
                          const Tendencies& tendencies) const {

    const bool do_aitacc_transfer = config_.do_aitacc_transfer;
    const bool do_adjust = config_.do_adjust;

    const int aitken_idx = int(ModeIndex::Aitken);
    const int accumulation_idx = int(ModeIndex::Accumulation);
    const int nmodes = AeroConfig::num_modes();
    const int nk = atmosphere.num_levels();

    // diameter for interstitial aerosols
    auto& dgncur_i = diagnostics.dgncur_i;
    // volume to number ratio for interstitial aerosols
    auto& v2ncur_i = diagnostics.v2ncur_i;
    // diameter for cloud-borne aerosols
    auto& dgncur_c = diagnostics.dgncur_c;
    // volume to number ratio for cloud-borne aerosols
    auto& v2ncur_c = diagnostics.v2ncur_c;

    const auto inv_density = _inv_density;
    const Real zero = 0;
    const Real close_to_one = 1.0 + 1.0e-15; // BAD_CONSTANT!!
    const Real seconds_in_a_day = 86400.0;   // BAD_CONSTANT!!
    //
    const auto acc_spec_in_ait = _acc_spec_in_ait;
    const auto ait_spec_in_acc = _ait_spec_in_acc;
    const auto n_common_species_ait_accum = _n_common_species_ait_accum;
    const auto no_transfer_acc2ait = _no_transfer_acc2ait;

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, nk), KOKKOS_CLASS_LAMBDA(int k) {
          const auto n_i = prognostics.n_mode_i;
          const auto n_c = prognostics.n_mode_c;

          // tendencies for interstitial number mixing ratios
          const auto dnidt = tendencies.n_mode_i;

          // tendencies for cloud-borne number mixing ratios
          const auto dncdt = tendencies.n_mode_c;

          // FIXME: should these non-reference variables be here vs. above as
          // references and outside of the parfor? (I suspect above is correct)
          // // diameter for interstitial aerosols
          // auto dgncur_i = diagnostics.dgncur_i;
          // // volume to number ratio for interstitial aerosols
          // auto v2ncur_i = diagnostics.v2ncur_i;
          // // diameter for cloud-borne aerosols
          // auto dgncur_c = diagnostics.dgncur_c;
          // // volume to number ratio for cloud-borne aerosols
          // auto v2ncur_c = diagnostics.v2ncur_c;

          Real dryvol_i = 0;
          Real dryvol_c = 0;

          //  initialize these variables that are used at the bottom of the
          //  imode loop and are needed outside the loop scope
          Real dryvol_i_aitsv = 0;
          Real num_i_k_aitsv = 0;
          Real dryvol_c_aitsv = 0;
          Real num_c_k_aitsv = 0;
          Real dryvol_i_accsv = 0;
          Real num_i_k_accsv = 0;
          Real dryvol_c_accsv = 0;
          Real num_c_k_accsv = 0;
          // NOTE: these were work arrays and may not be necessary
          // Real drv_i_sv[nmodes];
          // Real num_i_sv[nmodes];
          // Real drv_c_sv[nmodes];
          // Real num_c_sv[nmodes];

          // time scale for number adjustment
          const auto adj_tscale = max(seconds_in_a_day, dt);

          // inverse of the adjustment time scale
          const auto adj_tscale_inv = 1.0 / (adj_tscale * close_to_one);

          for (int imode = 0; imode < nmodes; imode++) {

            // ----------------------------------------------------------------------
            // Algorithm to compute dry aerosol diameter:
            // calculate aerosol diameter volume, volume is computed from mass
            // and density
            // ----------------------------------------------------------------------

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

            // Initialize diameter(dgnum), volume to number ratios(v2ncur) and
            // dry volume (dryvol) for both interstitial and cloudborne
            // aerosols we did not implement set_initial_sz_and_volumes
            dgncur_i[imode](k) = dgnnom_nmodes[imode]; // diameter [m]
            v2ncur_i[imode](k) = v2nnom_nmodes[imode]; // volume to number

            dgncur_c[imode](k) = dgnnom_nmodes[imode]; // diameter [m]
            v2ncur_c[imode](k) = v2nnom_nmodes[imode]; // volume to number

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
            calcsize::compute_dry_volume_k(k, imode, inv_density, prognostics,
                                           dryvol_i, dryvol_c);

            auto v2nmin = v2nmin_nmodes[imode];
            auto v2nmax = v2nmax_nmodes[imode];
            const auto dgnmin = dgnmin_nmodes[imode];
            const auto dgnmax = dgnmax_nmodes[imode];
            const auto common_factor = common_factor_nmodes[imode];

            Real v2nminrl, v2nmaxrl;

            // compute upper and lower limits for volume to num (v2n) ratios and
            // diameters (dgn)
            //      Get relaxed limits for volume_to_num
            // (we use relaxed limits for aerosol number "adjustment"
            // calculations via "adjust_num_sizes" subroutine. Note: The
            // relaxed limits will be artificially inflated (or deflated) for
            // the aitken and accumulation modes if "do_aitacc_transfer" flag is
            // true to effectively shut-off aerosol number "adjustment"
            // calculations for these modes because we do the explicit transfer
            // (via "aitken_accum_exchange" subroutine) from one mode to
            // another instead of adjustments for these modes)

            calcsize::get_relaxed_v2n_limits(
                do_aitacc_transfer, imode == aitken_idx,
                imode == accumulation_idx, v2nmin, v2nmax, v2nminrl,
                v2nmaxrl); // outputs (NOTE: v2nmin and v2nmax are only updated
                           // for aitken and accumulation modes)

            // initial value of num interstitial for this Real and mode
            auto init_num_i = n_i[imode](k);

            // `adjust_num_sizes` will use the initial value, but other
            // calculations require this to be nonzero.
            // Make it non-negative
            auto num_i_k = init_num_i < 0 ? zero : init_num_i;

            auto init_num_c = n_c[imode](k);
            // Make it non-negative
            auto num_c_k = init_num_c < 0 ? zero : init_num_c;

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

              auto& interstitial_tend = dnidt[imode](k);
              auto& cloudborne_tend = dncdt[imode](k);

              /*NOTE: Only number tendencies (NOT mass mixing ratios) are
               updated in adjust_num_sizes Effect of these adjustment will be
               reflected in the particle diameters (via
               "update_diameter_and_vol2num" subroutine call below) */
              calcsize::adjust_num_sizes(
                  dryvol_i, dryvol_c, init_num_i, init_num_c, dt,     // in
                  v2nmin, v2nmax, v2nminrl, v2nmaxrl, adj_tscale_inv, // in
                  close_to_one,                                       // in
                  num_i_k, num_c_k,                                   // out
                  interstitial_tend, cloudborne_tend);                // out
            }

            // update diameters and volume to num ratios for interstitial
            // aerosols
            auto& dgncur_i_k = dgncur_i[imode](k);
            auto& v2ncur_i_k = v2ncur_i[imode](k);

            calcsize::update_diameter_and_vol2num(
                dryvol_i, num_i_k, v2nmin, v2nmax, dgnmin, dgnmax,
                common_factor, dgncur_i_k, v2ncur_i_k);

            // update diameters and volume to num ratios for cloudborne aerosols
            auto& dgncur_c_k = dgncur_c[imode](k);
            auto& v2ncur_c_k = v2ncur_c[imode](k);
            calcsize::update_diameter_and_vol2num(
                dryvol_c, num_c_k, v2nmin, v2nmax, dgnmin, dgnmax,
                common_factor, dgncur_c_k, v2ncur_c_k);

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
            // drv_i_sv[imode] = dryvol_i;
            // num_i_sv[imode] = num_i_k;
            // drv_c_sv[imode] = dryvol_c;
            // num_c_sv[imode] = num_c_k;
          } // for(imode)

          // ------------------------------------------------------------------
          //  Overall logic for aitken<-->accumulation transfer:
          //  ------------------------------------------------
          //  when the aitken mode mean size is too big, the largest
          //     aitken particles are transferred into the accum mode
          //     to reduce the aitken mode mean size
          //  when the accum mode mean size is too small, the smallest
          //     accum particles are transferred into the aitken mode
          //     to increase the accum mode mean size
          // ------------------------------------------------------------------
          if (do_aitacc_transfer) {

            calcsize::aitken_accum_exchange(
                k, aitken_idx, accumulation_idx, no_transfer_acc2ait,
                n_common_species_ait_accum, ait_spec_in_acc, acc_spec_in_ait,
                v2nmax_nmodes, v2nmin_nmodes, v2nnom_nmodes, dgnmax_nmodes,
                dgnmin_nmodes, dgnnom_nmodes, common_factor_nmodes, inv_density,
                adj_tscale_inv, dt, prognostics, dryvol_i_aitsv, num_i_k_aitsv,
                dryvol_c_aitsv, num_c_k_aitsv, dryvol_i_accsv, num_i_k_accsv,
                dryvol_c_accsv, num_c_k_accsv, diagnostics, tendencies);

          } // end do_aitacc_transfer
        }); // kokkos::parfor(k)
  }

private:
};

} // namespace mam4

#endif
