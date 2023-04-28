// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MAM4XX_CALCSIZE_SUB_HPP
#define MAM4XX_CALCSIZE_SUB_HPP

#include <haero/atmosphere.hpp>
#include <haero/math.hpp>

#include <mam4xx/aero_config.hpp>
#include <mam4xx/calcsize.hpp>
#include <mam4xx/conversions.hpp>
#include <mam4xx/mam4_types.hpp>
#include <mam4xx/utils.hpp>

namespace mam4 {

namespace calcsize_sub {
KOKKOS_INLINE_FUNCTION
void set_initial_sz_and_volumes(
    const int top_lev,                                               // input
    const int nlev,                                                  // input
    const int imode,                                                 // input
    const Real dry_geometric_mean_diameter[AeroConfig::num_modes()], // input
    Real dgncur[][AeroConfig::num_modes()], // output length nlev
    Real v2ncur[][AeroConfig::num_modes()], // output length nlev
    Real dryvol[])                          // output length nlev
{
  constexpr Real pi = Constants::pi;
  // -----------------------------------------------------------------------------
  // Purpose: Set initial defaults for the dry diameter, volume to num
  //  and dry volume
  //
  // Called by: modal_aero_calcsize_sub
  //
  // Author: Richard Easter (Refactored by Balwinder Singh)
  // -----------------------------------------------------------------------------
  //
  // geometric dry mean diameter of the number distribution for aerosol mode
  const Real dgnum = dry_geometric_mean_diameter[imode];

  // geometric standard deviation of the number distribution for aerosol mode
  const Real sigmag = modes(imode).mean_std_dev;
  using haero::pow, haero::exp, haero::log;
  const Real voltonumb =
      1.0 / ((pi / 6.0) * pow(dgnum, 3) * exp(4.5 * pow(log(sigmag), 2)));

  for (int klev = top_lev; klev < nlev; ++klev) {
    dgncur[klev][imode] = dgnum;     // diameter
    v2ncur[klev][imode] = voltonumb; // volume to number
    dryvol[klev] = 0.0;              // initialize dry vol
  }
}

KOKKOS_INLINE_FUNCTION
void compute_dgn_vol_limits(
    const int imode,               // input
    const int nait,                // input  mode number of aitken mode
    const int nacc,                // input  mode number of accumulation mode
    const bool do_aitacc_transfer, // input  allows aitken <--> accum mode
                                   // transfer to be turned on/off
    Real &v2nmin,                  // output voltonumblo of current mode
    Real &v2nmax,                  // output voltonumbhi of current mode
    Real &v2nminrl,                // output relaxed voltonumblo
    Real &v2nmaxrl,                // output relaxed voltonumbhi
    Real &dgnxx,                   // output dgnumlo of current mode
    Real &dgnyy)                   // output dgnumhi of current mode
{
  using haero::pow, haero::exp, haero::log;
  constexpr Real pi = Constants::pi;
  // -----------------------------------------------------------------------------
  // Purpose: Compute hi and lo limits for diameter and volume based on the
  //  relaxation factor
  //
  // Called by: size_adjustment
  // Calls    : None
  //
  // Author: Richard Easter (Refactored by Balwinder Singh)
  // -----------------------------------------------------------------------------

  // local
  // dgnumlo_relaxed = dgnumlo/3 and dgnumhi_relaxed = dgnumhi*3
  const Real relax_factor = 27.0; // relax_factor=3**3=27,
  const Real szadj_block_fac = 1.0e6;

  // FIXME: There is no mean geometric number diameter in a mode.
  // FIXME: Assume "nominal" diameter for now?
  // FIXME: There is a comment in modal_aero_newnuc.F90 that Dick Easter
  // FIXME: thinks that dgnum_aer isn't used in MAM4, but it is actually
  // FIXME: used in this nucleation parameterization. So we will have to
  // FIXME: figure this out.
  const Real dgnumlo = modes(imode).min_diameter;
  const Real dgnumhi = modes(imode).max_diameter;
  const Real sigmag = modes(imode).mean_std_dev;

  // v2nmin = voltonumbhi is proportional to dgnumhi**(-3),
  //        and produces the minimum allowed number for a given volume
  v2nmin =
      1.0 / ((pi / 6.0) * pow(dgnumhi, 3) * exp(4.5 * pow(log(sigmag), 2)));

  // v2nmax = voltonumblo is proportional to dgnumlo**(-3),
  //         and produces the maximum allowed number for a given volume
  v2nmax =
      1.0 / ((pi / 6.0) * pow(dgnumlo, 3) * exp(4.5 * pow(log(sigmag), 2)));

  // v2nminrl and v2nmaxrl are their "relaxed" equivalents.
  v2nminrl = v2nmin / relax_factor;
  v2nmaxrl = v2nmax * relax_factor;
  dgnxx = dgnumhi;
  dgnyy = dgnumlo;

  // if do_aitacc_transfer is turned on, we will do the ait<->acc tranfer
  // separately in aitken_accum_exchange subroutine, so we are turning the size
  // adjustment for these two modes here.
  if (do_aitacc_transfer) {
    // for n=nait, divide v2nmin by 1.0e6 to effectively turn off the
    //         adjustment when number is too small (size is too big)
    if (imode == nait)
      v2nmin = v2nmin / szadj_block_fac;

    // for n=nacc, multiply v2nmax by 1.0e6 to effectively turn off the
    //          adjustment when number is too big (size is too small)
    if (imode == nacc)
      v2nmax = v2nmax * szadj_block_fac;

    // Also change the v2nmaxrl/v2nminrl so that
    // the interstitial<-->activated adjustment is turned off
    v2nminrl = v2nmin / relax_factor;
    v2nmaxrl = v2nmax * relax_factor;
  }
}

KOKKOS_INLINE_FUNCTION
void update_dgn_voltonum(const bool update_mmr, // input
                         const int aer_type,    // input
                         const Real gravit,     // input
                         const Real cmn_factor, // input
                         const Real drv,        // input
                         const Real num,        // input
                         const Real v2nmin,     // input
                         const Real v2nmax,     // input
                         const Real dgnxx,      // input
                         const Real dgnyy,      // input
                         const Real pdel,       // input
                         Real &dgncur,          // inout
                         Real &v2ncur,          // inout
                         Real qsrflx[4][2],     // inout
                         const Real dqdt)       // input
{
  // -----------------------------------------------------------------------------
  // Purpose: updates diameter and volume to num based on limits
  //
  // Called by: size_adjustment
  // Calls    : None
  //
  // Author: Richard Easter (Refactored by Balwinder Singh)
  // -----------------------------------------------------------------------------
  const Real third = 1.0 / 3.0;
  if (drv > 0.0) {
    if (num <= drv * v2nmin) {
      dgncur = dgnxx;
      v2ncur = v2nmin;
    } else if (num >= drv * v2nmax) {
      dgncur = dgnyy;
      v2ncur = v2nmax;
    } else {
      dgncur = haero::pow(drv / (cmn_factor * num), third);
      v2ncur = num / drv;
    }
  }
  if (update_mmr) {
    const Real pdel_fac = pdel / gravit; // = rho*dz
    qsrflx[0][aer_type] += haero::max(0.0, dqdt * pdel_fac);
    qsrflx[1][aer_type] += haero::min(0.0, dqdt * pdel_fac);
  }
}

static constexpr int gas_pcnst = 30;

KOKKOS_INLINE_FUNCTION
void size_adjustment(const int imode, const int top_lev, const int nlev,
                     const Real dryvol_a, const Real state_q[gas_pcnst],
                     const Real dryvol_c, const Real pdel, const bool do_adjust,
                     const bool update_mmr, const bool do_aitacc_transfer,
                     const Real deltatinv, const Real fracadj,
                     const Real fldcw, // specie mmr (cloud borne)
                     Real &dgncur_a, Real &dgncur_c, Real &v2ncur_a,
                     Real &v2ncur_c, Real &drv_a_accsv, Real &drv_c_accsv,
                     Real &drv_a_aitsv, Real &drv_c_aitsv,
                     Real drv_a_sv[AeroConfig::num_modes()],
                     Real drv_c_sv[AeroConfig::num_modes()], Real &num_a_accsv,
                     Real &num_c_accsv, Real &num_a_aitsv, Real &num_c_aitsv,
                     Real num_a_sv[AeroConfig::num_modes()],
                     Real num_c_sv[AeroConfig::num_modes()],
                     Real dotend[gas_pcnst], Real dotendqqcw[gas_pcnst],
                     Real &dqdt, Real &dqqcwdt, Real qsrflx[4][2]) {
  using haero::pow, haero::exp, haero::log, haero::max;
  constexpr Real pi = Constants::pi;
  constexpr Real gravit = Constants::gravity;

  // -----------------------------------------------------------------------------
  // Purpose: Do the aerosol size adjustment if needed
  //
  // Called by: modal_aero_calcsize_sub
  // Calls    : compute_dgn_vol_limits, adjust_num_sizes, update_dgn_voltonum
  //
  // Author: Richard Easter (Refactored by Balwinder Singh)
  // -----------------------------------------------------------------------------

  const int numptr_amode[AeroConfig::num_modes()] = {12, 17, 25, 29};
  const int num_mode_idx = numptr_amode[imode];
  const int num_cldbrn_mode_idx = numptr_amode[imode];
  // find out accumulation and aitken modes in the radiation list
  const int nacc = static_cast<int>(ModeIndex::Accumulation);
  const int nait = static_cast<int>(ModeIndex::Aitken);

  Real v2nmin = 0, v2nmax = 0, v2nminrl = 0, v2nmaxrl = 0, dgnxx = 0, dgnyy = 0;
  // set hi/lo limits (min, max and their relaxed counterparts) for volumes and
  // diameters
  compute_dgn_vol_limits(imode, nait, nacc, do_aitacc_transfer, v2nmin, v2nmax,
                         v2nminrl, v2nmaxrl, dgnxx, dgnyy);

  // set tendency logicals to true if we need to update mmr
  if (update_mmr) {
    dotend[num_mode_idx] = true;
    dotendqqcw[num_cldbrn_mode_idx] = true;
  }

  // Compute a common factor for size computations
  // FIXME: There is no mean geometric number diameter in a mode.
  // FIXME: Assume "nominal" diameter for now?
  // FIXME: There is a comment in modal_aero_newnuc.F90 that Dick Easter
  // FIXME: thinks that dgnum_aer isn't used in MAM4, but it is actually
  // FIXME: used in this nucleation parameterization. So we will have to
  // FIXME: figure this out.
  const Real dgnumlo = modes(imode).min_diameter;
  const Real dgnumhi = modes(imode).max_diameter;
  const Real sigmag = modes(imode).mean_std_dev;
  const Real cmn_factor = exp(4.5 * pow(log(sigmag), 2)) * pi / 6.0;

  const Real drv_a = dryvol_a;

  // NOTE: number mixing ratios are always present for diagnostic
  // calls, so it is okay to use "state_q" instead of rad_cnst calls
  const Real num_a0 = state_q[num_mode_idx];
  Real num_a = max(0.0, num_a0);
  const Real drv_c = dryvol_c;
  const Real num_c0 = fldcw;
  Real num_c = max(0.0, num_c0);

  if (do_adjust) {
    // -----------------------------------------------------------------
    //  Do number adjustment for interstitial and activated particles
    // -----------------------------------------------------------------
    // Adjustments that:
    // (1) make numbers non-negative or
    // (2) make numbers zero when volume is zero
    // are applied over time-scale deltat
    // Adjustments that bring numbers to within specified bounds are
    // applied over time-scale tadj
    // -----------------------------------------------------------------
    calcsize::adjust_num_sizes(drv_a, drv_c, num_a0, num_c0, deltatinv, v2nmin,
                               v2nmax, fracadj / deltatinv, num_a, num_c, dqdt,
                               dqqcwdt);
  }
  //
  // now compute current dgn and v2n
  //
  const int inter_aero = 0;
  update_dgn_voltonum(update_mmr, inter_aero, gravit, cmn_factor, drv_a, num_a,
                      v2nmin, v2nmax, dgnxx, dgnyy, pdel, dgncur_a, v2ncur_a,
                      qsrflx, dqdt);

  // for cloud borne aerosols
  const int cld_brn_aero = 1;
  update_dgn_voltonum(update_mmr, cld_brn_aero, gravit, cmn_factor, drv_c,
                      num_c, v2nmin, v2nmax, dgnumhi, dgnumlo, pdel, dgncur_c,
                      v2ncur_c, qsrflx, dqqcwdt);

  // save number and dryvol for aitken <--> accum transfer
  if (do_aitacc_transfer) {
    if (imode == nait) {
      drv_a_aitsv = drv_a;
      num_a_aitsv = num_a;
      drv_c_aitsv = drv_c;
      num_c_aitsv = num_c;
    } else if (imode == nacc) {
      drv_a_accsv = drv_a;
      num_a_accsv = num_a;
      drv_c_accsv = drv_c;
      num_c_accsv = num_c;
    }
  }
  drv_a_sv[imode] = drv_a;
  num_a_sv[imode] = num_a;
  drv_c_sv[imode] = drv_c;
  num_c_sv[imode] = num_c;
}

} // namespace calcsize_sub

/// @class CalcSizeSub
/// This class implements MAM4's CalcSizeSub parameterization.
class CalcSizeSub {
public:
  // nucleation-specific configuration
  struct Config {

    // default constructor -- sets default values for parameters
    Config() {}

    Config(const Config &) = default;
    ~Config() = default;
    Config &operator=(const Config &) = default;
  };

private:
  Config config_;

public:
  // name -- unique name of the process implemented by this class
  const char *name() const { return "MAM4 calcsize_sub"; }

  // init -- initializes the implementation with MAM4's configuration and with
  // a process-specific configuration.
  void init(const AeroConfig &aero_config,
            const Config &calcsize_sub_config = Config()) {
    // Set nucleation-specific config parameters.
    config_ = calcsize_sub_config;
  } // end(init)

  KOKKOS_INLINE_FUNCTION
  void compute_tendencies(const AeroConfig &config, const ThreadTeam &team,
                          Real t, Real dt, const Atmosphere &atmosphere,
                          const Prognostics &prognostics,
                          const Diagnostics &diagnostics,
                          const Tendencies &tendencies) const {}
};

} // namespace mam4

#endif
