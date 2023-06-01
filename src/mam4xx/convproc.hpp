// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MAM4XX_CONVPROC_HPP
#define MAM4XX_CONVPROC_HPP

#include <haero/atmosphere.hpp>
#include <haero/math.hpp>

#include <mam4xx/aero_config.hpp>
#include <mam4xx/mam4_types.hpp>
#include <mam4xx/utils.hpp>

namespace mam4 {
// Namespace for WetDep function until wetdep is implimented.
namespace WetDepTemp {
KOKKOS_INLINE_FUNCTION
Real faer_resusp_vs_fprec_evap_mpln(const Real fprec_evap, const int) {
  // --------------------------------------------------------------------------------
  // corresponding fraction of precipitation-borne aerosol flux that is
  // resuspended Options of assuming log-normal or marshall-palmer raindrop size
  // distribution note that these fractions are relative to the cloud-base
  // fluxes, and not to the layer immediately above fluxes
  // --------------------------------------------------------------------------------

  // faer_resusp_vs_fprec_evap_mpln ! [fraction]
  // in fprec_evap [fraction]
  // in jstrcnv - current only two options:
  //    1 for marshall-palmer distribution
  //    2 for log-normal distribution
  Real a01, a02, a03, a04, a05, a06, a07, a08, a09, x_lox_lin, y_lox_lin;
  // log-normal distribution
  a01 = 6.1944215103685640E-02;
  a02 = -2.0095166685965378E+00;
  a03 = 2.3882460251821236E+01;
  a04 = -1.2695611774753374E+02;
  a05 = 4.0086943562320101E+02;
  a06 = -7.4954272875943707E+02;
  a07 = 8.1701055892023624E+02;
  a08 = -4.7941894659538502E+02;
  a09 = 1.1710291076059025E+02;
  x_lox_lin = 1.0000000000000001E-01;
  y_lox_lin = 6.2227889828044350E-04;
  Real x_var, y_var;
  x_var = utils::min_max_bound(0.0, 1.0, fprec_evap);
  if (x_var < x_lox_lin)
    y_var = y_lox_lin * (x_var / x_lox_lin);
  else
    y_var =
        x_var *
        (a01 +
         x_var *
             (a02 +
              x_var *
                  (a03 +
                   x_var *
                       (a04 +
                        x_var * (a05 +
                                 x_var * (a06 +
                                          x_var * (a07 +
                                                   x_var * (a08 +
                                                            x_var * a09))))))));
  return y_var;
}
} // namespace WetDepTemp

/// @class ConvProc
/// This class implements MAM4's ConvProc parameterization.
class ConvProc {
public:
  // nucleation-specific configuration
  struct Config {

    // default constructor -- sets default values for parameters
    Config() {}

    Config(const Config &) = default;
    ~Config() = default;
    Config &operator=(const Config &) = default;
  };

  static constexpr int num_modes = AeroConfig::num_modes();
  static constexpr int num_aerosol_ids = AeroConfig::num_aerosol_ids();
  enum species_class {
    undefined = 0,
    cldphysics = 1,
    aerosol = 2,
    gas = 3,
    other = 4
  };
  // maxd_aspectype = maximum allowable number of chemical species
  // in each aerosol mode.
  static constexpr int maxd_aspectype = 14;
  // gas_pcnst number of "gas phase" species
  // This should come from the chemistry model being used and will
  // probably need to be dynamic.  This forces lmassptr_amode to
  // also be dynamic.
  static constexpr int gas_pcnst = 40;

  // ====================================================================================
  // The diagnostic arrays are twice the lengths of ConvProc::gas_pcnst because
  // cloudborne aerosols are appended after interstitial aerosols both of which
  // are of length gas_pcnst.
  static constexpr int pcnst_extd = 2 * gas_pcnst;

  // Where lmapcc_val_num are defined in lmapcc_all
  //
  // numptr_amode(m) = gchm r-array index for the number mixing ratio
  // (particles/mole-air) for aerosol mode m that is in clear air or
  // interstitial are (but not in cloud water).  If zero or negative,
  // then number is not being simulated.
  KOKKOS_INLINE_FUNCTION
  static constexpr int numptr_amode(const int i) {
    const int numptr_amode[num_modes] = {22, 27, 35, 39};
    return numptr_amode[i];
  }
  // Use the same index for Q and QQCW arrays
  KOKKOS_INLINE_FUNCTION
  static constexpr int numptrcw_amode(const int i) { return numptr_amode(i); }

  // Where lmapcc_val_aer are defined in lmapcc_all
  //
  // lmassptr_amode(l,m) = gchm r-array index for the mixing ratio
  // (moles-x/mole-air) for chemical species l in aerosol mode m
  // that is in clear air or interstitial air (but not in cloud water).
  // If negative then number is not being simulated.
  KOKKOS_INLINE_FUNCTION
  static constexpr int lmassptr_amode(const int i, const int j) {
    const int lmassptr_amode[maxd_aspectype][num_modes] = {
        {15, 23, 28, 36}, {16, 24, 29, 37}, {17, 25, 30, 38}, {18, 26, 31, -1},
        {19, -1, 32, -1}, {20, -1, 33, -1}, {21, -1, 34, -1}, {-1, -1, -1, -1},
        {-1, -1, -1, -1}, {-1, -1, -1, -1}, {-1, -1, -1, -1}, {-1, -1, -1, -1},
        {-1, -1, -1, -1}, {-1, -1, -1, -1}};
    return lmassptr_amode[i][j];
  }

  // use the same index for Q and QQCW arrays
  //
  // lmassptrcw_amode(l,m) = gchm r-array index for the mixing ratio
  // (moles-x/mole-air) for chemical species l in aerosol mode m
  // that is currently bound/dissolved in cloud water
  KOKKOS_INLINE_FUNCTION
  static constexpr int lmassptrcw_amode(const int i, const int j) {
    return lmassptr_amode(i, j);
  }

private:
  Config config_;

public:
  // name -- unique name of the process implemented by this class
  const char *name() const { return "MAM4 convproc"; }

  // init -- initializes the implementation with MAM4's configuration and with
  // a process-specific configuration.
  void init(const AeroConfig &aero_config,
            const Config &convproc_config = Config()) {
    // Set nucleation-specific config parameters.
    config_ = convproc_config;
  } // end(init)

  KOKKOS_INLINE_FUNCTION
  void compute_tendencies(const AeroConfig &config, const ThreadTeam &team,
                          Real t, Real dt, const Atmosphere &atmosphere,
                          const Prognostics &prognostics,
                          const Diagnostics &diagnostics,
                          const Tendencies &tendencies) const {}
};

namespace convproc {

KOKKOS_INLINE_FUNCTION
void assign_la_lc(const int imode, const int ispec, int &la, int &lc) {
  // ---------------------------------------------------------------------
  // get the index of interstital (la) and cloudborne (lc) aerosols
  // from mode index and species index
  // Cloudborne aerosols are appended after interstitial aerosol array
  // so lc (cloudborne) is offset from ic (interstitial) by gas_pcnst.
  //-----------------------------------------------------------------------
  if (ispec == -1) {
    la = ConvProc::numptr_amode(imode);
    lc = ConvProc::numptrcw_amode(imode);
  } else {
    la = ConvProc::lmassptr_amode(ispec, imode);
    lc = ConvProc::lmassptrcw_amode(ispec, imode);
  }
  lc += ConvProc::gas_pcnst;
}

// nsrflx is the number of process-specific column tracer tendencies:
// activation, resuspension, aqueous chemistry, wet removal, actual and pseudo.
static constexpr int nsrflx = 6;
// clang-format off
KOKKOS_INLINE_FUNCTION
void update_tendency_diagnostics(
    const int ntsub,   // IN  number of sub timesteps
    const int ncnst,   // IN  number of tracers to transport
    const bool doconvproc[], // IN  flag for doing convective transport
    Real sumactiva[ConvProc::pcnst_extd], // INOUT sum (over layers) of dp*dconudt_activa [kg/kg/s * mb]
    Real sumaqchem[ConvProc::pcnst_extd], // INOUT sum (over layers) of dp*dconudt_aqchem [kg/kg/s * mb]
    Real sumwetdep[ConvProc::pcnst_extd], // INOUT sum (over layers) of dp*dconudt_wetdep [kg/kg/s * mb]
    Real sumresusp[ConvProc::pcnst_extd], // INOUT sum (over layers) of dp*dcondt_resusp [kg/kg/s * mb]
    Real sumprevap[ConvProc::pcnst_extd], // INOUT sum (over layers) of dp*dcondt_prevap [kg/kg/s * mb]
    Real sumprevap_hist[ConvProc::pcnst_extd],// INOUT sum (over layers) of dp*dcondt_prevap_hist [kg/kg/s * mb]
    Real qsrflx[][nsrflx]) // INOUT process-specific column tracer tendencies [kg/m2/s]
{

  // -----------------------------------------------------------------------
  //  update tendencies to final output of ma_convproc_tend
  // 
  //  note that the ma_convproc_tend does not apply convective cloud processing
  //     to the stratiform-cloudborne aerosol
  //  within this routine, cloudborne aerosols are convective-cloudborne
  // 
  //  the individual process column tendencies (sumwetdep, sumprevap, ...)
  //     are just diagnostic fields that can be written to history
  //  tendencies for interstitial and convective-cloudborne aerosol could
  //     both be passed back and output, if desired
  //  currently, however, the interstitial and convective-cloudborne tendencies
  //     are combined (in the next code block) before being passed back (in qsrflx)
  // -----------------------------------------------------------------------

  // clang-format on

  if (ConvProc::gas_pcnst != ncnst)
    Kokkos::abort("Invalid number of tracers to transport.");
  const int ntot_amode = AeroConfig::num_modes();
  int la = 0, lc = 0;
  // update diagnostic variables
  for (int imode = 0; imode < ntot_amode; ++imode) {
    for (int ispec = -1; ispec < num_species_mode(imode); ++ispec) {
      // cloudborne aerosols are appended after intersitial
      assign_la_lc(imode, ispec, la, lc);
      if (doconvproc[la]) {
        sumactiva[la] += sumactiva[lc];
        sumresusp[la] += sumresusp[lc];
        sumaqchem[la] += sumaqchem[lc];
        sumwetdep[la] += sumwetdep[lc];
        sumprevap[la] += sumprevap[lc];
        sumprevap_hist[la] += sumprevap_hist[lc];
      }
    }
  }
  // scatter overall tendency back to full array
  // The indexing started at 2 for Fortran, so 1 for C++
  const Real hund_ovr_g = 100.0 / Constants::gravity;
  const Real xinv_ntsub = 1.0 / ntsub;
  Real qsrflx_i[6] = {};
  for (int icnst = 1; icnst < ncnst; ++icnst) {
    if (doconvproc[icnst]) {
      // scatter column burden tendencies for various processes to qsrflx
      // process-specific column tracer tendencies [kg/m2/s]
      //   0 = activation   of interstial to conv-cloudborne
      //   1 = resuspension of conv-cloudborne to interstital
      //   2 = aqueous chemistry (not implemented yet, so zero)
      //   3 = wet removal
      //   4 = actual precip-evap resuspension (what actually is applied to a
      //   species)
      //   5 = pseudo precip-evap resuspension (for history file)
      qsrflx_i[0] = sumactiva[icnst] * hund_ovr_g;
      qsrflx_i[1] = sumresusp[icnst] * hund_ovr_g;
      qsrflx_i[2] = sumaqchem[icnst] * hund_ovr_g;
      qsrflx_i[3] = sumwetdep[icnst] * hund_ovr_g;
      qsrflx_i[4] = sumprevap[icnst] * hund_ovr_g;
      qsrflx_i[5] = sumprevap_hist[icnst] * hund_ovr_g;
      for (int i = 0; i < nsrflx; ++i) {
        qsrflx[icnst][i] += qsrflx_i[i] * xinv_ntsub;
      }
    }
  }
}
// =========================================================================================
// clang-format off
KOKKOS_INLINE_FUNCTION
void update_tendency_final(
    const int ntsub,   // IN  number of sub timesteps
    const int jtsub,   // IN  index of sub timesteps from the outer loop
    const int ncnst,   // IN  number of tracers to transport
    const Real dt,     // IN delta t (model time increment) [s]
    const Real dcondt[ConvProc::pcnst_extd], // IN grid-average TMR tendency for current column  [kg/kg/s]
    const bool doconvproc[], // IN  flag for doing convective transport
    Real dqdt[],           // INOUT Tracer tendency array
    Real q_i[ConvProc::gas_pcnst]) // INOUT  q(icol,kk,icnst) at current icol
{

  // -----------------------------------------------------------------------
  //  update tendencies to final output of ma_convproc_tend
  // 
  //  note that the ma_convproc_tend does not apply convective cloud processing
  //     to the stratiform-cloudborne aerosol
  //  within this routine, cloudborne aerosols are convective-cloudborne
  // 
  //  before tendencies (dcondt, which is loaded into dqdt) are returned,
  //     the convective-cloudborne aerosol tendencies must be combined
  //     with the interstitial tendencies
  //  ma_resuspend_convproc has already done this for the dcondt
  // -----------------------------------------------------------------------

  // clang-format on

  // inverse of ntsub (1.0/ntsub)
  const Real xinv_ntsub = 1.0 / ntsub;
  // delta t of sub timestep (dt/ntsub) [s]
  const Real dtsub = dt * xinv_ntsub;

  // scatter overall tendency back to full array
  // The indexing started at 2 for Fortran, so 1 for C++
  for (int icnst = 1; icnst < ncnst; ++icnst) {
    if (doconvproc[icnst]) {
      // scatter overall dqdt tendency back
      const Real dqdt_i = dcondt[icnst];
      dqdt[icnst] += dqdt_i * xinv_ntsub;
      // update the q_i for the next interation of the jtsub loop
      if (jtsub < ntsub) {
        q_i[icnst] = haero::max((q_i[icnst] + dqdt_i * dtsub), 0.0);
      }
    }
  }
}
// =========================================================================================
// clang-format off
// nlev = number of atmospheric levels: 0 <= ktop <= kbot_prevap <= nvel
// nlevp = nlev + 1
//
// (note:  TMR = tracer mixing ratio)
//
KOKKOS_INLINE_FUNCTION
void compute_column_tendency(
  const bool doconvproc_extd[ConvProc::pcnst_extd],  // IN flag for doing convective transport
  const int ktop,                                    // IN top level index
  const int kbot_prevap,                             // IN bottom level index, for resuspension and evaporation only
  const Real dpdry_i[/*nlev*/],                      // IN dp [mb]
  const Real dcondt_resusp[/*nlev*/][ConvProc::pcnst_extd],    // IN portion of TMR tendency due to resuspension [kg/kg/s]
  const Real dcondt_prevap[/*nlev*/][ConvProc::pcnst_extd],    // IN portion of TMR tendency due to precip evaporation [kg/kg/s]
  const Real dcondt_prevap_hist[/*nlev*/][ConvProc::pcnst_extd], // IN portion of TMR tendency due to precip evaporation, goes into the history [kg/kg/s]
  const Real dconudt_activa[/*nlevp*/][ConvProc::pcnst_extd], //  IN d(conu)/dt by activation [kg/kg/s]
  const Real dconudt_wetdep[/*nlevp*/][ConvProc::pcnst_extd], //  IN d(conu)/dt by wet removal[kg/kg/s]
  const Real fa_u[/*nlev*/],                       //  IN  fractional area of the updraft [fraction]
  Real sumactiva[ConvProc::pcnst_extd],  //  IN/OUT sum (over layers) of dp*dconudt_activa [kg/kg/s * mb] 
  Real sumaqchem[ConvProc::pcnst_extd],  //  IN/OUT sum (over layers) of dp*dconudt_aqchem [kg/kg/s * mb]
  Real sumwetdep[ConvProc::pcnst_extd],  //  IN/OUT sum (over layers) of dp*dconudt_wetdep [kg/kg/s * mb]
  Real sumresusp[ConvProc::pcnst_extd],  //  IN/OUT sum (over layers) of dp*dconudt_resusp [kg/kg/s * mb]
  Real sumprevap[ConvProc::pcnst_extd],  //  IN/OUT sum (over layers) of dp*dconudt_prevap [kg/kg/s * mb]
  Real sumprevap_hist[ConvProc::pcnst_extd]) // IN/OUT sum (over layers) of dp*dconudt_prevap_hist [kg/kg/s * mb]
{
  // clang-format on
  const Real dconudt_aqchem = 0; // aqueous chemistry is ignored in current code
  // initialize variables
  for (int i = 0; i < ConvProc::pcnst_extd; ++i) {
    sumactiva[i] = 0;
    sumaqchem[i] = 0;
    sumwetdep[i] = 0;
    sumresusp[i] = 0;
    sumprevap[i] = 0;
    sumprevap_hist[i] = 0;
  }

  for (int icnst = 1; icnst < ConvProc::pcnst_extd; ++icnst) {
    if (doconvproc_extd[icnst]) {
      // should go to kk=pver for dcondt_prevap, and this should be safe for
      // other sums
      for (int kk = ktop; kk < kbot_prevap; ++kk) {
        sumactiva[icnst] += dconudt_activa[kk][icnst] * dpdry_i[kk] * fa_u[kk];
        sumaqchem[icnst] += dconudt_aqchem * dpdry_i[kk] * fa_u[kk];
        sumwetdep[icnst] += dconudt_wetdep[kk][icnst] * dpdry_i[kk] * fa_u[kk];
        sumresusp[icnst] += dcondt_resusp[kk][icnst] * dpdry_i[kk];
        sumprevap[icnst] += dcondt_prevap[kk][icnst] * dpdry_i[kk];
        sumprevap_hist[icnst] += dcondt_prevap_hist[kk][icnst] * dpdry_i[kk];
      }
    }
  }
}
// =========================================================================================

KOKKOS_INLINE_FUNCTION
void tmr_tendency(const int la, const int lc, Real dcondt[ConvProc::pcnst_extd],
                  Real dcondt_resusp[ConvProc::pcnst_extd]) {
  // clang-format off
  // -----------------------------------------------------------------------
  //  calculate tendency of TMR
  // -----------------------------------------------------------------------
  // arguments (note:  TMR = tracer mixing ratio)
  /*
    in    :: la, lc             indices from assign_la_lc
    inout :: dcondt[pcnst_extd]  overall TMR tendency from convection [#/kg/s or kg/kg/s] 
    inout :: dcondt_resusp[pcnst_extd] portion of TMR tendency due to resuspension [#/kg/s or kg/kg/s]
  */
  // Only apply adjustments to dcondt for pairs of 
  // unactivated (la) and activated (lc) aerosol species
  // clang-format on
  if (la > -1 && la < ConvProc::pcnst_extd && lc > -1 &&
      lc < ConvProc::pcnst_extd) {
    // cam5 approach
    dcondt[la] += dcondt[lc];
    dcondt_resusp[la] = dcondt[lc];
    dcondt_resusp[lc] = -dcondt[lc];
    dcondt[lc] = 0;
  }
}

// =========================================================================================
KOKKOS_INLINE_FUNCTION
void ma_resuspend_convproc(Real dcondt[ConvProc::pcnst_extd],
                           Real dcondt_resusp[ConvProc::pcnst_extd]) {
  // -----------------------------------------------------------------------
  //
  //  Purpose:
  //  Calculate resuspension of activated aerosol species resulting from both
  //     detrainment from updraft and downdraft into environment
  //     subsidence and lifting of environment, which may move air from
  //        levels with large-scale cloud to levels with no large-scale cloud
  //
  //  Method:
  //  Three possible approaches were considered:
  //
  //  1. Ad-hoc #1 approach.  At each level, adjust dcondt for the activated
  //     and unactivated portions of a particular aerosol species so that the
  //     ratio of dcondt (activated/unactivate) is equal to the ratio of the
  //     mixing ratios before convection.
  //     THIS WAS IMPLEMENTED IN MIRAGE2
  //
  //  2. Ad-hoc #2 approach.  At each level, adjust dcondt for the activated
  //     and unactivated portions of a particular aerosol species so that the
  //     change to the activated portion is minimized (zero if possible).  The
  //     would minimize effects of convection on the large-scale cloud.
  //     THIS IS CURRENTLY IMPLEMENTED IN CAM5 where we assume that convective
  //     clouds have no impact on the stratiform-cloudborne aerosol
  //
  //  3. Mechanistic approach that treats the details of interactions between
  //     the large-scale and convective clouds.  (Something for the future.)
  //
  //  Author: R. Easter
  //
  //  C++ porting: only method #2 is implemented.
  // -----------------------------------------------------------------------

  // -----------------------------------------------------------------------
  //  arguments
  //  (note:  TMR = tracer mixing ratio)
  /*
    inout :: dcondt[pcnst_extd]
               overall TMR tendency from convection [#/kg/s or kg/kg/s]
    out   :: dcondt_resusp[pcnst_extd]
          portion of TMR tendency due to resuspension [#/kg/s or kg/kg/s]
          (actually, due to the adjustments made here)
  */
  for (int i = 0; i < ConvProc::pcnst_extd; ++i)
    dcondt_resusp[i] = 0;
  const int ntot_amode = AeroConfig::num_modes();
  int la, lc;
  for (int imode = 0; imode < ntot_amode; ++imode) {
    for (int ispec = -1; ispec < num_species_mode(imode); ++ispec) {
      // cloudborne aerosols are appended after intersitial
      assign_la_lc(imode, ispec, la, lc);
      tmr_tendency(la, lc, dcondt, dcondt_resusp);
    }
  }
}

// =========================================================================================
KOKKOS_INLINE_FUNCTION
void ma_precpevap(const Real dpdry_i, const Real evapc, const Real pr_flux,
                  Real &pr_flux_base, Real &pr_flux_tmp, Real &x_ratio) {
  // clang-format off
  // ------------------------------------------
  // step 1 in ma_precpevap_convproc: aerosol resuspension from precipitation evaporation
  // ------------------------------------------

  /*
  in      :: evapc         conv precipitataion evaporation rate [kg/kg/s]
  in      :: dpdry_i       pressure thickness of level [mb]
  in      :: pr_flux       precip flux at base of current layer [(kg/kg/s)*mb]
  inout   :: pr_flux_base  precip flux at an effective cloud base for calculations 
                           in a particular layer
  out     :: pr_flux_tmp   precip flux at base of current layer, after adjustment 
                           in step 1 [(kg/kg/s)*mb]
  out     :: x_ratio       ratio of adjusted and old fraction of precipitation-borne 
                           aerosol flux that is NOT resuspended, used in step 2
  */
  // clang-format on

  // a small value that variables smaller than it are considered as zero
  const Real small_value = 1.0e-30;

  // adjust pr_flux due to local evaporation
  const Real ev_flux_local = haero::max(0.0, evapc * dpdry_i);
  pr_flux_tmp =
      utils::min_max_bound(0.0, pr_flux_base, pr_flux - ev_flux_local);

  x_ratio = 0.0;
  if (pr_flux_base < small_value) {
    // this will start things fresh at the next layer
    pr_flux_base = 0.0;
    pr_flux_tmp = 0.0;
    return;
  }

  // calculate fraction of resuspension
  Real pr_ratio_old = pr_flux / pr_flux_base;
  pr_ratio_old = utils::min_max_bound(0.0, 1.0, pr_ratio_old);
  // 2: log-normal distribution
  Real frac_aer_resusp_old =
      1.0 - WetDepTemp::faer_resusp_vs_fprec_evap_mpln(1.0 - pr_ratio_old, 2);
  frac_aer_resusp_old = utils::min_max_bound(0.0, 1.0, frac_aer_resusp_old);

  Real pr_ratio_tmp =
      utils::min_max_bound(0.0, 1.0, pr_flux_tmp / pr_flux_base);
  pr_ratio_tmp = haero::min(pr_ratio_tmp, pr_ratio_old);
  // 2: log-normal distribution
  Real frac_aer_resusp_tmp =
      1.0 - WetDepTemp::faer_resusp_vs_fprec_evap_mpln(1.0 - pr_ratio_tmp, 2);
  frac_aer_resusp_tmp = utils::min_max_bound(0.0, 1.0, frac_aer_resusp_tmp);
  frac_aer_resusp_tmp = haero::min(frac_aer_resusp_tmp, frac_aer_resusp_old);

  // compute x_ratio
  if (frac_aer_resusp_tmp > small_value) {
    x_ratio = frac_aer_resusp_tmp / frac_aer_resusp_old;
  } else {
    // this will start things fresh at the next layer
    pr_flux_base = 0.0;
    pr_flux_tmp = 0.0;
  }
}

//=========================================================================================
KOKKOS_INLINE_FUNCTION
void ma_precpprod(const Real rprd, const Real dpdry_i,
                  const bool doconvproc_extd[ConvProc::pcnst_extd],
                  const Real x_ratio,
                  const int species_class[ConvProc::gas_pcnst],
                  const int mmtoo_prevap_resusp[ConvProc::gas_pcnst],
                  Real &pr_flux, Real &pr_flux_tmp, Real &pr_flux_base,
                  ColumnView wd_flux, const ColumnView dcondt_wetdep,
                  ColumnView dcondt, ColumnView dcondt_prevap,
                  ColumnView dcondt_prevap_hist) {
  // clang-format off
  // ------------------------------------------
  //  step 2 in ma_precpevap_convproc: aerosol scavenging from precipitation production
  // ------------------------------------------
  /*
  in  rprd  -  conv precip production  rate (at a certain level) [kg/kg/s]
  in  dcondt_wetdep[pcnst_extd] - portion of TMR tendency due to wet removal [kg/kg/s]
  in  dpdry_i      - pressure thickness of level [mb]
  in  doconvproc_extd[pcnst_extd]  - indicates which species to process
  in  x_ratio    - ratio of adjusted and old fraction of precipitation-borne aerosol 
                   flux that is NOT resuspended, calculated in step 1
  in  species_class(:) specify what kind of species it is. defined as
          spec_class::undefined  = 0
          spec_class::cldphysics = 1
          spec_class::aerosol    = 2
          spec_class::gas        = 3
          spec_class::other      = 4
  in  mmtoo_prevap_resusp[ConvProc::gas_pcnst]
        pointers for resuspension mmtoo_prevap_resusp values are
           >0 for aerosol mass species with    coarse mode counterpart
           -1 for aerosol mass species WITHOUT coarse mode counterpart
           -2 for aerosol number species
            0 for other species

  inout  pr_flux   - precip flux at base of current layer [(kg/kg/s)*mb]
  inout  pr_flux_tmp   - precip flux at base of current layer, after adjustment in step 1 [(kg/kg/s)*mb]
  inout  pr_flux_base   - precip flux at an effective cloud base for calculations in a particular layer
  inout  wd_flux[pcnst_extd]   - tracer wet deposition flux at base of current layer [(kg/kg/s)*mb]
  inout  dcondt[pcnst_extd]  - overall TMR tendency from convection at a certain layer [kg/kg/s]
  inout  dcondt_prevap[pcnst_extd]  - portion of TMR tendency due to precip evaporation [kg/kg/s]
  inout  dcondt_prevap_hist[pcnst_extd]   - dcondt_prevap_hist at a certain layer [kg/kg/s]
  */
  // clang-format on
  // local precip flux [(kg/kg/s)*mb]
  const Real pr_flux_local = haero::max(0.0, rprd * dpdry_i);
  pr_flux_base = haero::max(0.0, pr_flux_base + pr_flux_local);
  pr_flux =
      utils::min_max_bound(0.0, pr_flux_base, pr_flux_tmp + pr_flux_local);

  for (int icnst = 1; icnst < ConvProc::pcnst_extd; ++icnst) {

    if (doconvproc_extd[icnst]) {
      // wet deposition flux from the aerosol resuspension
      // wd_flux_tmp (updated) =
      //            (wd_flux coming into the layer) - (resuspension ! decrement)
      // wd_flux_tmp - updated wet deposition flux [(kg/kg/s)*mb]
      const Real wd_flux_tmp = haero::max(0.0, wd_flux[icnst] * x_ratio);

      // change to wet deposition flux from evaporation [(kg/kg/s)*mb]
      const Real del_wd_flux_evap =
          haero::max(0.0, wd_flux[icnst] - wd_flux_tmp);

      // wet deposition flux from the aerosol scavenging
      // wd_flux (updated) = (wd_flux after resuspension) - (scavenging !
      // increment)

      // local wet deposition flux [(kg/kg/s)*mb]
      const Real wd_flux_local =
          haero::max(0.0, -dcondt_wetdep[icnst] * dpdry_i);
      wd_flux[icnst] = haero::max(0.0, wd_flux_tmp + wd_flux_local);

      // dcondt due to wet deposition flux change [kg/kg/s]
      const Real dcondt_wdflux = del_wd_flux_evap / dpdry_i;

      // for interstitial icnst2=icnst;  for activated icnst2=icnst-pcnst
      const int icnst2 = icnst % ConvProc::gas_pcnst;

      // not sure what this mean exactly. Only do it for aerosol mass species
      // (mmtoo>0).  mmtoo<=0 represents aerosol number species
      const int mmtoo = mmtoo_prevap_resusp[icnst2];
      if (species_class[icnst2] == ConvProc::species_class::aerosol) {
        if (mmtoo > 0) {
          // add the precip-evap (resuspension) to the history-tendency of the
          // current species
          dcondt_prevap_hist[icnst] += dcondt_wdflux;
          // add the precip-evap (resuspension) to the actual tendencies of
          // appropriate coarse-mode species
          dcondt_prevap[mmtoo] += dcondt_wdflux;
          dcondt[mmtoo] += dcondt_wdflux;
        }
      } else {
        // do this for trace gases (although currently modal_aero_convproc does
        // not treat trace gases)
        dcondt_prevap_hist[icnst] += dcondt_wdflux;
        dcondt_prevap[icnst] += dcondt_wdflux;
        dcondt[icnst] += dcondt_wdflux;
      }
    }
  }
}
// =========================================================================================
KOKKOS_INLINE_FUNCTION
void ma_precpevap_convproc(
    const int ktop, const int nlev,
    const Kokkos::View<Real **, Kokkos::MemoryUnmanaged> dcondt_wetdep,
    const ColumnView rprd, const ColumnView evapc, const ColumnView dpdry_i,
    const bool doconvproc_extd[ConvProc::pcnst_extd],
    const int species_class[ConvProc::gas_pcnst],
    const int mmtoo_prevap_resusp[ConvProc::gas_pcnst], ColumnView wd_flux,
    Kokkos::View<Real **, Kokkos::MemoryUnmanaged> dcondt_prevap,
    Kokkos::View<Real **, Kokkos::MemoryUnmanaged> dcondt_prevap_hist,
    Kokkos::View<Real **, Kokkos::MemoryUnmanaged> dcondt) {
  // clang-format off
  // -----------------------------------------------------------------------
  // 
  //  Purpose:
  //  Calculate resuspension of wet-removed aerosol species resulting precip evaporation
  // 
  //      for aerosol mass   species, do non-linear resuspension to coarse mode
  //      for aerosol number species, all the resuspension is done in wetdepa_v2, so do nothing here
  // 
  //  Author: R. Easter
  // 
  // -----------------------------------------------------------------------

  // -----------------------------------------------------------------------
  //  arguments
  //  (note:  TMR = tracer mixing ratio)
  /*
    inout :: dcondt[nlev][pcnst_extd]
                 overall TMR tendency from convection [kg/kg/s]
    in    :: dcondt_wetdep[nlev][pcnst_extd]
                 portion of TMR tendency due to wet removal [kg/kg/s]
    inout :: dcondt_prevap[nlev][pcnst_extd]
                 portion of TMR tendency due to precip evaporation [kg/kg/s]
                 (actually, due to the adjustments made here)
                 (on entry, this is 0.0)
    inout :: dcondt_prevap_hist[nlev][pcnst_extd]
                 this determines what goes into the history
                    precip-evap SFSEC variables
                 currently, the SFSEC resuspension are attributed
                    to the species that got scavenged,
                    WHICH IS NOT the species that actually
                    receives the resuspension
                    when modal_aero_wetdep_resusp_opt > 0
                 so when scavenged so4_c1 is resuspended as so4_a1,
                    this resuspension column-tendency shows
                    up in so4_c1SFSES
                 this is done to allow better tracking of the
                    resuspension in the mass-budget post-processing s
    in    :: rprd   conv precip production  rate (gathered) [kg/kg/s]
    in    :: evapc  conv precip evaporation rate (gathered) [kg/kg/s]
    in    :: dpdry_i  pressure thickness of leve
    in    :: doconvproc_extd[pcnst_extd]   indicates which species to process
    in    :: species_class[gas_pcnst]   specify what kind of species it is. defined at physconst.F90
                                   undefined  = 0
                                   cldphysics = 1
                                   aerosol    = 2
                                   gas        = 3
                                   other      = 4
    in  mmtoo_prevap_resusp[ConvProc::gas_pcnst]
         pointers for resuspension mmtoo_prevap_resusp values are
            >0 for aerosol mass species with    coarse mode counterpart
            -1 for aerosol mass species WITHOUT coarse mode counterpart
            -2 for aerosol number species
             0 for other species
  */
  //
  // *** note use of non-standard units
  //
  // precip
  //    dpdry_i is mb
  //    rprd and evapc are kgwtr/kgair/s
  //    pr_flux = dpdry_i(kk)*rprd is mb*kgwtr/kgair/s
  //
  // precip-borne aerosol
  //    dcondt_wetdep is kgaero/kgair/s
  //    wd_flux = tmpdp*dcondt_wetdep is mb*kgaero/kgair/s
  //    dcondt_prevap = del_wd_flux_evap/dpdry_i is kgaero/kgair/s
  // so this works ok too
  //
  // *** dilip switched from tmpdg (or dpdry_i) to tmpdpg = tmpdp/gravit
  // that is incorrect, but probably does not matter
  //    for aerosol, wd_flux units do not matter
  //        only important thing is that tmpdp (or tmpdpg) is used
  //        consistently when going from dcondt to wd_flux then to dcondt
  // clang-format on

  // initiate variables that are integrated in vertical
  // precip flux at base of current layer [(kg/kg/s)*mb]
  Real pr_flux = 0.0;
  // precip flux at an effective cloud base for calculations in a particular
  // layer
  Real pr_flux_base = 0.0;
  // precip flux at base of current layer, after adjustment of resuspension in
  // step 1 [(kg/kg/s)*mb]
  Real pr_flux_tmp = 0;
  // ratio of adjusted and old fraction of precipitation-borne aerosol
  // flux that is NOT resuspended, calculated in step 1 and used in step 2 (see
  // below)
  Real x_ratio = 0;

  // tracer wet deposition flux at base of current layer [(kg/kg/s)*mb]
  for (int i = 0; i < ConvProc::pcnst_extd; ++i)
    wd_flux[i] = 0;
  for (int kk = 0; kk < nlev; ++kk)
    for (int i = 0; i < ConvProc::pcnst_extd; ++i)
      dcondt_prevap(kk, i) = 0;
  for (int kk = 0; kk < nlev; ++kk)
    for (int i = 0; i < ConvProc::pcnst_extd; ++i)
      dcondt_prevap_hist(kk, i) = 0;

  for (int kk = ktop; kk < nlev; ++kk) {
    // step 1 - precip evaporation and aerosol resuspension
    ma_precpevap(dpdry_i[kk], evapc[kk], pr_flux, pr_flux_base, pr_flux_tmp,
                 x_ratio);
    // step 2 - precip production and aerosol scavenging
    ColumnView dcondt_wetdep_sub =
        Kokkos::subview(dcondt_wetdep, kk, Kokkos::ALL());
    ColumnView dcondt_sub = Kokkos::subview(dcondt, kk, Kokkos::ALL());
    ColumnView dcondt_prevap_sub =
        Kokkos::subview(dcondt_prevap, kk, Kokkos::ALL());
    ColumnView dcondt_prevap_hist_sub =
        Kokkos::subview(dcondt_prevap_hist, kk, Kokkos::ALL());
    ma_precpprod(rprd[kk], dpdry_i[kk], doconvproc_extd, x_ratio, species_class,
                 mmtoo_prevap_resusp, pr_flux, pr_flux_tmp, pr_flux_base,
                 wd_flux, dcondt_wetdep_sub, dcondt_sub, dcondt_prevap_sub,
                 dcondt_prevap_hist_sub);
  }
}
} // namespace convproc

} // namespace mam4
#endif
