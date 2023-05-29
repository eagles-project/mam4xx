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

} // namespace convproc

} // namespace mam4

#endif
