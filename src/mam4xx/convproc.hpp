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
  //
  // TODO: Why is this not 7?
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
  // lspectype_amode(l,m) = species type/i.d. for chemical species l
  // in aerosol mode m.  (0=sulfate, others to be defined)
  KOKKOS_INLINE_FUNCTION
  static constexpr int lspectype_amode(const int i, const int j) {
    const int lspectype_amode[maxd_aspectype][num_modes] = {
        {0, 0, 7, 3},     {3, 4, 6, 5},     {4, 6, 0, 8},     {5, 8, 5, -1},
        {7, -1, 3, -1},   {6, -1, 4, -1},   {8, -1, 8, -1},   {-1, -1, -1, -1},
        {-1, -1, -1, -1}, {-1, -1, -1, -1}, {-1, -1, -1, -1}, {-1, -1, -1, -1},
        {-1, -1, -1, -1}, {-1, -1, -1, -1}};
    return lspectype_amode[i][j];
  }

  // specdens_amode(l) = dry density (kg/m^3) of aerosol chemical species type l
  // This is indexed by the values returned from lspectype_amode and should
  // be simplified to just a direct call to aero_species(i).density
  // but there is the problem of the indexes being in a different order.
  //
  // TODO: figure out what maxd_aspectype is 14 and why not use 7?

  KOKKOS_INLINE_FUNCTION
  static constexpr Real specdens_amode(const int i) {
    // clang-format off
    const Real specdens_amode[maxd_aspectype] = {
      mam4::mam4_density_so4,
      std::numeric_limits<Real>::max(),
      std::numeric_limits<Real>::max(),
      mam4::mam4_density_pom,
      mam4::mam4_density_soa,
      mam4::mam4_density_bc ,
      mam4::mam4_density_nacl,
      mam4::mam4_density_dst,
      mam4::mam4_density_mom,
      0, 0, 0, 0, 0};
    // clang-format on
    return specdens_amode[i];
  }

  // specdens_amode(l) = dry density (kg/m^3) of aerosol chemical species type l
  // The same concerns specified for specdens_amode apply to spechygro.
  KOKKOS_INLINE_FUNCTION
  static constexpr Real spechygro(const int i) {
    // clang-format off
    const Real spechygro[maxd_aspectype] = {
       mam4::mam4_hyg_so4,
       std::numeric_limits<Real>::max(),
       std::numeric_limits<Real>::max(),
       mam4::mam4_hyg_pom,
       // (BAD CONSTANT) mam4::mam4_hyg_soa = 0.1
       0.1400000000e+00,
       mam4::mam4_hyg_bc,
       mam4::mam4_hyg_nacl,
       // (BAD CONSTANT) mam4_hyg_dst = 0.14
       0.6800000000e-01,
       mam4::mam4_hyg_mom,
       0, 0, 0, 0, 0};
    // clang-format on
    return spechygro[i];
  }

  KOKKOS_INLINE_FUNCTION
  static constexpr Real voltonumbhi_amode(const int i) {
    const Real voltonumbhi_amode[num_modes] = {
        0.4736279937e+19, 0.5026599108e+22, 0.6303988596e+16, 0.7067799781e+21};
    return voltonumbhi_amode[i];
  }

  KOKKOS_INLINE_FUNCTION
  static constexpr Real voltonumblo_amode(const int i) {
    const Real voltonumblo_amode[num_modes] = {
        0.2634717443e+22, 0.1073313330e+25, 0.4034552701e+18, 0.7067800158e+24};
    return voltonumblo_amode[i];
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
using Const_Kokkos_2D_View =
    Kokkos::View<const Real * [ConvProc::pcnst_extd], Kokkos::MemoryUnmanaged>;
using Kokkos_2D_View =
    Kokkos::View<Real * [ConvProc::pcnst_extd], Kokkos::MemoryUnmanaged>;

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

  EKAT_KERNEL_REQUIRE(ConvProc::gas_pcnst == ncnst);
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
  Const_Kokkos_2D_View dcondt_resusp,      // IN nlev - portion of TMR tendency due to resuspension [kg/kg/s]
  Const_Kokkos_2D_View dcondt_prevap,      // IN nlev - portion of TMR tendency due to precip evaporation [kg/kg/s]
  Const_Kokkos_2D_View dcondt_prevap_hist, // IN nlev - portion of TMR tendency due to precip evaporation, goes into the history [kg/kg/s]
  Const_Kokkos_2D_View dconudt_activa,     // IN nlevp- d(conu)/dt by activation [kg/kg/s]
  Const_Kokkos_2D_View dconudt_wetdep,     // IN nlevp- d(conu)/dt by wet removal[kg/kg/s]
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
        sumactiva[icnst] += dconudt_activa(kk, icnst) * dpdry_i[kk] * fa_u[kk];
        sumaqchem[icnst] += dconudt_aqchem * dpdry_i[kk] * fa_u[kk];
        sumwetdep[icnst] += dconudt_wetdep(kk, icnst) * dpdry_i[kk] * fa_u[kk];
        sumresusp[icnst] += dcondt_resusp(kk, icnst) * dpdry_i[kk];
        sumprevap[icnst] += dcondt_prevap(kk, icnst) * dpdry_i[kk];
        sumprevap_hist[icnst] += dcondt_prevap_hist(kk, icnst) * dpdry_i[kk];
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
template <typename SubView>
KOKKOS_INLINE_FUNCTION void
ma_precpprod(const Real rprd, const Real dpdry_i,
             const bool doconvproc_extd[ConvProc::pcnst_extd],
             const Real x_ratio, const int species_class[ConvProc::gas_pcnst],
             const int mmtoo_prevap_resusp[ConvProc::gas_pcnst], Real &pr_flux,
             Real &pr_flux_tmp, Real &pr_flux_base, ColumnView wd_flux,
             const SubView dcondt_wetdep, SubView dcondt, SubView dcondt_prevap,
             SubView dcondt_prevap_hist) {
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
    auto dcondt_wetdep_sub = Kokkos::subview(dcondt_wetdep, kk, Kokkos::ALL());
    auto dcondt_sub = Kokkos::subview(dcondt, kk, Kokkos::ALL());
    auto dcondt_prevap_sub = Kokkos::subview(dcondt_prevap, kk, Kokkos::ALL());
    auto dcondt_prevap_hist_sub =
        Kokkos::subview(dcondt_prevap_hist, kk, Kokkos::ALL());
    ma_precpprod(rprd[kk], dpdry_i[kk], doconvproc_extd, x_ratio, species_class,
                 mmtoo_prevap_resusp, pr_flux, pr_flux_tmp, pr_flux_base,
                 wd_flux, dcondt_wetdep_sub, dcondt_sub, dcondt_prevap_sub,
                 dcondt_prevap_hist_sub);
  }
}

// =========================================================================================
// TODO: initialize_dcondt uses multiple levels for computation but ONLY
// sets a SINGLE level on output for the loop over ktop to kbot.  So, it should
// be possible to input kk and then call in parallel from ktop to kbot.
KOKKOS_INLINE_FUNCTION
void initialize_dcondt(const bool doconvproc_extd[ConvProc::pcnst_extd],
                       const int iflux_method, const int ktop, const int kbot,
                       const int nlev, const Real dpdry_i[/* nlev */],
                       const Real fa_u[/* nlev */],
                       const Real mu_i[/* nlev+1 */],
                       const Real md_i[/* nlev+1 */], Const_Kokkos_2D_View chat,
                       Const_Kokkos_2D_View gath, Const_Kokkos_2D_View conu,
                       Const_Kokkos_2D_View cond,
                       Const_Kokkos_2D_View dconudt_activa,
                       Const_Kokkos_2D_View dconudt_wetdep,
                       const Real dudp[/* nlev */], const Real dddp[/* nlev */],
                       const Real eudp[/* nlev */], const Real eddp[/* nlev */],
                       Kokkos_2D_View dcondt) {
  // clang-format off
  // -----------------------------------------------------------------------
  //  initialize dondt and update with aerosol activation and wetdeposition
  //  will update later with dcondt_prevap and dcondt_resusp
  //  NOTE:  The approach used in convtran applies to inert tracers and
  //         must be modified to include source and sink terms
  // -----------------------------------------------------------------------

  /* cloudborne aerosol, so the arrays are dimensioned with pcnst_extd = pcnst*2
   in :: doconvproc_extd(pcnst_extd) ! flag for doing convective transport
   in :: iflux_method             ! 1=as in convtran (deep), 2=uwsh
   in :: ktop                     ! top level index
   in :: kbot                     ! bottom level index
   in :: dpdry_i(pver)            ! dp [mb]
   in :: fa_u(pver)               ! fractional area of the updraft [fraction]
   in :: mu_i(pverp)              ! mu at current i (note pverp dimension, see ma_convproc_tend) [mb/s]
   in :: md_i(pverp)              ! md at current i (note pverp dimension) [mb/s]
   in :: chat(pverp,pcnst_extd)   ! mix ratio in env at interfaces [kg/kg]
   in :: gath(pver,pcnst_extd)   ! gathered tracer array [kg/kg]
   in :: conu(pverp,pcnst_extd)   ! mix ratio in updraft at interfaces [kg/kg]
   in :: cond(pverp,pcnst_extd)   ! mix ratio in downdraft at interfaces [kg/kg]
   in :: dconudt_activa(pverp,pcnst_extd) ! d(conu)/dt by activation [kg/kg/s]
   in :: dconudt_wetdep(pverp,pcnst_extd) ! d(conu)/dt by wet removal[kg/kg/s]
   in :: dudp(pver)           ! du(i,k)*dp(i,k) at current i [mb/s]
   in :: dddp(pver)           ! dd(i,k)*dp(i,k) at current i [mb/s]
   in :: eudp(pver)           ! eu(i,k)*dp(i,k) at current i [mb/s]
   in :: eddp(pver)           ! ed(i,k)*dp(i,k) at current i [mb/s]
   out :: dcondt(pver,pcnst_extd)  ! grid-average TMR tendency for current column  [kg/kg/s]
  */
  // clang-format on
  // initialize variables
  for (int i = 0; i < nlev; ++i)
    for (int j = 0; j < ConvProc::pcnst_extd; ++j)
      dcondt(i, j) = 0.;

  // loop from ktop to kbot
  for (int kk = ktop; kk < kbot; ++kk) {
    const int kp1 = kk + 1;
    const int kp1x = haero::min(kp1, nlev - 1);
    const int km1x = haero::max(kk - 1, 0);
    const Real fa_u_dp = fa_u[kk] * dpdry_i[kk];
    for (int icnst = 1; icnst < ConvProc::pcnst_extd; ++icnst) {
      if (doconvproc_extd[icnst]) {
        // compute fluxes as in convtran, and also source/sink terms
        // (version 3 limit fluxes outside convection to mass in appropriate
        // layer (these limiters are probably only safe for positive definite
        // quantitities (it assumes that mu and md already satify a courant
        // number limit of 1)
        Real fluxin = 0, fluxout = 0;
        if (iflux_method != 2) {
          fluxin =
              mu_i[kp1] * conu(kp1, icnst) +
              mu_i[kk] * haero::min(chat(kk, icnst), gath(km1x, icnst)) -
              (md_i[kk] * cond(kk, icnst) +
               md_i[kp1] * haero::min(chat(kp1, icnst), gath(kp1x, icnst)));
          fluxout = mu_i[kk] * conu(kk, icnst) +
                    mu_i[kp1] * haero::min(chat(kp1, icnst), gath(kk, icnst)) -
                    (md_i[kp1] * cond(kp1, icnst) +
                     md_i[kk] * haero::min(chat(kk, icnst), gath(kk, icnst)));
        } else {
          // new method -- simple upstream method for the env subsidence
          // tmpa = net env mass flux (positive up) at top of layer k
          fluxin = mu_i[kp1] * conu(kp1, icnst) - md_i[kk] * cond(kk, icnst);
          fluxout = mu_i[kk] * conu(kk, icnst) - md_i[kp1] * cond(kp1, icnst);
          Real tmpa = -(mu_i[kk] + md_i[kk]);
          if (tmpa <= 0.0) {
            fluxin -= tmpa * gath(km1x, icnst);
          } else {
            fluxout += tmpa * gath(kk, icnst);
          }
          // tmpa = net env mass flux (positive up) at base of layer k
          tmpa = -(mu_i[kp1] + md_i[kp1]);
          if (tmpa >= 0.0) {
            fluxin += tmpa * gath(kp1x, icnst);
          } else {
            fluxout -= tmpa * gath(kk, icnst);
          }
        }
        //  net flux [kg/kg/s * mb]
        const Real netflux = fluxin - fluxout;

        // note for C++ refactoring:
        // I was trying to separate dconudt_activa and dconudt_wetdep out
        // into a subroutine, but for some reason it doesn't give consistent
        // dcondt values. have to leave them here.   Shuaiqi Tang, 2022
        const Real netsrce =
            fa_u_dp * (dconudt_activa(kk, icnst) + dconudt_wetdep(kk, icnst));
        dcondt(kk, icnst) = (netflux + netsrce) / dpdry_i[kk];
      }
    }
  }
}
// =========================================================================================
// TODO: compute_downdraft_mixing_ratio uses multiple levels for computation but
// ONLY sets a SINGLE level on output for the loop over ktop to kbot.  So, it
// should be possible to input kk and then call in parallel from ktop to kbot.
// It is a bit confusing that the level set is kk+1 so it would be better to
// rewrite as kkp1=>kk and kk=>kk-1 and iterate from ktop+1 to kbot inclusive.
KOKKOS_INLINE_FUNCTION
void compute_downdraft_mixing_ratio(
    const bool doconvproc_extd[ConvProc::pcnst_extd], const int ktop,
    const int kbot, const Real md_i[/* nlev+1 */], const Real eddp[/* nlev */],
    const Real gath[/* nlev */][ConvProc::pcnst_extd],
    Real cond[/* nlev+1 */][ConvProc::pcnst_extd]) {
  // clang-format off
  //----------------------------------------------------------------------
  // Compute downdraft mixing ratios from cloudtop to cloudbase
  // No special treatment is needed at k=2
  // No transformation or removal is applied in the downdraft
  // ---------------------------------------------------------------------

  /* cloudborne aerosol, so the arrays are dimensioned with pcnst_extd = pcnst*2
   in doconvproc_extd[pcnst_extd] ! flag for doing convective transport
   in ktop                     ! top level index
   in kbot                     ! bottom level index
   in md_i[pverp]              ! md at current i (note pverp dimension) [mb/s]
   in eddp[pver]               ! ed(i,k)*dp(i,k) at current i [mb/s]
   in gath[pver][(pcnst_extd]  ! gathered tracer array [kg/kg]
   inout  cond[pverp][pcnst_extd,pverp] ! mix ratio in downdraft at interfaces [kg/kg]
  */
  // clang-format on
  // threshold below which we treat the mass fluxes as zero [mb/s]
  const Real mbsth = 1.e-15;
  for (int kk = ktop; kk < kbot; ++kk) {
    const int kp1 = kk + 1;
    // md_m_eddp = downdraft massflux at kp1, without detrainment between k,kp1
    const Real md_m_eddp = md_i[kk] - eddp[kk];
    if (md_m_eddp < -mbsth) {
      for (int icnst = 1; icnst < ConvProc::pcnst_extd; ++icnst) {
        if (doconvproc_extd[icnst]) {
          cond[kp1][icnst] =
              (md_i[kk] * cond[kk][icnst] - eddp[kk] * gath[kk][icnst]) /
              md_m_eddp;
        }
      }
    }
  }
}
// ==================================================================================
KOKKOS_INLINE_FUNCTION
void update_conu_from_act_frac(Real conu[ConvProc::pcnst_extd],
                               Real dconudt[ConvProc::pcnst_extd], const int la,
                               const int lc, const Real act_frac,
                               const Real dt_u_inv) {
  // clang-format off
  // ---------------------------------------------------------------------
  // update conu and dconudt from activation fraction
  // ---------------------------------------------------------------------
  /* arguments:
   inout :: conu(pcnst_extd)    ! TMR concentration [#/kg or kg/kg]
   inout :: dconudt(pcnst_extd) ! TMR tendencies due to activation [#/kg/s or kg/kg/s]
   in    :: act_frac            ! activation fraction [fraction]
   in    :: dt_u_inv            ! 1.0/dt_u  [1/s]
   in    :: la                  ! indices for interstitial aerosols
   in    :: lc                  ! indices for in-cloud water aerosols
  */
  // clang-format on

  const Real delact = utils::min_max_bound(0.0, conu[la], conu[la] * act_frac);
  // update conu in interstitial and in-cloud condition
  conu[la] -= delact;
  conu[lc] += delact;
  // update dconu/dt
  dconudt[la] = -delact * dt_u_inv;
  dconudt[lc] = delact * dt_u_inv;
}
KOKKOS_INLINE_FUNCTION
void aer_vol_num_hygro(const Real conu[ConvProc::pcnst_extd], const Real rhoair,
                       Real vaerosol[AeroConfig::num_modes()],
                       Real naerosol[AeroConfig::num_modes()],
                       Real hygro[AeroConfig::num_modes()]) {
  // clang-format off
  // -----------------------------------------------------------------------
  //  calculate aerosol volume, number and hygroscopicity
  // -----------------------------------------------------------------------

  // -----------------------------------------------------------------------
  //  arguments:
  /*
   conu = tracer mixing ratios in updraft at top of this (current) level. The
   conu are changed by activation
   in    :: conu[pcnst_extd]       ! TMR [#/kg or kg/kg]
   in    :: rhoair                 ! air density [kg/m3]
   out   :: vaerosol[ntot_amode]   ! int+act volume [m3/m3]
   out   :: naerosol[ntot_amode]   ! interstitial+activated number conc [#/m3]
   out   :: hygro[ntot_amode]      ! current hygroscopicity for int+act [unitless]
  */
  // clang-format on

  // a small value that variables smaller than it are considered as zero for
  // aerosol volume [m3/kg]
  const Real small_vol = 1.0e-35;

  // -----------------------------------------------------------------------
  const int ntot_amode = AeroConfig::num_modes();
  for (int imode = 0; imode < ntot_amode; ++imode) {
    // compute aerosol (or a+cw) volume and hygroscopicity
    Real tmp_vol = 0.0;
    Real tmp_hygro = 0.0;
    const int nspec_amode = mam4::num_species_mode(imode);
    for (int ispec = 0; ispec < nspec_amode; ++ispec) {
      // mass divided by density
      const Real tmp_vol_spec =
          haero::max(conu[ConvProc::lmassptr_amode(ispec, imode)], 0.0) /
          ConvProc::specdens_amode(ConvProc::lspectype_amode(ispec, imode));
      // total aerosol volume
      tmp_vol += tmp_vol_spec;
      //  volume*hygro suming up for all species
      tmp_hygro += tmp_vol_spec *
                   ConvProc::spechygro(ConvProc::lspectype_amode(ispec, imode));
    }
    // change volume from m3/kgair to m3/m3air
    vaerosol[imode] = tmp_vol * rhoair;
    if (tmp_vol < small_vol) {
      hygro[imode] = 0.2;
    } else {
      hygro[imode] = tmp_hygro / tmp_vol;
    }

    // computer a (or a+cw) number and bound it
    const Real tmp_num = haero::max(conu[ConvProc::numptr_amode(imode)], 0.0);
    const Real n_min = vaerosol[imode] * ConvProc::voltonumbhi_amode(imode);
    const Real n_max = vaerosol[imode] * ConvProc::voltonumblo_amode(imode);
    naerosol[imode] = utils::min_max_bound(n_min, n_max, tmp_num * rhoair);
  }
}
// ======================================================================================
// Note: compute_wup sets wup[kk] only so can be called in parallel over the
// vertical level index, kk.
KOKKOS_INLINE_FUNCTION
void compute_wup(const int iconvtype, const int kk,
                 const Real mu_i[/* nlev+1 */],
                 const Real cldfrac_i[/* nlev */],
                 const Real rhoair_i[/* nlev */], const Real zmagl[/* nlev */],
                 Real wup[/* nlev */]) {
  // -----------------------------------------------------------------------
  //  estimate updraft velocity (wup)
  //  do it differently for deep and shallow convection
  // -----------------------------------------------------------------------

  // clang-format off
  /*
  in :: iconvtype       ! 1=deep, 2=uw shallow
  in :: kk              ! vertical level index
  in :: mu_i[pver+1]     ! mu at current i (note pverp dimension) [mb/s]
  in :: cldfrac_i[pver] ! cldfrac at current icol (with adjustments) [fraction]
  in :: rhoair_i[pver]  ! air density at current i [kg/m3]
  in :: zmagl[pver]     ! height above surface [m]
  inout :: wup[pver]    ! mean updraft vertical velocity at current level updraft [m/s]
  */
  // clang-format on
  // pre-defined minimum updraft [m/s]
  const Real w_min = 0.1;
  // pre-defined peak updraft [m/s]
  const Real w_peak = 4.0;

  const int kp1 = kk + 1;
  const Real hund_ovr_g = 100.0 / Constants::gravity;

  if (iconvtype != 1) {
    // shallow - wup = (mup in kg/m2/s) / [rhoair * (updraft area)]
    wup[kk] = (mu_i[kp1] + mu_i[kk]) * 0.5 * hund_ovr_g /
              (rhoair_i[kk] * 0.5 * cldfrac_i[kk]);
    wup[kk] = haero::max(w_min, wup[kk]);
  } else {
    // deep - the above method overestimates updraft area and underestimate wup
    // the following is based Lemone and Zipser (J Atmos Sci, 1980, p. 2455)
    // peak updraft (= 4 m/s) is sort of a "grand median" from their GATE data
    // and Thunderstorm Project data which they also show
    // the vertical profile shape is a crude fit to their median updraft profile
    // height above surface [km]
    const Real zkm = zmagl[kk] * 1.0e-3;
    if (1.0 <= zkm) {
      wup[kk] = w_peak * haero::pow((zkm / w_peak), 0.21);
    } else {
      wup[kk] = 2.9897 * haero::sqrt(zkm);
    }
    wup[kk] = utils::min_max_bound(w_min, w_peak, wup[kk]);
  }
}
// ======================================================================================
KOKKOS_INLINE_FUNCTION
void compute_massflux(const int nlev, const int ktop, const int kbot,
                      const Real dpdry_i[/* nlev */], const Real du[/* nlev */],
                      const Real eu[/* nlev */], const Real ed[/* nlev */],
                      Real mu_i[/* nlev+1 */], Real md_i[/* nlev+1 */],
                      Real &xx_mfup_max) {
  // -----------------------------------------------------------------------
  //  compute dry mass fluxes
  //  This is approximate because the updraft air is has different temp and qv
  //  than the grid mean, but the whole convective parameterization is highly
  //  approximate values below a threshold and at "top of cloudtop", "base of
  //  cloudbase" are set as zero
  // -----------------------------------------------------------------------
  // clang-format off
  /*
   in  :: nlev           ! number of levels
   in  :: ktop           ! top level index
   in  :: kbot           ! bottom level index
   in  :: dpdry_i[nlev]  ! dp [mb]
   in  :: du[nlev]       ! Mass detrain rate from updraft [1/s]
   in  :: eu[nlev]       ! Mass entrain rate into updraft [1/s]
   in  :: ed[nlev]       ! Mass entrain rate into downdraft [1/s]
   out :: mu_i[nlev+1]   ! mu at current i (note nlev+1 dimension, see ma_convproc_tend) [mb/s]
   out :: md_i[nlev+1]   ! md at current i (note nlev+1 dimension) [mb/s]
   inout :: xx_mfup_max  ! diagnostic field of column maximum updraft mass flux [mb/s]
  */
  // clang-format on
  // threshold below which we treat the mass fluxes as zero [mb/s]
  const Real mbsth = 1.e-15;

  // load mass fluxes at cloud layers
  // then load mass fluxes only over cloud layers
  // excluding "top of cloudtop", "base of cloudbase"

  //  first calculate updraft and downdraft mass fluxes for all layers
  for (int i = 0; i < nlev + 1; ++i)
    mu_i[i] = 0, md_i[i] = 0;
  // (eu-du) = d(mu)/dp -- integrate upwards, multiplying by dpdry
  for (int kk = nlev - 1; 0 <= kk; --kk) {
    mu_i[kk] = mu_i[kk + 1] + (eu[kk] - du[kk]) * dpdry_i[kk];
    xx_mfup_max = haero::max(xx_mfup_max, mu_i[kk]);
  }
  // (ed) = d(md)/dp -- integrate downwards, multiplying by dpdry
  for (int kk = 1; kk < nlev; ++kk)
    md_i[kk] = md_i[kk - 1] - ed[kk - 1] * dpdry_i[kk - 1];

  for (int kk = 0; kk < nlev + 1; ++kk) {
    if (ktop + 1 <= kk && kk < kbot) {
      // zero out values below threshold
      if (mu_i[kk] <= mbsth)
        mu_i[kk] = 0;
      if (md_i[kk] >= -mbsth)
        md_i[kk] = 0;
    } else {
      mu_i[kk] = 0, md_i[kk] = 0;
    }
  }
}
// ======================================================================================
// Because the local variable courantmax is over the whole column, this can not
// be called in parallel.  Could pass courantmax back as an array and then
// max-reduc over it.
KOKKOS_INLINE_FUNCTION
void compute_ent_det_dp(const int pver, const int ktop, const int kbot,
                        const Real dt, const Real dpdry_i[/* nlev */],
                        const Real mu_i[/* nlev+1 */],
                        const Real md_i[/* nlev+1 */],
                        const Real du[/* nlev */], const Real eu[/* nlev */],
                        const Real ed[/* nlev */], int &ntsub,
                        Real eudp[/* nlev */], Real dudp[/* nlev */],
                        Real eddp[/* nlev */], Real dddp[/* nlev */]) {
  // clang-format off
  // -----------------------------------------------------------------------
  //  calculate mass flux change (from entrainment or detrainment) in the current dp
  //  also get number of time substeps
  // -----------------------------------------------------------------------
  /*
  in  :: ii             ! index to gathered arrays
  in  :: ktop           ! top level index
  in  :: kbot           ! bottom level index
  in  :: dt             ! delta t (model time increment) [s]
  in  :: dpdry_i[pver]  ! dp [mb]
  in  :: mu_i[pverp]    ! mu at current i (note pverp dimension, see ma_convproc_tend) [mb/s]
  in  :: md_i[pverp]    ! md at current i (note pverp dimension) [mb/s]
  in  :: du[pver]       ! Mass detrain rate from updraft [1/s]
  in  :: eu[pver]       ! Mass entrain rate into updraft [1/s]
  in  :: ed[pver]       ! Mass entrain rate into downdraft [1/s]

  out :: ntsub          ! number of sub timesteps
  out :: eudp[pver]     ! eu(i,k)*dp(i,k) at current i [mb/s]
  out :: dudp[pver]     ! du(i,k)*dp(i,k) at current i [mb/s]
  out :: eddp[pver]     ! ed(i,k)*dp(i,k) at current i [mb/s]
  out :: dddp[pver]     ! dd(i,k)*dp(i,k) at current i [mb/s]
  */
  // clang-format on

  for (int i = 0; i < pver; ++i)
    eudp[i] = dudp[i] = eddp[i] = dddp[i] = 0.0;

  // maximum value of courant number [unitless]
  Real courantmax = 0.0;
  ntsub = 1;

  //  Compute updraft and downdraft "entrainment*dp" from eu and ed
  //  Compute "detrainment*dp" from mass conservation (total is mass flux
  //  difference between the top an bottom interface of this layer)
  for (int kk = ktop; kk < kbot; ++kk) {
    if ((mu_i[kk] > 0) || (mu_i[kk + 1] > 0)) {
      if (du[kk] <= 0.0) {
        eudp[kk] = mu_i[kk] - mu_i[kk + 1];
      } else {
        eudp[kk] = haero::max(eu[kk] * dpdry_i[kk], 0.0);
        dudp[kk] = (mu_i[kk + 1] + eudp[kk]) - mu_i[kk];
        if (dudp[kk] < 1.0e-12 * eudp[kk]) {
          eudp[kk] = mu_i[kk] - mu_i[kk + 1];
          dudp[kk] = 0.0;
        }
      }
    }
    if ((md_i[kk] < 0) || (md_i[kk + 1] < 0)) {
      eddp[kk] = haero::max(ed[kk] * dpdry_i[kk], 0.0);
      dddp[kk] = (md_i[kk + 1] + eddp[kk]) - md_i[kk];
      if (dddp[kk] < 1.0e-12 * eddp[kk]) {
        eddp[kk] = md_i[kk] - md_i[kk + 1];
        dddp[kk] = 0.0;
      }
    }
    // get courantmax to calculate ntsub
    courantmax =
        haero::max(courantmax, (mu_i[kk + 1] + eudp[kk] - md_i[kk] + eddp[kk]) *
                                   dt / dpdry_i[kk]);
  }
  // number of time substeps needed to maintain "courant number" <= 1
  if (courantmax > (1.0 + 1.0e-6)) {
    ntsub = 1 + static_cast<int>(courantmax);
  }
}
// ======================================================================================
// This function can not be called in parallel over kk,
// it is a recursive calculation by level
KOKKOS_INLINE_FUNCTION
void compute_midlev_height(const int nlev, const Real dpdry_i[/* nlev */],
                           const Real rhoair_i[/* nlev */],
                           Real zmagl[/* nlev */]) {
  // -----------------------------------------------------------------------
  //  compute height above surface for middle of level kk
  // -----------------------------------------------------------------------
  /*
  in  :: dpdry_i[nlev]  ! dp [mb]
  in  :: rhoair_i[nlev] ! air density [kg/m3]
  out :: zmagl[nlev]    ! height above surface at middle level [m]
  */

  const Real hund_ovr_g = 100.0 / Constants::gravity;
  const int surface = nlev - 1;
  for (int i = 0; i < nlev; ++i)
    zmagl[i] = 0;
  // at surface layer thickness [m]
  Real dz = dpdry_i[surface] * hund_ovr_g / rhoair_i[surface];
  zmagl[surface] = 0.5 * dz;
  // other levels
  for (int kk = surface - 1; 0 <= kk; --kk) {
    // add half layer below
    zmagl[kk] = zmagl[kk + 1] + 0.5 * dz;

    // update layer thickness at level kk
    dz = dpdry_i[kk] * hund_ovr_g / rhoair_i[kk];

    // add half layer in this level
    zmagl[kk] += 0.5 * dz;
  }
}

// ======================================================================================
// I think if gath were set to q_i before calling this function and then a const
// gath was passed instead of a const q_i, then this function could take a kk
// and be called in parallel over the levels.
KOKKOS_INLINE_FUNCTION
void initialize_tmr_array(const int nlev, const int iconvtype,
                          const bool doconvproc_extd[ConvProc::pcnst_extd],
                          const Real q_i[/* nlev */][ConvProc::gas_pcnst],
                          Real gath[/* nlev   */][ConvProc::pcnst_extd],
                          Real chat[/* nlev+1 */][ConvProc::pcnst_extd],
                          Real conu[/* nlev+1 */][ConvProc::pcnst_extd],
                          Real cond[/* nlev+1 */][ConvProc::pcnst_extd]) {
  // -----------------------------------------------------------------------
  //  initialize tracer mixing ratio arrays (const, chat, conu, cond)
  //  chat, conu and cond are at interfaces; interpolation needed
  //  Note: for deep convection, some values between the two layers
  //  differ significantly, use geometric averaging under certain conditions
  // -----------------------------------------------------------------------

  // clang-format off
  /*
  in :: iconvtype                 ! 1=deep, 2=uw shallow
  in :: doconvproc_extd[pcnst_extd] ! flag for doing convective transport
  in :: q_i[nlev][pcnst]          ! q(icol,kk,icnst) at current icol

  out :: gath[nlev]  [pcnst_extd]   ! gathered tracer array [kg/kg]
  out :: chat[nlev+1][pcnst_extd]   ! mix ratio in env at interfaces [kg/kg]
  out :: conu[nlev+1][pcnst_extd]   ! mix ratio in updraft at interfaces [kg/kg]
  out :: cond[nlev+1][pcnst_extd]   ! mix ratio in downdraft at interfaces [kg/kg]
  */
  // clang-format on

  // threshold of constitute as zero [kg/kg]
  const Real small_con = 1.e-36;
  // small value for relative comparison
  const Real small_rel = 1.0e-6;

  const int ncnst = ConvProc::gas_pcnst;
  const int pcnst_extd = ConvProc::pcnst_extd;

  // initiate variables
  for (int j = 0; j < nlev; ++j)
    for (int i = 0; i < pcnst_extd; ++i)
      gath[j][i] = 0;
  for (int icnst = 1; icnst < ncnst; ++icnst) {
    if (doconvproc_extd[icnst]) {
      // Gather up the constituent
      for (int kk = 0; kk < nlev; ++kk)
        gath[kk][icnst] = q_i[kk][icnst];
    }
  }

  for (int j = 0; j < nlev + 1; ++j)
    for (int i = 0; i < pcnst_extd; ++i)
      chat[j][i] = conu[j][i] = cond[j][i] = 0;

  for (int icnst = 1; icnst < ncnst; ++icnst) {
    if (doconvproc_extd[icnst]) {
      // Interpolate environment tracer values to interfaces
      for (int kk = 0; kk < nlev; ++kk) {
        const int km1 = haero::max(0, kk - 1);
        // get relative difference between the two levels

        // min gath concentration at level kk and kk-1 [kg/kg]
        const Real min_con = haero::min(gath[km1][icnst], gath[kk][icnst]);
        // max gath concentration at level kk and kk-1 [kg/kg]
        const Real max_con = haero::max(gath[km1][icnst], gath[kk][icnst]);

        // relative difference between level kk and kk-1 [unitless]
        const Real c_dif_rel =
            min_con < 0 ? 0
                        : haero::abs(gath[kk][icnst] - gath[km1][icnst]) /
                              haero::max(max_con, small_con);

        // If the two layers differ significantly use a geometric averaging
        // procedure But only do that for deep convection.  For shallow, use the
        // simple averaging which is used in subr cmfmca
        if (iconvtype != 1) {
          // simple averaging for non-deep convection
          chat[kk][icnst] = 0.5 * (gath[kk][icnst] + gath[km1][icnst]);
        } else if (c_dif_rel > small_rel) {
          // deep convection using geometric averaging

          // gath at the above (kk-1 level) [kg/kg]
          const Real c_above = haero::max(gath[km1][icnst], max_con * 1.e-12);

          // gath at the below (kk level) [kg/kg]
          const Real c_below = haero::max(gath[kk][icnst], max_con * 1.e-12);
          chat[kk][icnst] = haero::log(c_above / c_below) /
                            (c_above - c_below) * c_above * c_below;
        } else {
          // Small diff, so just arithmetic mean
          chat[kk][icnst] = 0.5 * (gath[kk][icnst] + gath[kk][icnst]);
        }
        // Set provisional up and down draft values, and tendencies
        conu[kk][icnst] = chat[kk][icnst];
        cond[kk][icnst] = chat[kk][icnst];
      }
    }
  }
  for (int icnst = 1; icnst < ncnst; ++icnst) {
    if (doconvproc_extd[icnst]) {
      // Values at surface inferface == values in lowest layer
      chat[nlev][icnst] = gath[nlev - 1][icnst];
      conu[nlev][icnst] = gath[nlev - 1][icnst];
      cond[nlev][icnst] = gath[nlev - 1][icnst];
    }
  }
}

// ======================================================================================
KOKKOS_INLINE_FUNCTION
void set_cloudborne_vars(const bool doconvproc[ConvProc::gas_pcnst],
                         Real aqfrac[ConvProc::pcnst_extd],
                         bool doconvproc_extd[ConvProc::pcnst_extd]) {
  // -----------------------------------------------------------------------
  //  set cloudborne aerosol related variables:
  //  doconvproc_extd: extended array for both activated and unactivated
  //  aerosols aqfrac: set as 1.0 for activated aerosols and 0.0 otherwise
  // -----------------------------------------------------------------------
  /*
  // cloudborne aerosol, so the arrays are dimensioned with pcnst_extd = pcnst*2
  in :: doconvproc[pcnst]    ! flag for doing convective transport
  out :: doconvproc_extd[pcnst_extd]    ! flag for doing convective transport
  out :: aqfrac[pcnst_extd]  ! aqueous fraction of constituent in updraft
  [fraction]
  */
  const int pcnst_extd = ConvProc::pcnst_extd;
  const int gas_pcnst = ConvProc::gas_pcnst;
  const int num_modes = AeroConfig::num_modes();
  int la, lc;

  for (int i = 0; i < pcnst_extd; ++i)
    doconvproc_extd[i] = false;

  for (int i = 1; i < gas_pcnst; ++i)
    doconvproc_extd[i] = doconvproc[i];

  for (int i = 0; i < pcnst_extd; ++i)
    aqfrac[i] = 0;

  for (int imode = 0; imode < num_modes; ++imode) {
    const int nspec_amode = mam4::num_species_mode(imode);
    for (int ispec = 0; ispec < nspec_amode; ++ispec) {
      // append cloudborne aerosols after intersitial
      assign_la_lc(imode, ispec, la, lc);
      if (doconvproc[la]) {
        doconvproc_extd[lc] = true;
        aqfrac[lc] = 1.0;
      }
    }
  }
}
// ======================================================================================
KOKKOS_INLINE_FUNCTION
void update_qnew_ptend(const bool dotend[ConvProc::gas_pcnst],
                       const bool is_update_ptend,
                       const Real dqdt[ConvProc::gas_pcnst], const Real dt,
                       bool ptend_lq[ConvProc::gas_pcnst],
                       Real ptend_q[ConvProc::gas_pcnst],
                       Real qnew[ConvProc::gas_pcnst]) {
  // ---------------------------------------------------------------------------------------
  // update qnew, ptend_q and ptend_lq
  // ---------------------------------------------------------------------------------------

  // Arguments
  // clang-format off
  /*
   in :: dotend[pcnst]     ! if do tendency
   in :: is_update_ptend   ! if update ptend with dqdt
   in :: dqdt[pcnst] ! time tendency of tracer [kg/kg/s]
   in :: dt                ! model time step [s]
   inout :: ptend_lq[pcnst]  ! if do tendency
   inout :: ptend_q[pcnst] ! time tendency of q [kg/kg/s]
   inout :: qnew[pcnst]    ! Tracer array including moisture [kg/kg]
  */
  // clang-format on 
  for (int ll = 0; ll < ConvProc::gas_pcnst; ++ll) {
    if (dotend[ll]) {
      // calc new q (after ma_convproc_sh_intr)
      qnew[ll] = haero::max(0.0, qnew[ll] + dt*dqdt[ll]);

      if ( is_update_ptend ) {
        // add dqdt onto ptend_q and set ptend_lq
        ptend_lq[ll] = true;
        ptend_q[ll] += dqdt[ll];
      }
    }
  }
}
// ======================================================================================
// This can be parallelized over kk. All the "nlev" dimensioned arrays could be subscripted 
// with kk and passed as scalars or 1D arrays.
KOKKOS_INLINE_FUNCTION
void compute_wetdep_tend(
  const bool doconvproc_extd[ConvProc::pcnst_extd],
  const int kk,     
  const Real dt,   
  const Real dt_u[/* nlev */],   
  const Real dp_i[/* nlev */],   
  const Real cldfrac_i[/* nlev */],      
  const Real mu_p_eudp[/* nlev */],   
  const Real aqfrac[ConvProc::pcnst_extd],         
  const Real icwmr[/* nlev */],          
  const Real rprd[/* nlev */],       
  Real conu[/* nlev+1 */][ConvProc::pcnst_extd],           
  Real dconudt_wetdep[/* nlev+1 */][ConvProc::pcnst_extd])
{
  // clang-format off
  // -----------------------------------------------------------------------
  //  compute tendency from wet deposition
  // 
  //     rprd               = precip formation as a grid-cell average (kgW/kgA/s)
  //     icwmr              = cloud water MR within updraft area (kgW/kgA)
  //     fupdr              = updraft fractional area (--)
  //     A = rprd/fupdr     = precip formation rate within updraft area (kgW/kgA/s)
  //     clw_preloss = cloud water MR before loss to precip
  //                 = icwmr + dt*(rprd/fupdr)
  //     B = A/clw_preloss  = (rprd/fupdr)/(icwmr + dt*rprd/fupdr)
  //                        = rprd/(fupdr*icwmr + dt*rprd)
  //                        = first-order removal rate (1/s)
  //     C = dp/(mup/fupdr) = updraft air residence time in the layer (s)
  // 
  //     fraction removed = (1.0 - exp(-cdt)) where
  //                  cdt = B*C = (fupdr*dp/mup)*[rprd/(fupdr*icwmr + dt*rprd)]
  // 
  //     Note1:  *** cdt is now sensitive to fupdr, which we do not really know,
  //                 and is not the same as the convective cloud fraction
  //     Note2:  dt is appropriate in the above cdt expression, not dtsub
  // 
  //     Apply wet removal at levels where
  //        icwmr(k) > clw_cut  AND  rprd(k) > 0.0
  //     as wet removal occurs in both liquid and ice clouds
  // -----------------------------------------------------------------------
  /*
   cloudborne aerosol, so the arrays are dimensioned with pcnst_extd = pcnst*2
   in :: doconvproc_extd[pcnst_extd] ! flag for doing convective transport
   in :: kk                   ! vertical level index
   in :: dt                   ! Model timestep [s]
   in :: dt_u[pver]           ! lagrangian transport time in the updraft[s]
   in :: dp_i[pver]            ! dp [mb]
   in :: cldfrac_i[pver]      ! cldfrac at current (with adjustments) [fraction]
   in :: mu_p_eudp[pver]      ! = mu_i[kp1] + eudp[k] [mb/s]
   in :: aqfrac[pcnst_extd]   ! aqueous fraction of constituent in updraft [fraction]
   in :: icwmr[pver]    ! Convective cloud water from zm scheme [kg/kg]
   in :: rprd[pver]     ! Convective precipitation formation rate [kg/kg/s]
   inout :: conu[pverp][pcnst_extd]   ! mix ratio in updraft at interfaces [kg/kg]
   inout :: dconudt_wetdep[pverp][pcnst_extd] ! d(conu)/dt by wet removal[kg/kg/s]
  */
  // clang-format on
  // cutoff value of cloud water for doing updraft [kg/kg]
  const Real clw_cut = 1.0e-6;

  // (in-updraft first order wet removal rate) * dt [unitless]
  Real cdt = 0.0;
  if (icwmr[kk] > clw_cut && rprd[kk] > 0.0) {
    const Real half_cld = 0.5 * cldfrac_i[kk];
    cdt = (half_cld * dp_i[kk] / mu_p_eudp[kk]) * rprd[kk] /
          (half_cld * icwmr[kk] + dt * rprd[kk]);
  }
  if (cdt > 0.0) {
    const Real expcdtm1 = haero::exp(-cdt) - 1;
    for (int icnst = 1; icnst < ConvProc::pcnst_extd; ++icnst) {
      if (doconvproc_extd[icnst]) {
        dconudt_wetdep[kk][icnst] = conu[kk][icnst] * aqfrac[icnst] * expcdtm1;
        conu[kk][icnst] += dconudt_wetdep[kk][icnst];
        dconudt_wetdep[kk][icnst] /= dt_u[kk];
      }
    }
  }
}

} // namespace convproc
} // namespace mam4
#endif
