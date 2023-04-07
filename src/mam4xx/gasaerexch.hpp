// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MAM4XX_GASAEREXCH_HPP
#define MAM4XX_GASAEREXCH_HPP

#include "mam4xx/aero_modes.hpp"
#include <mam4xx/aero_config.hpp>
#include <mam4xx/gasaerexch_soaexch.hpp>
#include <mam4xx/mam4_types.hpp>

#include <Kokkos_Array.hpp>
#include <haero/atmosphere.hpp>
#include <haero/constants.hpp>
#include <haero/haero.hpp>
#include <iomanip>
#include <iostream>

namespace mam4 {

/// @class GasAerExch
/// This class implements MAM4's gas/aersol exchange  parameterization. Its
/// structure is defined by the usage of the impl_ member in the AeroProcess
/// class in
/// ../aero_process.hpp.
class GasAerExch {
public:
  static constexpr int num_gas_to_aer = 2;
  static constexpr int num_mode = AeroConfig::num_modes();
  static constexpr int num_gas = AeroConfig::num_gas_ids();
  static constexpr int num_aer = AeroConfig::num_aerosol_ids();
  static constexpr int nait = static_cast<int>(ModeIndex::Aitken);
  static constexpr int npca = static_cast<int>(ModeIndex::PrimaryCarbon);
  static constexpr int igas_h2so4 = static_cast<int>(GasId::H2SO4);
  static constexpr int igas_soag = static_cast<int>(GasId::SOAG);
  static constexpr int igas_nh3 = static_cast<int>(GasId::NH3);
  static constexpr int iaer_h2so4 = static_cast<int>(GasId::H2SO4);
  static constexpr int iaer_so4 = static_cast<int>(AeroId::SO4);
  static constexpr int iaer_pom = static_cast<int>(AeroId::POM);
  static constexpr int iaer_soag_bgn = static_cast<int>(AeroId::SOA);
  static constexpr int iaer_soag_end = static_cast<int>(AeroId::SOA);

  KOKKOS_INLINE_FUNCTION
  static const ModeIndex (&Modes())[num_mode] {
    static const ModeIndex modes[num_mode] = {
        ModeIndex::Accumulation, ModeIndex::Aitken, ModeIndex::Coarse,
        ModeIndex::PrimaryCarbon};
    return modes;
  }
  KOKKOS_INLINE_FUNCTION
  static const GasId (&Gases())[num_gas] {
    static const GasId gases[num_gas] = {GasId::SOAG, GasId::H2SO4, GasId::NH3};
    return gases;
  }
  // AeroId::NH4 is a future enhancement.
  static constexpr int iaer_nh4 = -1; // static_cast<int>(AeroId::NH4);
                                      //
  // what aerosol species does the gas become when condensing?
  // There are two gases that condense to aerosols:
  static constexpr int igas_nh3_index = 2;

  // There are three gases, AeroConfig::num_gas_ids(), and for each
  // one it can condense to an aerosol.
  // 0 = GasId::SOAG  -> AeroId::SOA
  // 1 = GasId::H2SO4 -> AeroId::SO4
  // 2 = GasId::NH3   -> AeroId::None
  // Gas/Aerosol exchange does not handle NH3 but the code is
  // written in an extensible way so it could in the future.
  KOKKOS_INLINE_FUNCTION
  static constexpr AeroId gas_to_aer(const GasId gas) {
    const AeroId gas_to_aer[num_gas] = {AeroId::SOA, AeroId::SO4, AeroId::None};
    return gas_to_aer[static_cast<int>(gas)];
  }
  //------------------------------------------------------------------
  // MAM4xx currently assumes that the uptake rate of other gases
  // are proportional to the uptake rate of sulfuric acid gas (H2SO4).
  // Here the array uptk_rate_factor contains the uptake rate ratio
  // w.r.t. that of H2SO4.
  //------------------------------------------------------------------
  //  Indices correspond to those in idx_gas_to_aer: gas_soag, gas_h2so4,
  //  gas_nh3
  KOKKOS_INLINE_FUNCTION
  static constexpr Real uptk_rate_factor(const int i) {
    const Real uptk_rate[num_gas] = {Constants::soag_h2so4_uptake_coeff_ratio,
                                     1.0,
                                     Constants::nh3_h2so4_uptake_coeff_ratio};
    return uptk_rate[i];
  }

  // -----------------------------------------------------------------
  // The NA, ANAL, and IMPL flags are used in the eqn_and_numerics_category
  // array to flag how to deal with gas-aerosol mass exchange.
  //
  // MAM currently uses a splitting method to deal with gas-aerosol
  // mass exchange. A quasi-analytical solution assuming time step-wise
  // constant uptake rate is applied to nonvolatile species, while
  // an implicit time stepping method with adaptive step size
  // is applied to gas-phase SOA species which are assumed semi-volatile.
  // There are two different subroutines to deal
  // with the two categories of cases. The array
  // eqn_and_numerics_category flags which gas species belongs to
  // which category using the flags in this enum.
  // -----------------------------------------------------------------
  enum { NA, ANAL, IMPL };

  // process-specific configuration data (if any)
  struct Config {
    Config() {}
    Config(const Config &) = default;
    ~Config() = default;
    Config &operator=(const Config &) = default;
    Real dtsub_soa_fixed = -1;
    int ntot_soamode = 4;

    // Default Aging:
    // 0: Accumulation - No
    // 1: Aitken - No
    // 2: Coarse - No
    // 3: PrimaryCarbon - Yes
    bool l_mode_can_age[num_mode] = {false, false, false, true};

    bool calculate_gas_uptake_coefficient = true;

    // Do we have NH3? Not something supported at this time.
    static constexpr bool igas_nh3 = false;

    // qgas_netprod_otrproc = gas net production rate from other processes
    // such as gas-phase chemistry and emissions (mol/mol/s)
    // this allows the condensation (gasaerexch) routine to apply production and
    // condensation loss together, which is more accurate numerically
    // NOTE - must be >= zero, as numerical method can fail when it is negative
    // NOTE - currently only the values for h2so4 and nh3 should be non-zero
    Real qgas_netprod_otrproc[num_gas] = {0, 5.0e-016, 0};
  };

  // name -- unique name of the process implemented by this class
  const char *name() const { return "MAM4 gas/aersol exchange"; }

  // init -- initializes the implementation with MAM4's configuration
  void init(const AeroConfig &aero_config,
            const Config &process_config = Config());

  // validate -- validates the given atmospheric state and prognostics against
  // assumptions made by this implementation, returning true if the states are
  // valid, false if not
  KOKKOS_INLINE_FUNCTION
  bool validate(const AeroConfig &config, const ThreadTeam &team,
                const Atmosphere &atm, const Prognostics &progs) const {
    // Make sure relevant atmospheric quantities are physical.
    const int nk = atm.num_levels();
    int violations = 0;
    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team, nk),
        KOKKOS_LAMBDA(int k, int &violation) {
          if ((atm.temperature(k) < 0) || (atm.pressure(k) < 0) ||
              (atm.vapor_mixing_ratio(k) < 0)) {
            violation = 1;
          }
        },
        violations);

    if (violations == 0) { // all clear so far
      // Check for negative mixing ratios.
      if (!progs.quantities_nonnegative(team)) {
        ++violations;
      }
    }
    return (violations > 0);
  }

  // compute_tendencies -- computes tendencies and updates diagnostics
  // NOTE: that both diags and tends are const below--this means their views
  // NOTE: are fixed, but the data in those views is allowed to vary.
  KOKKOS_INLINE_FUNCTION
  void compute_tendencies(const AeroConfig &config, const ThreadTeam &team,
                          Real t, Real dt, const Atmosphere &atm,
                          const Prognostics &progs, const Diagnostics &diags,
                          const Tendencies &tends) const;

private:
  // Gas-Aerosol-Exchange-specific configuration
  Config config_;

  bool l_gas_condense_to_mode[num_gas][num_mode] = {};
  int eqn_and_numerics_category[num_gas] = {};
  Real modes_mean_std_dev[num_mode] = {};
};

namespace gasaerexch {

KOKKOS_INLINE_FUNCTION
void mam_gasaerexch_1subarea_1gas_nonvolatile(
    const Real dt, const Real qgas_netprod_otrproc,
    const Real uptkaer[GasAerExch::num_mode], Real &qgas_cur, Real &qgas_avg,
    Real qaer_cur[GasAerExch::num_mode]) {

  // qgas_netprod_otrproc = gas net production rate from other processes
  //    such as gas-phase chemistry and emissions (mol/mol/s)
  // this allows the condensation (gasaerexch) routine to apply production and
  // condensation loss
  //    together, which is more accurate numerically
  // NOTE - must be >= zero, as numerical method can fail when it is negative
  // NOTE - currently only the values for h2so4 and nh3 should be non-zero

  const int num_mode = GasAerExch::num_mode;

  Real qaer_prev[num_mode];
  // Save current values as the "previous" value for new time step
  for (int i = 0; i < num_mode; ++i)
    qaer_prev[i] = qaer_cur[i];
  // misc. quantities that will occur repeatedly later equations

  // the uptake coefficient of a single gas species
  // summed over all aerosol modes
  Real total_uptake_coeff_of_all_modes = 0;
  for (int i = 0; i < num_mode; ++i)
    total_uptake_coeff_of_all_modes += uptkaer[i];

  const Real tmp_kxt = total_uptake_coeff_of_all_modes * dt; // uptake rate * dt
  const Real tmp_pxt = qgas_netprod_otrproc * dt;            // prodc. rate * dt

  // zqgas_init(was tmp_q1) = mix - rat at t = tcur zqgas_init = qgas_prv
  const Real zqgas_init = qgas_cur;

  // zqgas_end (was tmp_q3) = mix-rat at t=tcur+dt
  // zqgas_avg (was tmp_q4) = avg mix-rat between t=tcur and t=tcur+dt
  if (tmp_kxt < 1.0e-20) {
    // Consider uptake to aerosols = 0.0, hence there is no change in aerosol
    // mass; gas concentration is updated using the passed-in production rate
    // which is considered step-wise constant
    qgas_cur = zqgas_init + tmp_pxt;
    qgas_avg = zqgas_init + tmp_pxt * 0.5;
  } else {
    // tmp_kxt >= 1.0e-20_wp: there is non-negligible condensation;
    // calculate amount of gas condensation and update gas concentration.
    Real zqgas_end;
    Real zqgas_avg;
    if (tmp_kxt > 0.001) {
      // Stronger condesation, no worry about division by zero or singularity;
      // calculate analytical solution assuming step-wise constant condensation
      // coeff.
      const Real zqgas_equil = tmp_pxt / tmp_kxt;
      zqgas_end = (zqgas_init - zqgas_equil) * exp(-tmp_kxt) + zqgas_equil;
      zqgas_avg = (zqgas_init - zqgas_equil) * (1.0 - exp(-tmp_kxt)) / tmp_kxt +
                  zqgas_equil;
    } else {
      // Weaker condensation, use Taylor expansion to avoid large error
      // resulting from small denominator
      const Real tmp_kxt2 = tmp_kxt * tmp_kxt;
      zqgas_end = zqgas_init * (1.0 - tmp_kxt + tmp_kxt2 * 0.5) +
                  tmp_pxt * (1.0 - tmp_kxt * 0.5 + tmp_kxt2 / 6.0);
      zqgas_avg = zqgas_init * (1.0 - tmp_kxt * 0.5 + tmp_kxt2 / 6.0) +
                  tmp_pxt * (0.5 - tmp_kxt / 6.0 + tmp_kxt2 / 24.0);
    }
    qgas_cur = zqgas_end;
    qgas_avg = zqgas_avg;
    // amount of condensed mass
    const Real tmp_qdel_cond = (zqgas_init + tmp_pxt) - zqgas_end;

    // Distribute the condensed mass to different aerosol modes
    for (int n = 0; n < num_mode; ++n) {
      if (0 < uptkaer[n])
        qaer_cur[n] =
            qaer_prev[n] +
            tmp_qdel_cond * (uptkaer[n] / total_uptake_coeff_of_all_modes);
    }
  }
}

//------------------------------------------------------------------------
// gas_diffusivity       ! (m2/s)
KOKKOS_INLINE_FUNCTION
Real gas_diffusivity(
    const Real &T_in_K,     // temperature (K)
    const Real &p_in_atm,   // pressure (atmospheres)
    const Real mw_gas,      // molec. weight of the condensing gas (g/mol)
    const Real mw_air_gmol, // molec. weight of air (g/mol)
    const Real vd_gas,      // molec. diffusion volume of the condensing gas
    const Real vd_air) {    // molec. diffusion volume of air

  const Real onethird = 1.0 / 3.0;

  const Real gas_diffusivity =
      (1.0e-7 * haero::pow(T_in_K, 1.75) *
       haero::sqrt(1.0 / mw_gas + 1.0 / mw_air_gmol)) /
      (p_in_atm *
       haero::pow(haero::pow(vd_gas, onethird) + haero::pow(vd_air, onethird),
                  2.0));

  return gas_diffusivity;
}

//-----------------------------------------------------------------------
//  mean_molecular_speed    ! (m/s)
KOKKOS_INLINE_FUNCTION
Real mean_molecular_speed(
    const Real &temp,          // temperature (K)
    const Real rmw,            // molec. weight (g/mol)
    const Real r_universal_mJ, // universal gas constant (mJ/K mol)
    const Real pi) {
  const Real mean_molecular_speed =
      haero::sqrt(8.0 * r_universal_mJ * temp / (pi * rmw));

  return mean_molecular_speed;
}

//------------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION
Real fuchs_sutugin(const Real &D_p, const Real &gasfreepath,
                   const Real accomxp283, const Real accomxp75) {
  const Real knudsen = 2.0 * gasfreepath / D_p;

  // fkn = ( 0.75*accomcoef*(1. + xkn) ) /
  //       ( xkn*xhn + xkn + 0.283*xkn*accomcoef + 0.75*accomcoef )

  const Real fuchs_sutugin =
      (accomxp75 * (1.0 + knudsen)) /
      (knudsen * (knudsen + 1.0 + accomxp283) + accomxp75);

  return fuchs_sutugin;
}

KOKKOS_INLINE_FUNCTION
void gas_aer_uptkrates_1box1gas(
    const bool l_condense_to_mode[GasAerExch::num_mode], const Real temp,
    const Real pmid, const Real pstd, const Real mw_gas, const Real mw_air_gmol,
    const Real vol_molar_gas, const Real vol_molar_air, const Real accom,
    const Real r_universal_mJ, const Real pi, const Real beta_inp,
    const int nghq, const Real dgncur_awet[GasAerExch::num_mode],
    const Real lnsg[GasAerExch::num_mode], Real uptkaer[GasAerExch::num_mode]) {
  //----------------------------------------------------------------------
  //  Computes   uptake rate parameter uptkaer[0:num_mode] =
  //  uptkrate[0:num_mode]
  //
  //                           /
  //  where      uptkrate(i) = | gas_conden_rate(Dp) n_i(lnDp) dlnDp
  //                           /
  //
  //  is the uptake rate for aerosol mode i with size distribution n_i(lnDp)
  //  and number concentration of = 1 #/m3; aernum(i) is the actual number
  //  mixing ratio of mode i in the unit of  #/kmol-air, and aircon is
  //  the air concentration in the unit of kmol/m3.
  //
  //  gas_conden_rate(D_p) = 2 * pi * gasdiffus * D_p * F(Kn,ac), with
  //          gasdiffus = gas diffusivity
  //          F(Kn,ac) = Fuchs-Sutugin correction factor
  //          Kn = Knudsen number (which is a function of Dp)
  //          ac = accomodation coefficient (constant for each gas species)
  //----------------------------------------------------------------------
  //  using Gauss-Hermite quadrature of order nghq=2
  //
  //      D_p = particle diameter (cm)
  //      x = ln(D_p)
  //      dN/dx = log-normal particle number density distribution
  //----------------------------------------------------------------------
  const Real tworootpi = 2 * haero::sqrt(pi);
  const Real root2 = haero::sqrt(2.0);
  const Real one = 1.0;
  const Real two = 2.0;

  // Dick's old version
  // integer, parameter :: nghq = 2
  // real(wp), save :: xghq(nghq), wghq(nghq) ! quadrature abscissae and
  // weights data xghq / 0.70710678, -0.70710678 / data wghq / 0.88622693,
  // 0.88622693 /

  // choose
  // nghq-----------------------------------------------------------------
  const Kokkos::Array<Real, 20> xghq_20 = {
      -5.3874808900112,  -4.6036824495507, -3.9447640401156, -3.3478545673832,
      -2.7888060584281,  -2.2549740020893, -1.7385377121166, -1.2340762153953,
      -0.73747372854539, -0.2453407083009, 0.2453407083009,  0.73747372854539,
      1.2340762153953,   1.7385377121166,  2.2549740020893,  2.7888060584281,
      3.3478545673832,   3.9447640401156,  4.6036824495507,  5.3874808900112};
  const Kokkos::Array<Real, 20> wghq_20 = {
      2.229393645534e-13, 4.399340992273e-10, 1.086069370769e-7,
      7.80255647853e-6,   2.283386360164e-4,  0.003243773342238,
      0.024810520887464,  0.10901720602002,   0.28667550536283,
      0.46224366960061,   0.46224366960061,   0.28667550536283,
      0.10901720602002,   0.024810520887464,  0.003243773342238,
      2.283386360164e-4,  7.80255647853e-6,   1.086069370769e-7,
      4.399340992273e-10, 2.229393645534e-13};
  const Kokkos::Array<Real, 10> xghq_10 = {
      -3.436159118837737603327,  -2.532731674232789796409,
      -1.756683649299881773451,  -1.036610829789513654178,
      -0.3429013272237046087892, 0.3429013272237046087892,
      1.036610829789513654178,   1.756683649299881773451,
      2.532731674232789796409,   3.436159118837737603327};
  const Kokkos::Array<Real, 10> wghq_10 = {
      7.64043285523262062916e-6,  0.001343645746781232692202,
      0.0338743944554810631362,   0.2401386110823146864165,
      0.6108626337353257987836,   0.6108626337353257987836,
      0.2401386110823146864165,   0.03387439445548106313616,
      0.001343645746781232692202, 7.64043285523262062916E-6};
  const Kokkos::Array<Real, 4> xghq_4 = {-1.6506801238858, -0.52464762327529,
                                         0.52464762327529, 1.6506801238858};
  const Kokkos::Array<Real, 4> wghq_4 = {0.081312835447245, 0.8049140900055,
                                         0.8049140900055, 0.081312835447245};
  const Kokkos::Array<Real, 2> xghq_2 = {-7.0710678118654746e-01,
                                         7.0710678118654746e-01};
  const Kokkos::Array<Real, 2> wghq_2 = {8.8622692545275794e-01,
                                         8.8622692545275794e-01};
  Real const *xghq = nullptr;
  Real const *wghq = nullptr;
  if (20 == nghq) {
    xghq = xghq_20.data();
    wghq = wghq_20.data();
  } else if (10 == nghq) {
    xghq = xghq_10.data();
    wghq = wghq_10.data();
  } else if (4 == nghq) {
    xghq = xghq_4.data();
    wghq = wghq_4.data();
  } else if (2 == nghq) {
    xghq = xghq_2.data();
    wghq = wghq_2.data();
  } else {
    printf("nghq integration option is not available: %d, "
           "valid are 20, 10, 4, and 2\n",
           nghq);
    Kokkos::abort("Invalid integration order requested.");
  }
  //-----------------------------------------------------------------------

  // pressure (atmospheres)
  const Real p_in_atm = pmid / pstd;
  // gas diffusivity (m2/s)
  const Real gasdiffus = gas_diffusivity(temp, p_in_atm, mw_gas, mw_air_gmol,
                                         vol_molar_gas, vol_molar_air);
  // gas mean free path (m)
  const Real molecular_speed =
      mean_molecular_speed(temp, mw_gas, r_universal_mJ, pi);
  const Real gasfreepath = 3.0 * gasdiffus / molecular_speed;
  const Real accomxp283 = accom * 0.283;
  const Real accomxp75 = accom * 0.75;

  // outermost loop over all modes
  for (int n = 0; n < GasAerExch::num_mode; ++n) {
    const Real lndpgn = haero::log(dgncur_awet[n]); // (m)

    // beta = dln(uptake_rate)/dln(D_p)
    //      = 2.0 in free molecular regime, 1.0 in continuum regime
    // if uptake_rate ~= a * (D_p**beta), then the 2 point quadrature
    // is very accurate
    Real beta = 0;
    if (std::abs(beta_inp - 1.5) > 0.5) {
      // D_p = dgncur_awet(n) * haero::exp( 1.5*(lnsg[n]**2) )
      const Real D_p = dgncur_awet[n];
      const Real knudsen = two * gasfreepath / D_p;

      // tmpa = dln(fuchs_sutugin)/d(knudsen)
      const Real tmpa =
          one / (one + knudsen) -
          (two * knudsen + one + accomxp283) /
              (knudsen * (knudsen + one + accomxp283) + accomxp75);
      beta = one - knudsen * tmpa;
      beta = haero::max(one, haero::min(two, beta));
    } else {
      beta = beta_inp;
    }
    const Real constant =
        tworootpi *
        haero::exp(beta * lndpgn + 0.5 * haero::pow(beta * lnsg[n], 2.0));

    // sum over gauss-hermite quadrature points
    Real sumghq = 0.0;
    for (int iq = 0; iq < nghq; ++iq) {
      const Real lndp =
          lndpgn + beta * lnsg[n] * lnsg[n] + root2 * lnsg[n] * xghq[iq];
      const Real D_p = haero::exp(lndp);

      const Real hh = fuchs_sutugin(D_p, gasfreepath, accomxp283, accomxp75);
      sumghq += wghq[iq] * D_p * hh / haero::pow(D_p, beta);
    }
    // gas-to-aerosol mass transfer rates
    // (1/s) for number concentration = 1 #/m3
    const Real uptkrate = constant * gasdiffus * sumghq;

    // --------------------------------------------------------------------
    // Unit of uptkrate is for number = 1 #/m3.
    // --------------------------------------------------------------------
    uptkaer[n] = l_condense_to_mode[n] ? uptkrate : 0.0; // zero means no uptake
  }
}

KOKKOS_INLINE_FUNCTION
void mam_gasaerexch_1subarea(
    const int nghq,                               // in
    const int igas_h2so4,                         // in
    const bool igas_nh3,                          // in
    const int ntot_soamode,                       // in
    const AeroId gas_to_aer[GasAerExch::num_gas], // in
    const int iaer_so4,                           // in
    const int iaer_pom,                           // in
    const bool l_calc_gas_uptake_coeff,           // in
    const bool l_gas_condense_to_mode[GasAerExch::num_gas]
                                     [GasAerExch::num_mode],  // in
    const int eqn_and_numerics_category[GasAerExch::num_gas], // in
    const Real dt,                                            // in
    const Real dtsub_soa_fixed,                               // in
    const Real temp,                                          // in
    const Real pmid,                                          // in
    const Real aircon,                                        // in
    const int ngas, Real qgas_cur[GasAerExch::num_gas],       // in
    Real qgas_avg[GasAerExch::num_gas],                       // in/out
    const Real qgas_netprod_otrproc[GasAerExch::num_gas],     // in
    Real qaer_cur[AeroConfig::num_aerosol_ids()]
                 [GasAerExch::num_mode],                     // in/out
    Real qnum_cur[GasAerExch::num_mode],                     // in/out
    const Real dgn_awet[GasAerExch::num_mode],               // in
    const Real alnsg_aer[GasAerExch::num_mode],              // in
    const Real uptk_rate_factor[GasAerExch::num_gas],        // in
    Real uptkaer[GasAerExch::num_gas][GasAerExch::num_mode], // inout
    Real &uptkrate_h2so4,                                    // out
    int &niter_out,                                          // out
    Real &g0_soa_out) {                                      // out
  const int num_mode = GasAerExch::num_mode;
  const int num_gas = GasAerExch::num_gas;

  const Real pstd = Constants::pressure_stp;                       // [Pa]
  const Real mw_h2so4_gmol = 1000 * Constants::molec_weight_h2so4; // [g/mol]
  const Real mw_air_gmol = 1000 * Constants::molec_weight_dry_air; // [g/mol]
  const Real vol_molar_h2so4 = Constants::molec_diffusion_h2so4;   // [-]
  const Real vol_molar_air = Constants::molec_diffusion_dry_air;   // [-]
  const Real accom_coef_h2so4 = Constants::accom_coef_h2so4;       // [-]
  const Real r_universal_mJ = 1000 * Constants::r_gas; // [mJ/(K mol)]
  const Real r_pi = Constants::pi;

  const Real beta_inp = 0; // quadrature parameter (--)
  //===============================================================
  // Calculate the reference uptake coefficient for all
  // aerosol modes using properties of the H2SO4 gas
  //===============================================================
  if (l_calc_gas_uptake_coeff) {
    // do calcullation for ALL modes
    const bool l_condense_to_mode[num_mode] = {true, true, true, true};
    // initialize with zero (-> no uptake)
    Real uptkaer_ref[num_mode] = {0, 0, 0, 0};
    gasaerexch::gas_aer_uptkrates_1box1gas(
        l_condense_to_mode, temp, pmid, pstd, mw_h2so4_gmol, mw_air_gmol,
        vol_molar_h2so4, vol_molar_air, accom_coef_h2so4, r_universal_mJ, r_pi,
        beta_inp, nghq, dgn_awet, alnsg_aer, uptkaer_ref);

    // -------------------------------------------------------------
    // Unit conversion: uptkrate is for number = 1 #/m3, so mult. by
    // number conc. (#/m3)
    //--------------------------------------------------------------
    for (int imode = 0; imode < num_mode; ++imode) {
      if (l_condense_to_mode[imode]) {
        uptkaer_ref[imode] *= qnum_cur[imode] * aircon;
      }
    }
    //===============================================================
    // Assign uptake rate to each gas species and each mode using the
    // ref. value uptkaer_ref calculated above and the uptake rate
    // factor specified as constants at the beginning of the module
    //===============================================================
    // gas to aerosol mass transfer rate (1/s)
    for (int igas = 0; igas < num_gas; ++igas)
      for (int imode = 0; imode < num_mode; ++imode)
        uptkaer[igas][imode] = 0.0; // default is no uptake

    for (int igas = 0; igas < num_gas; ++igas) {
      for (int imode = 0; imode < num_mode; ++imode)
        if (l_gas_condense_to_mode[igas][imode])
          uptkaer[igas][imode] = uptkaer_ref[imode] * uptk_rate_factor[igas];
    }

    // total uptake rate (sum of all aerosol modes) for h2so4.
    // Diagnosd for calling routine. Not used in this subroutne.
    uptkrate_h2so4 = 0;
    for (int n = 0; n < num_mode; ++n)
      uptkrate_h2so4 += uptkaer[igas_h2so4][n];
  }
  // =============================================================
  //  Solve condensation equation for non-volatile species
  // =============================================================
  //  Using quasi-analytical solution (with no time sub-stepping)
  // -------------------------------------------------------------
  for (GasId gas : GasAerExch::Gases()) {
    const int igas = static_cast<int>(gas);
    const AeroId aer = gas_to_aer[igas];
    if (aer != AeroId::None &&
        eqn_and_numerics_category[igas] == GasAerExch::ANAL) {
      const int iaer = static_cast<int>(aer);
      const Real netprod = qgas_netprod_otrproc[igas];
      Real uptkaer_igas[num_mode] = {};
      Real qaer[num_mode] = {};
      for (ModeIndex mode : GasAerExch::Modes()) {
        const int n = static_cast<int>(mode);
        uptkaer_igas[n] = uptkaer[igas][n];
        if (mode_contains_species(mode, aer))
          qaer[n] = qaer_cur[iaer][n];
      }
      gasaerexch::mam_gasaerexch_1subarea_1gas_nonvolatile(
          dt, netprod, uptkaer_igas, qgas_cur[igas], qgas_avg[igas], qaer);

      for (ModeIndex mode : GasAerExch::Modes()) {
        const int n = static_cast<int>(mode);
        // Commit a small crime there.  There is a case where
        // mode_contains_species(mode, aer) is false and yet
        // a value is stored in qaer_cur at that point.
        // The e3sm_mam4 code also did this as a place to
        // store a temperary value.
        qaer_cur[iaer][n] = qaer[n];
      }
    }
  }
  //---------------------------------------------------------------n
  // Clip condensation rate when nh3 is on the list of non-volatile gases:
  // limit the condensation of nh3 so that nh4 does not exceed
  // aer_nh4_so4_molar_ratio_max * so4 (molar basis).
  // (Hui Wan's comment from Dec. 2020: chose to leave the following block here
  // instead of moving it to the subroutine
  // mam_gasaerexch_1subarea_nonvolatile_quasi_analytical, to make it a bit
  // easier to see the assumed relationship between species.)
  //---------------------------------------------------------------------
  if (igas_nh3) {
    // We know that igas_nh3 is the third entry in 1Gn
    const int igas = GasAerExch::igas_nh3_index;
    const AeroId aer = gas_to_aer[igas];
    if (aer != AeroId::None) {
      for (ModeIndex mode : GasAerExch::Modes()) {
        const int n = static_cast<int>(mode);
        const Real aer_nh4_so4_molar_ratio_max = 2;
        if (0 < uptkaer[igas][n]) {
          const int iaer = static_cast<int>(aer);
          const int iaer_so4 = static_cast<int>(AeroId::SO4);
          if (mode_contains_species(mode, aer) &&
              mode_contains_species(mode, AeroId::SO4)) {
            // if nh4 exceeds aer_nh4_so4_molar_ratio_max*so4 (molar basis),
            // put the excessive amount back to gas phase (nh3)
            const Real mass_excess =
                qaer_cur[iaer][n] -
                aer_nh4_so4_molar_ratio_max * qaer_cur[iaer_so4][n];
            if (0 < mass_excess) {
              qaer_cur[iaer][n] -= mass_excess;
              qgas_cur[igas] += mass_excess;
              qgas_avg[igas] += mass_excess * 0.5;
            }
          }
        }
      }
    }
  }
  //============================================
  // Solve condensation equations for SOA
  //============================================
  // starting index of POA on the species list
  // ending index of POA on the species list
  // For now we are only going to do a single volitle species.
  const int npoa = 1;
  Real qaer_poa[npoa][num_mode] = {};
  for (ModeIndex mode : GasAerExch::Modes()) {
    const int idxs =
        aerosol_index_for_mode(ModeIndex::Accumulation, AeroId::POM);
    const int n = static_cast<int>(mode);
    qaer_poa[0][n] = haero::max(qaer_cur[idxs][n], 0);
  }
  Real soa_out = 0;
  int niter = 0;

  const int ntot_soaspec = 1;
  const GasId soaspec[ntot_soaspec] = {GasId::SOAG};
  mam_soaexch_1subarea(GasAerExch::npca, ntot_soamode, ntot_soaspec, soaspec,
                       dt, dtsub_soa_fixed, pstd, r_universal_mJ, temp, pmid,
                       uptkaer, qaer_poa, qgas_cur, qgas_avg, qaer_cur, niter,
                       soa_out);
  niter_out = niter;
  g0_soa_out = soa_out;
}

KOKKOS_INLINE_FUNCTION
void gas_aerosol_uptake_rates_1box(
    const int k, const AeroConfig &aero_config, const Real dt,
    const Atmosphere &atm, const Prognostics &progs, const Diagnostics &diags,
    const Tendencies &tends, const GasAerExch::Config &config,
    const bool l_gas_condense_to_mode[GasAerExch::num_gas]
                                     [GasAerExch::num_mode],
    const int eqn_and_numerics_category[GasAerExch::num_gas],
    const Real uptk_rate_factor[GasAerExch::num_gas],
    const Real alnsg_aer[GasAerExch::num_mode]) {

  const Real r_universal = Constants::r_gas; // [J/(K mol)]
  const int num_gas = GasAerExch::num_gas;
  const int num_aer = AeroConfig::num_aerosol_ids();
  const int num_mode = GasAerExch::num_mode;
  const int igas_h2so4 = static_cast<int>(GasId::H2SO4);

  const bool igas_nh3 = config.igas_nh3;
  AeroId gas_to_aer[num_gas] = {};
  for (GasId gas : GasAerExch::Gases())
    gas_to_aer[static_cast<int>(gas)] = GasAerExch::gas_to_aer(gas);

  Real qgas_netprod_otrproc[num_gas];
  for (int i = 0; i < num_gas; ++i)
    qgas_netprod_otrproc[i] = config.qgas_netprod_otrproc[i];

  const int iaer_so4 = GasAerExch::iaer_so4;
  const int iaer_pom = GasAerExch::iaer_pom;
  const bool l_calc_gas_uptake_coeff = config.calculate_gas_uptake_coefficient;
  const Real dtsub_soa_fixed = config.dtsub_soa_fixed;
  const Real &temp = atm.temperature(k);
  const Real &pmid = atm.pressure(k);
  const Real aircon_kmol = pmid / (1000 * r_universal * temp);
  const int ngas = GasAerExch::num_gas_to_aer;

  // set number of ghq points for direct ghq
  const int nghq = 2; // aero_config.number_gauss_points_for_integration;

  // extract gas mixing ratios
  Real qgas_cur[num_gas], qgas_avg[num_gas], qaer_cur[num_aer][num_mode];
  for (int g = 0; g < num_gas; ++g) {
    qgas_cur[g] = progs.q_gas[g](k);
    qgas_avg[g] = 0;
  }
  for (int n = 0; n < num_mode; ++n)
    for (int g = 0; g < num_aer; ++g)
      qaer_cur[g][n] = progs.q_aero_i[n][g](k);

  Real qnum_cur[num_mode];
  for (int i = 0; i < num_mode; ++i) {
    qnum_cur[i] = progs.n_mode_i[i](k);
  }

  Real qgas_sv1[num_gas], qnum_sv1[num_mode], qaer_sv1[num_aer][num_mode];
  for (int i = 0; i < num_gas; ++i) {
    qgas_sv1[i] = qgas_cur[i];
  }
  for (int i = 0; i < num_mode; ++i) {
    qnum_sv1[i] = qnum_cur[i];
  }
  for (int i = 0; i < num_aer; ++i) {
    for (int j = 0; j < num_mode; ++j) {
      qaer_sv1[i][j] = qaer_cur[i][j];
    }
  }

  Real dgn_awet[num_mode] = {};
  for (int i = 0; i < num_mode; ++i)
    dgn_awet[i] = diags.wet_geometric_mean_diameter_i[i](k);

  // gas to aerosol mass transfer rate (1/s)
  Real uptkaer[num_gas][num_mode];
  for (int igas = 0; igas < num_gas; ++igas)
    for (int imode = 0; imode < num_mode; ++imode)
      uptkaer[igas][imode] = progs.uptkaer[igas][imode](k);

  Real uptkrate_h2so4 = diags.uptkrate_h2so4(k);
  int niter_out = 0;
  Real g0_soa_out = 0;
  const int ntot_soamode = config.ntot_soamode;

  mam_gasaerexch_1subarea(nghq, igas_h2so4, igas_nh3, ntot_soamode, gas_to_aer,
                          iaer_so4, iaer_pom, l_calc_gas_uptake_coeff,
                          l_gas_condense_to_mode, eqn_and_numerics_category, dt,
                          dtsub_soa_fixed, temp, pmid, aircon_kmol, ngas,
                          qgas_cur, qgas_avg, qgas_netprod_otrproc, qaer_cur,
                          qnum_cur, dgn_awet, alnsg_aer, uptk_rate_factor,
                          uptkaer, uptkrate_h2so4, niter_out, g0_soa_out);

  for (int i = 0; i < num_mode; ++i)
    tends.n_mode_i[i](k) = (qnum_cur[i] - qnum_sv1[i]) / dt;

  for (int n = 0; n < num_mode; ++n)
    for (int g = 0; g < num_aer; ++g)
      tends.q_aero_i[n][g](k) = (qaer_cur[g][n] - qaer_sv1[g][n]) / dt;

  for (int g = 0; g < num_gas; ++g)
    tends.q_gas[g](k) +=
        (qgas_cur[g] - (qgas_sv1[g] + qgas_netprod_otrproc[g] * dt)) / dt;

  for (int g = 0; g < num_gas; ++g) {
    progs.q_gas[g](k) = qgas_cur[g];
    progs.q_gas_avg[g](k) = qgas_avg[g];
  }
  for (int n = 0; n < num_mode; ++n)
    for (int g = 0; g < num_aer; ++g)
      progs.q_aero_i[n][g](k) = qaer_cur[g][n];

  for (int igas = 0; igas < num_gas; ++igas)
    for (int imode = 0; imode < num_mode; ++imode)
      progs.uptkaer[igas][imode](k) = uptkaer[igas][imode];

  diags.g0_soa_out(k) = g0_soa_out;
  diags.uptkrate_h2so4(k) = uptkrate_h2so4;
  diags.num_substeps(k) = niter_out;
}

} // namespace gasaerexch

// init -- initializes the implementation with MAM4's configuration
inline void GasAerExch::init(const AeroConfig &aero_config,
                             const Config &process_config) {

  config_ = process_config;

  for (int imode = 0; imode < num_mode; ++imode)
    modes_mean_std_dev[imode] = modes(imode).mean_std_dev;

  //-------------------------------------------------------------------
  // MAM currently uses a splitting method to deal with gas-aerosol
  // mass exchange. A quasi-analytical solution assuming timestep-wise
  // constant uptake rate is applied to nonvolatile species, while
  // an implicit time stepping method with adaptive step size
  // is applied to gas-phase SOA species which are assumed semi-volatile.
  // There are two different subroutines in this modules to deal
  // with the two categories of cases. The array
  // which category.
  //-------------------------------------------------------------------
  for (int k = 0; k < num_gas; ++k)
    eqn_and_numerics_category[k] = NA;
  eqn_and_numerics_category[igas_soag] = IMPL;
  eqn_and_numerics_category[igas_h2so4] = ANAL;
  eqn_and_numerics_category[igas_nh3] = ANAL;

  //-------------------------------------------------------------------
  // Determine whether specific gases will condense to specific modes
  //-------------------------------------------------------------------
  for (int igas = 0; igas < num_gas; ++igas)
    for (int imode = 0; imode < num_mode; ++imode)
      l_gas_condense_to_mode[igas][imode] = false;
  // loop through all registered gas species
  for (GasId gas : GasAerExch::Gases()) {
    const AeroId aero_id = GasAerExch::gas_to_aer(gas);
    // can this gas species condense?
    if (aero_id != AeroId::None) {
      const int igas = static_cast<int>(gas);
      if (eqn_and_numerics_category[igas] != NA) {
        // what aerosol species does the gas become when condensing?
        for (ModeIndex mode_index : GasAerExch::Modes()) {
          const bool mode_contains_species =
              mam4::mode_contains_species(mode_index, aero_id);
          const int imode = static_cast<int>(mode_index);
          l_gas_condense_to_mode[igas][imode] =
              mode_contains_species || config_.l_mode_can_age[imode];
        }
      }
    }
  }
}

// compute_tendencies -- computes tendencies and updates diagnostics
// NOTE: that both diags and tends are const below--this means their views
// NOTE: are fixed, but the data in those views is allowed to vary.
KOKKOS_INLINE_FUNCTION
void GasAerExch::compute_tendencies(const AeroConfig &config,
                                    const ThreadTeam &team, Real t, Real dt,
                                    const Atmosphere &atm,
                                    const Prognostics &progs,
                                    const Diagnostics &diags,
                                    const Tendencies &tends) const {
  // const int nghq = 2;  // set number of ghq points for direct ghq
  const int nk = atm.num_levels();
  Real alnsg_aer[num_mode];
  for (int k = 0; k < num_mode; ++k)
    alnsg_aer[k] = std::log(modes_mean_std_dev[k]);

  Real uptk_rate[num_gas];
  for (int k = 0; k < num_gas; ++k)
    uptk_rate[k] = GasAerExch::uptk_rate_factor(k);

  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team, nk), KOKKOS_CLASS_LAMBDA(int k) {
        gasaerexch::gas_aerosol_uptake_rates_1box(
            k, config, dt, atm, progs, diags, tends, config_,
            l_gas_condense_to_mode, eqn_and_numerics_category, uptk_rate,
            alnsg_aer);
      });
}
} // namespace mam4

#endif
