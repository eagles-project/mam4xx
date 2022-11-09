#ifndef MAM4XX_GASAEREXCH_SOAEXCH_HPP
#define MAM4XX_GASAEREXCH_SOAEXCH_HPP

#include "mam4xx/aero_config.hpp"
#include <mam4xx/aero_modes.hpp>

#include <haero/haero.hpp>
#include <haero/math.hpp>

namespace mam4 {
namespace gasaerexch {

using Real = haero::Real;

// ==============================================================================
// Calculate SOA species's eiquilibrium vapor mixing ratio under the ambient
// condition, ignoring the solute effect.
//
// History:
// - MAM box model by R. C. Easter, PNNL.
// - Subroutine cretated duing code refactoring by Hui Wan, PNNL, 2022.
// ==============================================================================
KOKKOS_INLINE_FUNCTION
void soa_equilib_mixing_ratio_no_solute(const Real &T_in_K,     // in
                                        const Real &p_in_Pa,    // in
                                        const Real &pstd_in_Pa, // in
                                        const Real r_universal, // in
                                        Real &g0_soa) {         // out

  // ntot_soaspec     number of SOA species, which is assumed 1 for this
  // function. T_in_K           temperature in Kelvin p_in_Pa          air
  // pressure in Pascal pstd_in_Pa       standard air pressure in Pascal
  // r_universal      universal gas constant in J/K/mol
  // g0_soa(ntot_soaspec)   ambient soa gas equilib mixing ratio (mol/mol at
  // actual mw)

  // Parameters: should they be changed to intent(in)? - Not ncessary for now,
  // but perhaps need to change to intent in when we get multiple SOA species.

  // delh_vap_soa = heat of vaporization for gas soa (J/mol)
  const Real delh_vap_soa = 156.0e3;
  // p0_soa_298 = soa gas equilib vapor presssure (atm) at 298 K
  const Real p0_soa_at_298K = 1.0e-10;

  //
  // Calculate equilibrium mixing ratio under the ambient condition
  //

  // The equilibrium vapor pressure
  const Real p0_soa =
      p0_soa_at_298K * exp(-(delh_vap_soa / (r_universal / 1.e3)) *
                           ((1.0 / T_in_K) - (1.0 / 298.0)));

  // Convert to mixing ratio in mol/mol
  g0_soa = pstd_in_Pa * p0_soa / p_in_Pa;
}

//================================================================================
// Determine adaptive time-step size for SOA condensation/evaporation
//================================================================================
KOKKOS_INLINE_FUNCTION
Real soa_exch_substepsize(
    const bool skip_soamode[1], // in   true if this mode does not have soa
    const Real uptkaer_soag_tmp[1][AeroConfig::num_modes()], // in
    // evolving SOA aerosol mixrat (mol/mol at actual mw), part of the unknowns
    // of the ODEs
    const Real a_soa[1][AeroConfig::num_modes()], // in  soa aerosol mixrat
                                                  // (mol/mol at actual mw)
    const Real a_opoa[AeroConfig::num_modes()],   // in  oxidized-poa aerosol
                                                // mixrat (mol/mol at actual mw)
    const Real g_soa[1],  // in  evolving SOA gas mixrat (mol/mol at actual mw),
                          // part of the unknowns of the ODEs
    const Real g0_soa[1], // in  ambient soa gas equilib mixrat (mol/mol at
                          // actual mw), len ntot_soaspec=1
    const Real alpha_astem, // in  error control parameter for sub-timesteps
    const Real dt_full,     // in  host model dt (s)
    Real &t_cur)            // inout urrent time (s) since last full time step
{
  static constexpr int ntot_soaspec = 1; //  "last" gas species that can be SOA
  static constexpr int ntot_soamode =
      1; //  "last" mode on which soa is allowed to condense

  const Real eps_aer = 1.0e-20; // epsilon to be used on denominator for
                                // avoiding division by zero
  const Real eps_gas = 1.0e-20; // epsilon to be used on denominator for
                                // avoiding division by zero

  //------------------------------------------------------------------
  // Oxygenated OA in each aerosol mode, summed over all SOA species
  //------------------------------------------------------------------
  Real a_ooa_sum_tmp[ntot_soamode]; // total ooa (=soa+opoa) in a mode
  for (int n = 0; n < ntot_soamode; ++n) {
    if (!skip_soamode[n]) {
      a_ooa_sum_tmp[n] = a_opoa[n];
      for (int i = 0; i < ntot_soaspec; ++i)
        a_ooa_sum_tmp[n] += a_soa[i][n];
    }
  }
  //----------------------------------------------------------------------------------------------------------
  // For each SOA species, estimate the normalized total mass exchange rate by
  // using a simple Euler forward scheme.  Here the normalized mass exchange
  // rate is defined as
  //   condensed or evaporated SOA /dt / max( available-SOA-gas,
  //   equilibrium-SOA-gas-mixing-ratio )
  // We first calculate this rate for each SOA species and each aerosol mode.
  // Then, for each species, we calculate the sum of its absolute value over all
  // modes, and saved the result in the array tot_frac_single_soa_species.
  //----------------------------------------------------------------------------------------------------------
  Real tot_frac_single_soa_species[ntot_soaspec];
  for (int i = 0; i < ntot_soaspec; ++i)
    tot_frac_single_soa_species[i] = 0.0;

  for (int ll = 0; ll < ntot_soaspec; ++ll) {
    for (int n = 0; n < ntot_soamode; ++n) {
      if (!skip_soamode[n]) {
        // Calculate g_star, the equilibrium SOA mixing ratio after taking into
        // account the solute effect

        // sat(ll,n) = g0_soa(ll)/a_ooa_sum_tmp(n) = g_star(ll,n)/a_soa(ll,n)
        // This this the pre-factor of SOA (aerosol) mixing ratio in the
        // expression of the equilibrium SOA mixing ratio -- it is not a
        // saturation rato!
        const Real sat = g0_soa[ll] / haero::max(a_ooa_sum_tmp[n], eps_aer);

        // g_star -  equilibrium SOA gas mixing ratio (mol/mol at actual mw),
        // with solute effect accounted for
        const Real g_star = sat * a_soa[ll][n];

        // Calculate the mass exchange rate (normalized by the larger between
        // g_soa and g_star) of this aerosol mode by using an Euler forward
        // scheme, then add this fraction to the total fraction (variable
        // tot_frac_single_soa_species) summed over all aerosol modes.

        const Real phi = (g_soa[ll] - g_star) /
                         haero::max(g_soa[ll], haero::max(g_star, eps_gas));
        tot_frac_single_soa_species[ll] +=
            uptkaer_soag_tmp[ll][n] * haero::abs(phi);
      }
    }
  }

  //------------------------------------------------------------------------------------------------------
  // Now, get the maximum of the normalized exchange rate of all SOA species
  //------------------------------------------------------------------------------------------------------
  Real max_frac_all_soa_species = -999999;
  for (int i = 0; i < ntot_soaspec; ++i)
    max_frac_all_soa_species =
        haero::max(max_frac_all_soa_species, tot_frac_single_soa_species[i]);

  //------------------------------------------------------------------------------------------------------
  // Determine the adaptive step size, dt_cur, requiring that the condensed or
  // evaporated amount of any SOA species cannot exceed a fraction (alpha_astem)
  // of
  //   max( available-SOA-gas, equilibrium-SOA-gas-mixing-ratio ).
  //------------------------------------------------------------------------------------------------------
  const Real dt_max =
      dt_full - t_cur; // how far we are from the end of the full time step

  Real dt_cur = 0; // out current integration timestep (s)
  if (dt_max * max_frac_all_soa_species <= alpha_astem) {
    // Here alpha_astem/max_frac_all_soa_species >= dt_max, so this is the final
    // substep. The substep length is limited by the length of full dt.
    dt_cur = dt_max;
    t_cur = dt_full;
  } else {
    // Using dt_max would lead to more than alpha_astem (fraction) of mass
    // exchange. Limit the step size to avoid exceeding alpha_astem.
    dt_cur = alpha_astem / max_frac_all_soa_species;
    t_cur += dt_cur;
  }
  return dt_cur;
}
//===============================================================================================

//===============================================================================================
// Time integration for the ODE set that describes the condensation/evaporation
// of SOA
//-----------------------------------------------------------------------------------------------
// History:
// - Original version from the MAM box model by R. C. Easter, PNNL.
// - Refactoring by Hui Wan and Qiyang Yan, PNNL, 2022.
//===============================================================================================
KOKKOS_INLINE_FUNCTION
void mam_soaexch_advance_in_time(
    const Real dt_full,      // in host model dt (s)
    const Real dt_sub_fixed, // in fixed sub-step. A negative value means using
                             // adaptive step sizes
    const int niter_max,     // in maximum number of substeps
    const Real alpha_astem,  // in error control parameter for sub-timesteps
    const Real uptkaer[AeroConfig::num_gas_ids()]
                      [AeroConfig::num_modes()], // in  uptake rate coefficient
    const Real g0_soa[1], // in ambient soa gas equilib mixrat (mol/mol at
                          // actual mw), len ntot_soaspec=1
    Real qgas_cur[AeroConfig::num_gas_ids()],   // inout
    const Real a_opoa[AeroConfig::num_modes()], // in oxidized-poa aerosol
                                                // mixrat (mol/mol at actual mw)
    Real qaer_cur[AeroConfig::num_aerosol_ids()]
                 [AeroConfig::num_modes()],   // inout
    Real qgas_avg[AeroConfig::num_gas_ids()], // inout
    int &niter)                               // out total number of sub-steps
{
  using haero::max;
  static constexpr int ntot_soamode = 1;
  static constexpr int ntot_soaspec = 1; //  "last" gas species that can be SOA
  static constexpr int max_gas = AeroConfig::num_gas_ids();
  static constexpr int max_mode = AeroConfig::num_modes();

  // The local arrays a_soa and g_soa declared below have the same meaning of
  // qaer_cur and qgas_cur.
  // - At the beginning of each substep, a_soa and g_soa get their values from
  // qaer_cur and qgas_cur but
  //   with negative values set to zero.
  //- At the end of each substep, the values of a_soa and g_soa are copied back
  // to qaer_cur and qgas_cur.

  Real a_soa[ntot_soaspec]
            [max_mode]; // evolving SOA aerosol mixrat (mol/mol at actual mw),
                        // part of the unknowns of the ODEs
  Real g_soa[ntot_soaspec]; // evolving SOA gas     mixrat (mol/mol at actual
                            // mw), part of the unknowns of the ODEs

  Real qgas_prv[max_gas];     // gas mixrat at beginning of substep, saved for
                              // calculating time average of qgas
  Real qgas_avg_sum[max_gas]; // qgas*dt integrated over a time step, used for
                              // calculating time average of qgas

  bool skip_soamode[ntot_soamode]; // true if this mode does not have soa
  Real uptkaer_soag_tmp[ntot_soaspec]
                       [max_mode];   // uptake rate of different modes for
                                     // different soa species
  Real beta[ntot_soaspec][max_mode]; // dt_cur * uptake-rate-coefficient

  const Real eps_aer = 1.0e-20; // epsilon to be used on denominator for
                                // avoiding division by zero
  const Real eps_dt = 1.0e-3;

  // variable name "sat": sat(m,ll) = g0_soa(ll)/a_ooa_sum(m) =
  // g_star(m,ll)/a_soa(m,ll) used by the numerical integration scheme -- it is
  // not a saturation rato!

  Real sat_hybrid[ntot_soaspec][max_mode];

  Real tot_soa[ntot_soaspec]; // g_soa + sum( a_soa(:) )

  // ----------------------------------------------------------------------
  //  Determine which modes have non-zero transfer rates and are hence
  //  involved in the subsequent calculations of soa gas-aerosol transfer.
  //  (For diameter = 1 nm and number = 1 #/cm3, xferrate ~= 1e-9 s-1)
  // ----------------------------------------------------------------------
  for (int n = 0; n < ntot_soamode; ++n) {
    skip_soamode[n] = true;
    for (int ll = 0; ll < ntot_soaspec; ++ll) {
      if (uptkaer[ll][n] > 1.0e-15) {
        uptkaer_soag_tmp[ll][n] = uptkaer[ll][n];
        skip_soamode[n] = false;
      } else {
        uptkaer_soag_tmp[ll][n] = 0.0;
      }
    }
  }
  // -------------------------------------
  //  Time loop for SOA sub-stepping
  // -------------------------------------
  niter = 0;
  Real tcur = 0.0; // time elapsed since the beginning of sub-cycling (last full
                   // timestep)

  Real dtsum_qgas_avg = 0.0; // total dt of all substeps, used for calculating
                             // time average of qgas
  for (int i = 0; i < ntot_soaspec; ++i)
    qgas_avg_sum[i] = 0.0;

  while (tcur < dt_full - eps_dt) {
    ++niter;
    if (niter > niter_max)
      break;

    // save gas mixing ratios at the beginning of substep.
    // This is used at the end of the substep to calculate a time average

    for (int i = 0; i < ntot_soaspec; ++i)
      qgas_prv[i] = qgas_cur[i];

    // ------------------------------------------------------------------------------
    //  SOA gas, aerosol, and total: get the current (old) values
    // ------------------------------------------------------------------------------
    //  Load incoming SOA gas into temporary array, force values to be
    //  non-negative

    for (int i = 0; i < ntot_soaspec; ++i)
      g_soa[i] = max(qgas_cur[i], 0.0);

    //  Load incoming SOA aerosols into temporary array, force values to be
    //  non-negative

    for (int n = 0; n < ntot_soamode; ++n) {
      if (!skip_soamode[n]) {
        for (int i = 0; i < ntot_soaspec; ++i)
          a_soa[i][n] = max(qaer_cur[i][n], 0.0);
      }
    }

    //  Calculate the total amount of each SOA species, defined as mixing ratio
    //  of gas + aerosol, saved in array tot_soa.

    for (int ll = 0; ll < ntot_soaspec; ++ll) {
      tot_soa[ll] = g_soa[ll];
      for (int n = 0; n < ntot_soamode; ++n) {
        if (!skip_soamode[n])
          tot_soa[ll] = tot_soa[ll] + a_soa[ll][n];
      }
    }

    // -----------------------------------------------------------------------------
    //  Determine step size: either fixed or adaptive
    // -----------------------------------------------------------------------------
    Real dt_cur = 0;          // substep length, fixed or adaptive
    if (dt_sub_fixed > 0.0) { //  Use prescribed, fixed step size
      dt_cur = dt_sub_fixed;
      tcur += dt_cur;
    } else { //  Chose an adaptive step size
      dt_cur = soa_exch_substepsize(skip_soamode, uptkaer_soag_tmp, a_soa,
                                    a_opoa, g_soa, g0_soa, alpha_astem, dt_full,
                                    tcur); // inout
    }

    // ----------------------------------------------------------------------------
    //  Now that we have determined the sub-step size dt_cur,
    //  define a variable beta = dt_cur * uptake-rate-coefficient.
    //  This is used at a few places in the semi-implicit time integration
    //  scheme.
    // ----------------------------------------------------------------------------
    for (int n = 0; n < ntot_soamode; ++n) {
      if (!skip_soamode[n]) {
        for (int ll = 0; ll < ntot_soaspec; ++ll)
          beta[ll][n] = dt_cur * uptkaer_soag_tmp[ll][n];
      }
    }

    // ------------------------------------------------------------------------------------------
    //  Linearize the ODE of each SOA species in each mode
    // ------------------------------------------------------------------------------------------
    //  Because the equilibrium SOA mixing ratio (that takes into account the
    //  solvent effect) depends on the SOA (aerosol) mixing ratio in each mode,
    //  the time evolution equations for SOAs (aerosol) are nonlinear.
    //  To numerically solve these nonlinear equations using a
    //  semi-implicit-in-time scheme, we need to linearize the equations. The
    //  nonlinear pre-factor in front of an SOA mixing ratio on the RHS of the
    //  SOA mixing ratio equation is
    //      uptake-rate-coefficient * g0_soa / mixing-ratio-of-OOA
    //  Let us denote
    //      sat = g0_soa / mixing-ratio-of-OOA
    //  The code block below provides an approximate value of sat (saved in the
    //  array sat_hybrid) to be used in the semi-implicit solve further down
    //  below. We refer to this "sat" variable as a "hybrid" one because the
    //  calculation below uses different expressions for SOA condensation and
    //  evaporation. The difference is in the SOA (aerosol) mixing ratios used
    //  for calculating the OOA mixing ratio in the denominaotor of sat.
    // ------------------------------------------------------------------------------------------
    Real a_soa_hybrid[ntot_soaspec]; // temporary SOA aerosol mixrat (mol/mol)
                                     // used for linearization
    for (int n = 0; n < ntot_soamode; ++n) {
      if (skip_soamode[n])
        continue;

      for (int ll = 0; ll < ntot_soaspec; ++ll) {

        //  First get an estimate of the equilibrium SOA mixing ratio (variable
        //  g_star_old) using the old SOA mixing ratio (variable a_soa)

        Real a_ooa_sum_old =
            a_opoa[n]; //// total ooa (=soa+opoa) in a mode,
                       /// calculated using old SOA mixing ratio
        for (int i = 0; i < ntot_soaspec; ++i)
          a_ooa_sum_old += a_soa[i][n];
        const Real sat_old = g0_soa[ll] / max(a_ooa_sum_old, eps_aer);

        const Real g_star_old =
            sat_old * a_soa[ll][n]; // soa gas mixrat that is in equilib with
                                    // each aerosol mode (mol/mol) diagnosed
                                    // using old gas and aerosol mixing ratios

        //  Using g_star_old and the current (old) g_soa to determine whether we
        //  have
        //   - supersaturation (meaning SOA will be condensing) or
        //   - undersaturation (meaning SOA will be evaporating)

        const Real g_soa_supersat =
            g_soa[ll] - g_star_old; // SOA gas supersaturation mixrat (mol/mol
                                    // at actual mw). < 0 means unsaturated

        if (g_soa_supersat > 0.0) {
          //  For modes where SOA is condensing, estimate an approximate "new"
          //  a_soa(ll,n) using the Euler forward scheme in which both the SOA
          //  gas mixing ratio and equilibrium mixing ratio are set to their
          //  "old" values, i.e., the current g_soa and the above-calculated
          //  g_star_old. Do this to get better estimate of "new" a_soa(ll,n)
          //  and g_star(ll,n)

          a_soa_hybrid[ll] = a_soa[ll][n] + beta[ll][n] * g_soa_supersat;

        } else {
          //  For modes where SOA is evaporating, simply use the "old" SOA
          //  mixing ratio
          a_soa_hybrid[ll] = a_soa[ll][n];
        }
      }

      //  Now, calculate the total OOA in the mode using the a_soa_hybrid
      //  calculated just now

      Real a_ooa_sum_hybrid =
          a_opoa[n]; // total ooa (=soa+opoa) in a mode, different for
                     // condensation/evaporation cases
      for (int i = 0; i < ntot_soaspec; ++i)
        a_ooa_sum_hybrid += a_soa_hybrid[i];

      //  With the newly calculated a_ooa_sum_hybrid, we can now calculate
      //  the pre-factor in front of the SOA mixing ratio on the RHS of each SOA
      //  equation, (i.e., variable sat_hybrid).

      for (int ll = 0; ll < ntot_soaspec; ++ll)
        sat_hybrid[ll][n] = g0_soa[ll] / max(a_ooa_sum_hybrid, eps_aer);
    }
    // ------------------------------------------------------------------------------------------
    //  Implicit solve for the linearize equations
    // ------------------------------------------------------------------------------------------
    for (int ll = 0; ll < ntot_soaspec; ++ll) {
      Real tmpa = 0.0;
      Real tmpb = 0.0;
      for (int n = 0; n < ntot_soamode; ++n) {
        if (!skip_soamode[n]) {
          tmpa += a_soa[ll][n] / (1.0 + beta[ll][n] * sat_hybrid[ll][n]);
          tmpb += beta[ll][n] / (1.0 + beta[ll][n] * sat_hybrid[ll][n]);
        }
      }
      g_soa[ll] = (tot_soa[ll] - tmpa) / (1.0 + tmpb);
      g_soa[ll] = max(0.0, g_soa[ll]);
      for (int n = 0; n < ntot_soamode; ++n) {
        if (!skip_soamode[n]) {
          a_soa[ll][n] = (a_soa[ll][n] + beta[ll][n] * g_soa[ll]) /
                         (1.0 + beta[ll][n] * sat_hybrid[ll][n]);
        }
      }
    }

    // ------------------------------------------------------------------------------------------
    //  Save mix ratios for soa species
    // ------------------------------------------------------------------------------------------
    for (int igas = 0; igas < ntot_soaspec; ++igas) {
      for (int n = 0; n < ntot_soamode; ++n)
        qaer_cur[igas][n] = a_soa[igas][n];
    }

    // ------------------------------------------------------------------------------------------
    //  Save mixing ratios for SOA gas species; diagnose time average
    // ------------------------------------------------------------------------------------------
    for (int igas = 0; igas < ntot_soaspec; ++igas) {
      qgas_cur[igas] = g_soa[igas]; //  new gas mixing ratio
      const Real tmpc = qgas_cur[igas] -
                        qgas_prv[igas]; //  amount of condensation/evaporation
      qgas_avg_sum[igas] =
          qgas_avg_sum[igas] + dt_cur * (qgas_prv[igas] + 0.5 * tmpc);
    }
    dtsum_qgas_avg = dtsum_qgas_avg + dt_cur;
  }

  // -------------------------------------------------------------------
  //  Convert qgas_avg from sum_over[ qgas*dt_cur ] to an average
  // -------------------------------------------------------------------
  for (int i = 0; i < ntot_soaspec; ++i)
    qgas_avg[i] = max(0.0, qgas_avg_sum[i] / dtsum_qgas_avg);
}

KOKKOS_INLINE_FUNCTION
void mam_soaexch_1subarea(
    const int mode_pca,          // in = mam4::ModeIndex::PrimaryCarbon
    const Real dt,               // in time step size used by parent subroutine
    const Real dt_sub_soa_fixed, // in fixed sub-step in s. A negative value
                                 // means using adaptive step sizes
    const Real pstd,             // in standard atmosphere in Pa
    const Real r_universal,      // in universal gas constant in J/K/mol
    const Real &temp,            // in temperature (K)
    const Real &pmid,            // in pressure at model levels (Pa)
    const Real uptkaer[AeroConfig::num_gas_ids()]
                      [AeroConfig::num_modes()],     // in
    const Real qaer_poa[1][AeroConfig::num_modes()], // in POA mixing ratio
                                                     // (mol/mol at actual mw)
    Real qgas_cur[AeroConfig::num_gas_ids()],        // inout
    Real qgas_avg[AeroConfig::num_gas_ids()],        // inout
    Real qaer_cur[AeroConfig::num_aerosol_ids()]
                 [AeroConfig::num_modes()], // inout
    int &niter,                             // out
    Real &g0_soa) { // out ambient soa gas equilib mixrat (mol/mol at actual mw)

  // ntot_poaspec is the number of reacting gas species. The code only supports
  // one but MAM4 had an initial support for more so it was decided to keep the
  // form of the multi-species code if not the function.
  static constexpr int ntot_soamode = 1;
  static constexpr int ntot_poaspec = 1;
  static constexpr int num_mode = AeroConfig::num_modes();
  static constexpr bool flag_pcarbon_opoa_frac_zero =
      true; // for backward compatibility
  static constexpr Real alpha_astem =
      0.05; // parameter used in calc of sub-step sizes
  static constexpr Real opoa_frac_const = 0.10; // assumed OPOA fraction

  Real a_opoa[num_mode];
  for (int n = 0; n < num_mode; ++n)
    a_opoa[n] = 0;
  // ----------------------------------------------------------------------
  // SOA species's equilibrium vapor mixing ratio at the ambient T and p,
  // assuming no solute effect. (This equilibrium mixing ratio depends only on
  // T, p and the parameters used inside the subroutine. There is no dependence
  // on SOA gas or aerosol mixing ratios, so the calculation does not need to
  // repeat during temporal sub-cycling.)
  // -----------------------------------------------------------------------
  soa_equilib_mixing_ratio_no_solute(temp, pmid, pstd, r_universal, g0_soa);
  // ------------------------------------------------------------------------------------
  // Calculate the mixing ratio of oxygeneated POA (OPOA) in each mode.
  // (This is needed to account for the solute effect on SOA's equilibrium
  // mixing ratio.)
  // --------------------------------------------------------------------------------------
  // Use the assumed OPOA fraction, per POA species and mode.
  Real opoa_frac[ntot_poaspec][num_mode];
  for (int i = 0; i < ntot_poaspec; ++i)
    for (int j = 0; j < num_mode; ++j)
      opoa_frac[i][j] = opoa_frac_const;

  // For primary carbon mode, keep the option to set opoa_frac=0 for consistency
  // with older code
  if (flag_pcarbon_opoa_frac_zero) {
    for (int i = 0; i < ntot_poaspec; ++i)
      opoa_frac[i][mode_pca] = 0.0;
  }

  // Within each mode, sum up the OPOA mixing ratios of all POA species.
  // The calculation uses the oxygenation fraction specified above and the
  // current (old) POA mixing ratios.

  for (int n = 0; n < ntot_soamode; ++n) {
    for (int i = 0; i < ntot_poaspec; ++i)
      a_opoa[n] += opoa_frac[i][n] * qaer_poa[i][n];
  }
  // -----------------------------------------------------------
  //  Time stepping -- uses multiple substeps to reach dtfull
  // -----------------------------------------------------------
  const int niter_max = 1000;

  mam_soaexch_advance_in_time(dt, dt_sub_soa_fixed, niter_max, alpha_astem,
                              uptkaer, &g0_soa, qgas_cur, a_opoa, qaer_cur,
                              qgas_avg, niter);
}
} // namespace gasaerexch
} // namespace mam4
#endif
