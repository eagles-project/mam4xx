#ifndef MAM4XX_GAS_CHEMISTRY_HPP
#define MAM4XX_GAS_CHEMISTRY_HPP

#include <haero/aero_species.hpp>
#include <haero/atmosphere.hpp>
#include <haero/constants.hpp>
#include <haero/math.hpp>

#include <mam4xx/aero_config.hpp>
#include <mam4xx/conversions.hpp>
#include <mam4xx/gas_chem_mechanism.hpp>
#include <mam4xx/mam4_types.hpp>
#include <mam4xx/utils.hpp>

using Real = haero::Real;

namespace mam4 {

namespace gas_chemistry {

// BAD CONSTANTs
const int itermax = 11;
const Real rel_err = 1.0e-3;
const Real high_rel_err = 1.0e-4;
const int max_time_steps = 1000;

// initialize the solver (error tolerance)
KOKKOS_INLINE_FUNCTION
void imp_slv_inti(Real epsilon[clscnt4]) {
  for (int i = 0; i < clscnt4; ++i) {
    epsilon[i] = rel_err;
  }
}

KOKKOS_INLINE_FUNCTION
void newton_raphson_iter(const Real dti, const Real lin_jac[nzcnt],
                         const Real lrxt[rxntot],
                         const Real lhet[gas_pcnst],         // in
                         const Real iter_invariant[clscnt4], // in
                         const bool factor[itermax],
                         const int permute_4[gas_pcnst],
                         const int clsmap_4[gas_pcnst], Real lsol[gas_pcnst],
                         Real solution[clscnt4],                     // inout
                         bool converged[clscnt4], bool &convergence, // out
                         Real prod[clscnt4], Real loss[clscnt4],
                         Real max_delta[clscnt4],
                         // work array
                         Real epsilon[clscnt4]) {

  // -----------------------------------------------------
  //  the newton-raphson iteration for f(y) = 0
  // -----------------------------------------------------

  Real sys_jac[nzcnt] = {};
  Real forcing[clscnt4] = {};
  // BAD CONSTANT
  const Real small = 1.0e-40;
  const Real zero = 0;

  for (int nr_iter = 0; nr_iter < itermax; ++nr_iter) {
    // -----------------------------------------------------------------------
    //  ... the non-linear component
    // -----------------------------------------------------------------------

    if (factor[nr_iter]) {
      nlnmat(sys_jac, // out
             lin_jac,
             dti); // in
      // -----------------------------------------------------------------------
      //  ... factor the "system" matrix
      // -----------------------------------------------------------------------

      lu_fac(sys_jac);

    } // factor
    // -----------------------------------------------------------------------
    //  ... form f(y)
    // -----------------------------------------------------------------------
    imp_prod_loss(prod, loss,        // out
                  lsol, lrxt, lhet); // in

    for (int mm = 0; mm < clscnt4; ++mm) {
      forcing[mm] =
          solution[mm] * dti - (iter_invariant[mm] + prod[mm] - loss[mm]);
    } // mm

    // -----------------------------------------------------------------------
    //  ... solve for the mixing ratio at t(n+1)
    // -----------------------------------------------------------------------
    lu_slv(sys_jac, forcing);
    for (int mm = 0; mm < clscnt4; ++mm) {
      solution[mm] += forcing[mm];
    } // mm

    // -----------------------------------------------------------------------
    //  ... convergence measures
    // -----------------------------------------------------------------------

    if (nr_iter > 0) {
      for (int kk = 0; kk < clscnt4; ++kk) {
        int mm = permute_4[kk];
        // BAD CONSTANT
        if (haero::abs(solution[mm]) > 1.0e-20) {
          max_delta[kk] = haero::abs(forcing[mm] / solution[mm]);
        } else {
          max_delta[kk] = zero;
        }

      } // kk

    } // nr_iter

    // -----------------------------------------------------------------------
    //  ... limit iterate
    // -----------------------------------------------------------------------
    for (int kk = 0; kk < clscnt4; ++kk) {
      if (solution[kk] < zero) {
        solution[kk] = zero;
      }
    } // end kk

    // -----------------------------------------------------------------------
    //  ... transfer latest solution back to work array
    // -----------------------------------------------------------------------

    for (int kk = 0; kk < clscnt4; ++kk) {
      int jj = clsmap_4[kk];
      int mm = permute_4[kk];
      lsol[jj] = solution[mm];
    } // end kk

    // -----------------------------------------------------------------------
    //  ... check for convergence
    // -----------------------------------------------------------------------

    if (nr_iter > 0) {
      convergence = true;
      for (int kk = 0; kk < clscnt4; ++kk) {
        converged[kk] = true;

        int mm = permute_4[kk];
        bool frc_mask = haero::abs(forcing[mm]) > small;
        if (frc_mask) {
          converged[kk] =
              haero::abs(forcing[mm]) <= epsilon[kk] * haero::abs(solution[mm]);
        } else {
          converged[kk] = true;

        } // frc_mask
        if (!converged[kk]) {
          convergence = false;
        }
      } // end

      if (convergence) {
        return;
      }
    } // end nr_iter
  }   // end nr_iter
} // newton_raphson_iter

KOKKOS_INLINE_FUNCTION
void imp_sol(Real base_sol[gas_pcnst], // inout - species mixing ratios [vmr]
             const Real reaction_rates[rxntot], const Real het_rates[gas_pcnst],
             const Real extfrc[extcnt], Real &delt,
             const int permute_4[gas_pcnst], const int clsmap_4[gas_pcnst],
             const bool factor[itermax], Real epsilon[clscnt4],
             Real prod_out[clscnt4], Real loss_out[clscnt4]) {

  // ---------------------------------------------------------------------------
  //  ... imp_sol advances the volumetric mixing ratio
  //  forward one time step via the fully implicit euler scheme.
  //
  // NOTE: does anyone know what this is referring to?
  // can probably lose it since it looks like these chips were axed in 2020/2023
  // this source is meant for small l1 cache machines such as
  // the intel pentium and itanium cpus
  // ---------------------------------------------------------------------------
  const Real zero = 0;
  const Real half = 0.5;
  const Real one = 1;
  const Real two = 2;

  const int cut_limit = 5;

  Real ind_prd[clscnt4] = {};
  Real lin_jac[nzcnt] = {};
  bool converged[clscnt4] = {};
  bool convergence = false;
  Real prod[clscnt4] = {};
  Real loss[clscnt4] = {};
  Real max_delta[clscnt4] = {};

  // -----------------------------------------------------------------------
  //  ... class independent forcing
  // -----------------------------------------------------------------------
  indprd(4,                       // in
         ind_prd,                 // inout
         reaction_rates, extfrc); // in

  Real solution[clscnt4] = {};
  Real iter_invariant[clscnt4] = {};

  // !-----------------------------------------------------------------------
  //       ! ... time step loop
  //       !-----------------------------------------------------------------------
  Real dt = delt;
  int cut_cnt = 0;
  int fail_cnt = 0;
  int stp_con_cnt = 0;
  Real interval_done = zero;
  // time_step_loop
  for (int i = 0; i < max_time_steps; ++i) {
    const Real dti = one / dt;
    // -----------------------------------------------------------------------
    //  ... transfer from base to local work arrays
    // -----------------------------------------------------------------------
    auto &lsol = base_sol;
    // -----------------------------------------------------------------------
    //  ... transfer from base to class array
    // -----------------------------------------------------------------------

    for (int kk = 0; kk < clscnt4; ++kk) {
      int jj = clsmap_4[kk];
      int mm = permute_4[kk];
      solution[mm] = lsol[jj];
    } // kk

    // -----------------------------------------------------------------------
    //  ... set the iteration invariant part of the function f(y)
    // -----------------------------------------------------------------------

    for (int mm = 0; mm < clscnt4; ++mm) {
      iter_invariant[mm] = dti * solution[mm] + ind_prd[mm];
    } // mm
    //-----------------------------------------------------------------------
    // ... the linear component
    //-----------------------------------------------------------------------
    linmat(lin_jac,                    //  out
           reaction_rates, het_rates); // in

    // =======================================================================
    //  the newton-raphson iteration for f(y) = 0
    // =======================================================================

    newton_raphson_iter(dti, lin_jac, reaction_rates, het_rates, // in
                        iter_invariant,                          // in
                        factor, permute_4, clsmap_4, lsol,
                        solution,                        // inout
                        converged, convergence,          // out
                        prod, loss, max_delta, epsilon); // out

    // -----------------------------------------------------------------------
    //  ... check for newton-raphson convergence
    // -----------------------------------------------------------------------
    if (!convergence) {
      // -----------------------------------------------------------------------
      //            ... non-convergence
      // -----------------------------------------------------------------------
      fail_cnt += fail_cnt;

      stp_con_cnt = 0;

      if (cut_cnt < cut_limit) {
        cut_cnt += cut_cnt;

        if (cut_cnt < cut_limit) {
          dt *= half;

        } else {
          dt *= 0.1;
        } // cut_cnt < cut_limit
        // FIXME: is this part of the below FIXME?
        // break;
        // cycle time_step_loop
      } else {
        // FIXME
        // write(iulog,'('' imp_sol: Failed to converge @
        // (lchnk,lev,col,nstep,dt,time) = '',4i6,1p,2e21.13)') &
        //                   lchnk,lev,icol,nstep,dt,interval_done+dt
        // do mm = 1,clscnt4
        //                 if( .not. converged(mm) ) then
        //                    write(iulog,'(1x,a8,1x,1pe10.3)')
        //                    solsym(clsmap(mm,4)), max_delta(mm)
        //                 endif
        //              enddo
      } //  cut_cnt < cut_limit
    }   // non-convergence

    // -----------------------------------------------------------------------
    // ... check for interval done
    // -----------------------------------------------------------------------

    interval_done += dt;

    // BAD CONSTANT
    if (haero::abs(delt - interval_done) <= 0.0001) {
      if (fail_cnt > 0) {
        // FIXME: probably handle this more gracefully via error logging?
        printf("'imp_sol : @ (lchnk,lev,col) = \n");
      }
      break;
    } else {
      // -----------------------------------------------------------------------
      //  ... transfer latest solution back to base array
      // -----------------------------------------------------------------------
      if (convergence) {
        stp_con_cnt += 1;
      }

      for (int mm = 0; mm < gas_pcnst; ++mm) {
        base_sol[mm] = lsol[mm];
      }

      if (stp_con_cnt >= 2) {
        dt *= two;
        stp_con_cnt = 0;
      }

      dt = haero::min(dt, delt - interval_done);

    } // abs( delt - interval_done ) <= .0001
  }   // time_step_loop

  //-----------------------------------------------------------------------
  // ... Transfer latest solution back to base array
  //     and calculate Prod/Loss history buffers
  //-----------------------------------------------------------------------

  for (int kk = 0; kk < clscnt4; ++kk) {
    const int jj = clsmap_4[kk];
    const int mm = permute_4[kk];
    //  ... Transfer latest solution back to base array
    base_sol[jj] = solution[mm];
    //  ... Prod/Loss history buffers...
    prod_out[kk] = prod[mm] + ind_prd[mm];
    loss_out[kk] = loss[mm];

  } // cls_loop
} // imp_sol
} // namespace gas_chemistry
} // namespace mam4
#endif
