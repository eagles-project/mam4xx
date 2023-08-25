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
// NOTE: high_rel_err is unused currently
// const Real high_rel_err = 1.0e-4;
const int max_time_steps = 1000;

KOKKOS_INLINE_FUNCTION
void usrrxt(Real rxt[rxntot], // inout
            const Real temperature, const Real invariants[nfs], const Real mtot,
            const int usr_HO2_HO2_ndx, const int usr_DMS_OH_ndx,
            const int usr_SO2_OH_ndx, const int inv_h2o_ndx) {

  /*-----------------------------------------------------------------
   ... ho2 + ho2 --> h2o2
   note: this rate involves the water vapor number density
  -----------------------------------------------------------------*/
  const Real one = 1.0;
  if (usr_HO2_HO2_ndx > 0) {
    // BAD CONSTANT
    const Real ko = 3.5e-13 * haero::exp(430.0 / temperature);
    const Real kinf = 1.7e-33 * mtot * haero::exp(1000. / temperature);
    const Real fc = one + 1.4e-21 * invariants[inv_h2o_ndx] *
                              haero::exp(2200. / temperature);
    rxt[usr_HO2_HO2_ndx] = (ko + kinf) * fc;
  }

  /*-----------------------------------------------------------------
       ... DMS + OH  --> .5 * SO2
   -----------------------------------------------------------------*/
  if (usr_DMS_OH_ndx > 0) {
    // BAD CONSTANT
    const Real ko =
        one + 5.5e-31 * haero::exp(7460. / temperature) * mtot * 0.21;
    rxt[usr_DMS_OH_ndx] =
        1.7e-42 * haero::exp(7810. / temperature) * mtot * 0.21 / ko;
  }

  /*-----------------------------------------------------------------
         ... SO2 + OH  --> SO4  (REFERENCE?? - not Liao)
  -----------------------------------------------------------------*/
  if (usr_SO2_OH_ndx > 0) {
    // BAD CONSTANT
    const Real fc = 3.0e-31 * haero::pow(300. / temperature, 3.3);
    const Real ko = fc * mtot / (one + fc * mtot / 1.5e-12);
    rxt[usr_SO2_OH_ndx] =
        ko * haero::pow(0.6, one / (one + haero::square(haero::log10(
                                              fc * mtot / 1.5e-12))));
  }

} // usrrxt

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

  // dti := 1 / dt
  // lrxt := reaction rates in 1D array [1/cm^3/s]
  // lhet := washout rates [1/s]
  // iter_invariant := dti * solution + ind_prd
  // factor := boolean controlling whether to do LU factorization
  // lsol is local solution--appears to be identical to 'solution'
  // looks like 'solution' is local to imp_sol() and 'lsol' is local to
  // newton_raphson_iter(), and the final solutions is 'base_sol'
  // these are volume mixing ratios [kmol species/kmol dry air]
  // lsol := base_sol (at initialization), then when converged base_sol = lsol
  // solution := array from imp_sol that holds the intermediate solutions and
  //         also holds the solution after converged [kmol species/kmol dry air]
  // converged := array for entrywise convergence bools
  // convergence := overall bool flag for convergence
  // prod/loss := chemical production/loss rates [1/cm^3/s]
  // NOTE: max_delta doesn't appear to be used for anything within gas_chem.hpp
  //       however, it looks like it's written to output in MAM4
  // max_delta := abs(forcing / solution) if abs(solution) > 1.0e-20 and
  //              0 otherwise
  // epsilon := rel_err = 1.0e-3 (hardcoded above)

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

    // the units are internally consistent here, providing that
    // iter_invariant, prod, loss all have units [1/s] to match up with
    // solution (vmr) [-] and dti [1/s]. however, there could be other answers
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

    // NOTE: is there a particular reason we don't check on the first iteration?
    // seems like it'd be better to avoid the if on every iteration loop.
    // same deal below
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
        // TODO: is there a computational reason this needs to happen?
        // I suspect not, given that epsilon is hard-coded to 1e-3, meaning that
        // all of this logic surrounding 'converged[kk] = ...' is unnecessary
        bool frc_mask = haero::abs(forcing[mm]) > small;
        if (frc_mask) {
          // this ends up effectively being:
          //                         if (small < abs(forcing) <= eps * abs(sol))
          //                            => converged
          // so the lower bound appears unnecessary
          converged[kk] =
              haero::abs(forcing[mm]) <= epsilon[kk] * haero::abs(solution[mm]);
        } else {
          // and this is just; if (abs(forcing) <= small <= eps) => converged
          // and the implicit comparison of small and eps is not helpful
          converged[kk] = true;
        } // frc_mask
        if (!converged[kk]) {
          convergence = false;
        }
      } // end

      if (convergence) {
        return;
      }
    } // end if (nr_iter > 0)
  }   // end nr_iter loop
} // newton_raphson_iter() function

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

  // NOTE:
  // extfrc := external in-situ forcing [1/cm^3/s]

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
  // FIXME: BAD CONSTANT
  // what does this 4 represent, and would it ever be different?
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
  // track how much of the outer time step = delt (interval) has been completed
  // during Newton-Raphson iteration
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

    // TODO: the units seem wrong here--could these arrays hold quantities
    // with different units?
    // ind_prd has units [1/cm^3/s] (for the entries that are nonzero)
    // dti units are [1/s], and
    // solution is a volume mixing ratio [kmol species/kmol dry air]
    // NOTE: this could be correct if solution had units [1/cm^3]
    // which would line up with a number concentration
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
        cut_cnt += 1;
        if (cut_cnt < cut_limit) {
          dt *= half;
        } else {
          dt *= 0.1;
        } // cut_cnt < cut_limit
        // FIXME: figure out how we want to do error handling/logging
        // break;
        // cycle time_step_loop
      } else {
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
        printf("ERROR: imp_sol failure @ (lchnk,lev,col) = \n");
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
