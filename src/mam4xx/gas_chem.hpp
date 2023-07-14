#ifndef MAM4XX_GAS_CHEMISTRY_HPP
#define MAM4XX_GAS_CHEMISTRY_HPP

#include <haero/aero_species.hpp>
#include <haero/atmosphere.hpp>
#include <haero/constants.hpp>
#include <haero/math.hpp>

#include <mam4xx/aero_config.hpp>
#include <mam4xx/conversions.hpp>
#include <mam4xx/mam4_types.hpp>
#include <mam4xx/utils.hpp>

using Real = haero::Real;

namespace mam4 {
namespace gas_chemistry {
const int nzcnt = 31;
const int rxntot = 7;
const int gas_pcnst = 7;
const int clscnt4 = 31;
const int itermax = 11;
const int extcnt = 7;

enum class SpeciesId {
  O3 = 0,
  H2O2 = 1,
  H2SO4 = 2,
  SO2 = 3,
  DMS = 4,
  SOAG = 5,
  so4_a1 = 6,
  pom_a1 = 7,
  soa_a1 = 8,
  bc_a1 = 9,
  dst_a1 = 10,
  ncl_a1 = 11,
  mom_a1 = 12,
  num_a1 = 13,
  so4_a2 = 14,
  soa_a2 = 15,
  ncl_a2 = 16,
  mom_a2 = 17,
  num_a2 = 18,
  dst_a3 = 19,
  ncl_a3 = 20,
  so4_a3 = 21,
  bc_a3 = 22,
  pom_a3 = 23,
  soa_a3 = 24,
  mom_a3 = 25,
  num_a3 = 26,
  pom_a4 = 27,
  bc_a4 = 28,
  mom_a4 = 29,
  num_a4 = 30,
}; // enum class GasId

enum class ReactionId {
  jh2o2 = 0,
  usr_HO2_HO2 = 1,
  r0003 = 2,
  usr_SO2_OH = 3,
  r0005 = 4,
  usr_DMS_OH = 5,
  r0007 = 6,
}; // enum class ReactionId

KOKKOS_INLINE_FUNCTION
void setrxt(Real rates[7], const Real temp) {
  rates[2] = 2.9000000000e-12 * haero::exp(-160.000000 / temp);
  rates[4] = 9.6000000000e-12 * haero::exp(-234.000000 / temp);
  rates[6] = 1.9000000000e-13 * haero::exp(520.000000 / temp);
}

KOKKOS_INLINE_FUNCTION
void set_rates(Real rxt_rates[7], Real sol[7]) {
  // rate_const*H2O2
  rxt_rates[0] *= sol[1];
  // rate_const*OH*H2O2
  rxt_rates[2] *= sol[1];
  // rate_const*OH*SO2
  rxt_rates[3] *= sol[3];
  // rate_const*OH*DMS
  rxt_rates[4] *= sol[4];
  // rate_const*OH*DMS
  rxt_rates[5] *= sol[4];
  // rate_const*NO3*DMS
  rxt_rates[6] *= sol[4];
}
KOKKOS_INLINE_FUNCTION
void lu_slv(Real lu[31], Real b[29]) {
  b[29] *= lu[31];
  b[28] *= lu[30];
  b[27] *= lu[29];
  b[26] *= lu[28];
  b[25] *= lu[27];
  b[24] *= lu[26];
  b[23] *= lu[25];
  b[22] *= lu[24];
  b[21] *= lu[23];
  b[20] *= lu[22];
  b[19] *= lu[21];
  b[18] *= lu[20];
  b[17] *= lu[19];
  b[16] *= lu[18];
  b[15] *= lu[17];
  b[14] *= lu[16];
  b[13] *= lu[15];
  b[12] *= lu[14];
  b[11] *= lu[13];
  b[10] *= lu[12];
  b[9] *= lu[11];
  b[8] *= lu[10];
  b[7] *= lu[9];
  b[6] *= lu[8];
  b[5] *= lu[7];
  b[4] *= lu[6];
  b[3] *= lu[5];
  b[2] -= lu[4] * b[3];
  b[2] *= lu[3];
  b[1] -= lu[2] * b[2];
  b[1] *= lu[1];
  b[0] *= lu[0];
}

KOKKOS_INLINE_FUNCTION
void lu_fac(Real lu[31]) {
  const Real one = 1;
  lu[0] /= one;
  lu[1] /= one;
  lu[3] /= one;
  lu[5] /= one;
  lu[6] /= one;
  lu[7] /= one;
  lu[8] /= one;
  lu[9] /= one;
  lu[10] /= one;
  lu[11] /= one;
  lu[12] /= one;
  lu[13] /= one;
  lu[14] /= one;
  lu[15] /= one;
  lu[16] /= one;
  lu[17] /= one;
  lu[18] /= one;
  lu[19] /= one;
  lu[20] /= one;
  lu[21] /= one;
  lu[22] /= one;
  lu[23] /= one;
  lu[24] /= one;
  lu[25] /= one;
  lu[26] /= one;
  lu[27] /= one;
  lu[28] /= one;
  lu[29] /= one;
  lu[30] /= one;
  lu[31] /= one;
}
// update
KOKKOS_INLINE_FUNCTION
void lin_matrix(Real mat[31], const Real rxt[7], const Real het_rates[7]) {
  mat[0] = -(+rxt[0] + rxt[2] + het_rates[1]);
  mat[1] = -(+het_rates[2]);
  mat[2] = +rxt[3];
  mat[3] = -(+rxt[3] + het_rates[3]);
  mat[4] = +rxt[4] + 0.500000 * rxt[5] + rxt[6];
  mat[5] = -(+rxt[4] + rxt[5] + rxt[6] + het_rates[4]);
  mat[6] = -(+het_rates[5]);
  mat[7] = -(+het_rates[6]);
  mat[8] = -(+het_rates[7]);
  mat[9] = -(+het_rates[8]);
  mat[10] = -(+het_rates[9]);
  mat[11] = -(+het_rates[10]);
  mat[12] = -(+het_rates[11]);
  mat[13] = -(+het_rates[12]);
  mat[14] = -(+het_rates[13]);
  mat[15] = -(+het_rates[14]);
  mat[16] = -(+het_rates[15]);
  mat[17] = -(+het_rates[16]);
  mat[18] = -(+het_rates[17]);
  mat[19] = -(+het_rates[18]);
  mat[20] = -(+het_rates[19]);
  mat[21] = -(+het_rates[20]);
  mat[22] = -(+het_rates[21]);
  mat[23] = -(+het_rates[22]);
  mat[24] = -(+het_rates[23]);
  mat[25] = -(+het_rates[24]);
  mat[26] = -(+het_rates[25]);
  mat[27] = -(+het_rates[26]);
  mat[28] = -(+het_rates[27]);
  mat[29] = -(+het_rates[28]);
  mat[30] = -(+het_rates[29]);
  mat[31] = -(+het_rates[30]);
}

KOKKOS_INLINE_FUNCTION
void indprd(const int class_id, Real prod[31], const Real rxt[7],
            const Real extfrc[7]) {
  const Real zero = 0;
  if (class_id == 1) {
    prod[0] = zero;
  } else if (class_id == 4) {
    prod[0] = +rxt[1];
    prod[1] = zero;
    prod[2] = +extfrc[0];
    prod[3] = zero;
    prod[4] = zero;
    prod[5] = +extfrc[1];
    prod[6] = zero;
    prod[7] = zero;
    prod[8] = zero;
    prod[9] = zero;
    prod[10] = zero;
    prod[11] = zero;
    prod[12] = +extfrc[5];
    prod[13] = +extfrc[2];
    prod[14] = zero;
    prod[15] = zero;
    prod[16] = zero;
    prod[17] = +extfrc[6];
    prod[18] = zero;
    prod[19] = zero;
    prod[20] = zero;
    prod[21] = zero;
    prod[22] = zero;
    prod[23] = zero;
    prod[24] = zero;
    prod[25] = zero;
    prod[26] = +extfrc[3];
    prod[27] = +extfrc[4];
    prod[28] = zero;
    prod[29] = +extfrc[7];
  }
}

KOKKOS_INLINE_FUNCTION
void imp_prod_loss(Real prod[31], Real loss[31], Real y[31], const Real rxt[7],
                   const Real het_rates[7]) {
  const Real zero = 0;
  loss[0] = (+het_rates[1] + rxt[0] + rxt[2]) * (+y[1]);
  prod[0] = zero;
  loss[1] = (+het_rates[2]) * (+y[2]);
  prod[1] = (+rxt[3]) * (+y[3]);
  loss[2] = (+het_rates[3] + rxt[3]) * (+y[3]);
  prod[2] = (+rxt[4] + 0.500000 * rxt[5] + rxt[6]) * (+y[4]);
  loss[3] = (+het_rates[4] + rxt[4] + rxt[5] + rxt[6]) * (+y[4]);
  prod[3] = zero;
  loss[4] = (+het_rates[5]) * (+y[5]);
  prod[4] = zero;
  loss[5] = (+het_rates[6]) * (+y[6]);
  prod[5] = zero;
  loss[6] = (+het_rates[7]) * (+y[7]);
  prod[6] = zero;
  loss[7] = (+het_rates[8]) * (+y[8]);
  prod[7] = zero;
  loss[8] = (+het_rates[9]) * (+y[9]);
  prod[8] = zero;
  loss[9] = (+het_rates[10]) * (+y[10]);
  prod[9] = zero;
  loss[10] = (+het_rates[11]) * (+y[11]);
  prod[10] = zero;
  loss[11] = (+het_rates[12]) * (+y[12]);
  prod[11] = zero;
  loss[12] = (+het_rates[13]) * (+y[13]);
  prod[12] = zero;
  loss[13] = (+het_rates[14]) * (+y[14]);
  prod[13] = zero;
  loss[14] = (+het_rates[15]) * (+y[15]);
  prod[14] = zero;
  loss[15] = (+het_rates[16]) * (+y[16]);
  prod[15] = zero;
  loss[16] = (+het_rates[17]) * (+y[17]);
  prod[16] = zero;
  loss[17] = (+het_rates[18]) * (+y[18]);
  prod[17] = zero;
  loss[18] = (+het_rates[19]) * (+y[19]);
  prod[18] = zero;
  loss[19] = (+het_rates[20]) * (+y[20]);
  prod[19] = zero;
  loss[20] = (+het_rates[21]) * (+y[21]);
  prod[20] = zero;
  loss[21] = (+het_rates[22]) * (+y[22]);
  prod[21] = zero;
  loss[22] = (+het_rates[23]) * (+y[23]);
  prod[22] = zero;
  loss[23] = (+het_rates[24]) * (+y[24]);
  prod[23] = zero;
  loss[24] = (+het_rates[25]) * (+y[25]);
  prod[24] = zero;
  loss[25] = (+het_rates[26]) * (+y[26]);
  prod[25] = zero;
  loss[26] = (+het_rates[27]) * (+y[27]);
  prod[26] = zero;
  loss[27] = (+het_rates[28]) * (+y[28]);
  prod[27] = zero;
  loss[28] = (+het_rates[29]) * (+y[29]);
  prod[28] = zero;
  loss[29] = (+het_rates[30]) * (+y[30]);
  prod[29] = zero;
}

KOKKOS_INLINE_FUNCTION
void exp_prod_loss(Real prod[31], Real loss[31], Real y[31], Real rxt[7],
                   Real het_rates[7]) {
  const Real zero = 0;
  loss[0] = (+het_rates[0]) * (+y[0]);
  prod[0] = zero;
}

KOKKOS_INLINE_FUNCTION
void adjrxt(Real rate[7], Real inv[7], Real m) {
  rate[2] *= inv[4];
  rate[3] *= inv[4];
  rate[4] *= inv[4];
  rate[5] *= inv[4];
  rate[6] *= inv[5];
  rate[1] *= inv[6] * inv[6] / m;
}

KOKKOS_INLINE_FUNCTION
void nlnmat(Real mat[nzcnt], const Real lmat[nzcnt], const Real dti) 
{
mat[0] = lmat[0] - dti;
mat[1] = lmat[1] - dti;
mat[2] = lmat[2];
mat[3] = lmat[3] - dti;
mat[4] = lmat[4];
mat[5] = lmat[5] - dti;
mat[6] = lmat[6] - dti;
mat[7] = lmat[7] - dti;
mat[8] = lmat[8] - dti;
mat[9] = lmat[9] - dti;
mat[10] = lmat[10] - dti;
mat[11] = lmat[11] - dti;
mat[12] = lmat[12] - dti;
mat[13] = lmat[13] - dti;
mat[14] = lmat[14] - dti;
mat[15] = lmat[15] - dti;
mat[16] = lmat[16] - dti;
mat[17] = lmat[17] - dti;
mat[18] = lmat[18] - dti;
mat[19] = lmat[19] - dti;
mat[20] = lmat[20] - dti;
mat[21] = lmat[21] - dti;
mat[22] = lmat[22] - dti;
mat[23] = lmat[23] - dti;
mat[24] = lmat[24] - dti;
mat[25] = lmat[25] - dti;
mat[26] = lmat[26] - dti;
mat[27] = lmat[27] - dti;
mat[28] = lmat[28] - dti;
mat[29] = lmat[29] - dti;
mat[30] = lmat[30] - dti;
mat[31] = lmat[31] - dti;
}

KOKKOS_INLINE_FUNCTION
void newton_raphson_iter(
    const Real dti, const Real lin_jac[nzcnt], const Real lrxt[rxntot],
    const Real lhet[gas_pcnst],        // & ! in
    const int iter_invariant[clscnt4], //              & ! in
    const bool factor[itermax], int permute_4[gas_pcnst],
    int clsmap_4[gas_pcnst], Real lsol[gas_pcnst],
    Real solution[clscnt4],                    //              & ! inout
    bool converged[clscnt4], bool convergence, //         & ! out
    Real prod[clscnt4], Real loss[clscnt4], Real max_delta[clscnt4],
    // work arrays
    Real epsilon[clscnt4]) {

  // !-----------------------------------------------------
  // ! the newton-raphson iteration for f(y) = 0
  // !-----------------------------------------------------

  Real sys_jac[nzcnt] = {};
  Real forcing[clscnt4] = {};
  // BAD CONSTANT
  const Real small = 1.e-40;
  const Real zero = 0;

  for (int nr_iter = 0; nr_iter < itermax; ++nr_iter) {
    // !-----------------------------------------------------------------------
    // ! ... the non-linear component
    // !-----------------------------------------------------------------------

    if (factor[nr_iter]) {
      nlnmat(
          sys_jac, //   & ! out
          lin_jac,
          dti); // ! in
                // !-----------------------------------------------------------------------
      // ! ... factor the "system" matrix
      // !-----------------------------------------------------------------------
      lu_fac(sys_jac);

    } // factor
    // !-----------------------------------------------------------------------
    // ! ... form f(y)
    // !-----------------------------------------------------------------------
    imp_prod_loss(prod, loss,        // & ! out
                  lsol, lrxt, lhet); // ! in

    for (int mm = 0; mm < clscnt4; ++mm) {
      forcing[mm] =
          solution[mm] * dti - (iter_invariant[mm] + prod[mm] - loss[mm]);
    } // mm

    // !-----------------------------------------------------------------------
    // ! ... solve for the mixing ratio at t(n+1)
    // !-----------------------------------------------------------------------
    lu_slv(sys_jac, forcing);
    for (int mm = 0; mm < clscnt4; ++mm) {
      solution[mm] += forcing[mm];
    } // mm

    // !-----------------------------------------------------------------------
    // ! ... convergence measures
    // !-----------------------------------------------------------------------

    if (nr_iter > 0) {
      for (int kk = 0; kk < clscnt4; ++kk) {
        int mm = permute_4[kk];
        // BAD CONSTANT
        if (haero::abs(solution[mm]) > 1.e-20) {
          max_delta[kk] = haero::abs(forcing[mm] / solution[mm]);
        } else {
          max_delta[kk] = zero;
        }

      } // kk

    } // nr_iter

    // !-----------------------------------------------------------------------
    // ! ... limit iterate
    // !-----------------------------------------------------------------------
    for (int kk = 0; kk < clscnt4; ++kk) {
      if (solution[kk] < zero) {
        solution[kk] = zero;
      }
    } // end kk

    // !-----------------------------------------------------------------------
    // ! ... transfer latest solution back to work array
    // !-----------------------------------------------------------------------

    for (int kk = 0; kk < clscnt4; ++kk) {
      int jj = clsmap_4[kk];
      int mm = permute_4[kk];
      lsol[jj] = solution[mm];
    } // end kk

    // !-----------------------------------------------------------------------
    // ! ... check for convergence
    // !-----------------------------------------------------------------------

    if (nr_iter > 0) {
      convergence = true;
      for (int kk = 0; kk < clscnt4; ++kk) {
        converged[kk] = true;

        int mm = permute_4[kk];
        bool frc_mask = haero::abs(forcing[mm]) > small ? true : false;
        if (frc_mask) {
          converged[kk] =
              haero::abs(forcing[mm]) <= epsilon[kk] * haero::abs(solution[mm])
                  ? true
                  : false;
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

  } // end nr_iter

} // newton_raphson_iter


KOKKOS_INLINE_FUNCTION
void imp_sol(Real base_sol[gas_pcnst], //    ! species mixing ratios [vmr] & !
                                       //    inout
             const Real reaction_rates[rxntot], const Real het_rates[gas_pcnst],
             const Real extfrc[extcnt], Real &delt, int permute_4[gas_pcnst],
             int clsmap_4[gas_pcnst], const bool factor[itermax],
             Real epsilon[clscnt4], Real prod_out[clscnt4],
             Real loss_out[clscnt4]
             //                   xhnm, ncol, lchnk, ltrop
) {

  //     !-----------------------------------------------------------------------
  // ! ... imp_sol advances the volumetric mixing ratio
  // ! forward one time step via the fully implicit euler scheme.
  // ! this source is meant for small l1 cache machines such as
  // ! the intel pentium and itanium cpus
  // !-----------------------------------------------------------------------
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
  // ! initiate variables
  // prod_out(:,:,:) = 0._r8
  // loss_out(:,:,:) = 0._r8
  // solution(:) = 0._r8

  // !-----------------------------------------------------------------------
  // ! ... class independent forcing
  // !-----------------------------------------------------------------------
  indprd(4,                     //                  & ! in
         ind_prd,               //               & ! inout
         reaction_rates, extfrc //   & ! in
  );                            // ! in

  // level_loop : do lev = 1,pver
  //    column_loop : do icol = 1,ncol
  //       if (lev <= ltrop(icol)) cycle column_loop

  // !-----------------------------------------------------------------------
  // ! ... transfer from base to local work arrays
  // !-----------------------------------------------------------------------

  // do mm = 1,rxntot
  //   lrxt(mm) = reaction_rates(icol,lev,mm)
  // enddo
  // if( gas_pcnst > 0 ) then
  //   do mm = 1,gas_pcnst
  //     lhet(mm) = het_rates(icol,lev,mm)
  //   enddo
  // endif

  Real solution[clscnt4] = {};
  int iter_invariant[clscnt4] = {};

  // !-----------------------------------------------------------------------
  //       ! ... time step loop
  //       !-----------------------------------------------------------------------
  Real dt = delt;
  int cut_cnt = 0;
  int fail_cnt = 0;
  int stp_con_cnt = 0;
  Real interval_done = zero;
  // time_step_loop : do
  for (int i = 0; i < 10; ++i) {
    const Real dti = one / dt;
    // !-----------------------------------------------------------------------
    // ! ... transfer from base to local work arrays
    // !-----------------------------------------------------------------------
    auto &lsol = base_sol;
    // !-----------------------------------------------------------------------
    // ! ... transfer from base to class array
    // !-----------------------------------------------------------------------

    for (int kk = 0; kk < clscnt4; ++kk) {
      int jj = clsmap_4[kk];
      int mm = permute_4[kk];
      solution[mm] = lsol[jj];
    } // kk

    // !-----------------------------------------------------------------------
    // ! ... set the iteration invariant part of the function f(y)
    // !-----------------------------------------------------------------------
    for (int mm = 0; mm < clscnt4; ++mm) {
      iter_invariant[mm] = dti * solution[mm] + ind_prd[mm];
    } // mm

    //-----------------------------------------------------------------------
    // ... the linear component
    //-----------------------------------------------------------------------
    // linmat( lin_jac,  //         & ! out
    //        lsol, lrxt, lhet   );// ! in
    lin_matrix(lin_jac,                    // ! out
               reaction_rates, het_rates); // in

    // !=======================================================================
    // ! the newton-raphson iteration for f(y) = 0
    // !=======================================================================

    newton_raphson_iter(dti, lin_jac, reaction_rates, het_rates, // & ! in
                        iter_invariant, //                & ! in
                        factor, permute_4, clsmap_4, lsol,
                        solution,               //              & ! inout
                        converged, convergence, //       & ! out
                        prod, loss, max_delta, epsilon); // ! out

    // -----------------------------------------------------------------------
    //  ... check for newton-raphson convergence
    // -----------------------------------------------------------------------

    if (!convergence) {
      // !-----------------------------------------------------------------------
      //           ! ... non-convergence
      //           !-----------------------------------------------------------------------

      // !-----------------------------------------------------------------------
      //           ! ... non-convergence
      //           !-----------------------------------------------------------------------
      fail_cnt += fail_cnt;
      // nstep = get_nstep()
      // write(iulog,'('' imp_sol: Time step '',1p,e21.13,'' failed to converge
      // @ (lchnk,lev,col,nstep) = '',4i6)') &
      //      dt,lchnk,lev,icol,nstep

      stp_con_cnt = 0;

      if (cut_cnt < cut_limit) {
        cut_cnt += cut_cnt;

        if (cut_cnt < cut_limit) {
          dt *= half;

        } else {
          dt *= 0.1;
        } // cut_cnt < cut_limit
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

    } // ! convergence

    // !-----------------------------------------------------------------------
    //            ! ... check for interval done
    //            !-----------------------------------------------------------------------

    interval_done += dt;

    // BAD CONSTANT
    if (haero::abs(delt - interval_done) <= .0001) {
      if (fail_cnt > 0) {
        printf("'imp_sol : @ (lchnk,lev,col) = \n");
        // write(iulog,*) 'imp_sol : @ (lchnk,lev,col) = ',lchnk,lev,icol,'
        // failed ',fail_cnt,' times'
      }
      break;
    } else {
      //                  !-----------------------------------------------------------------------
      // ! ... transfer latest solution back to base array
      // !-----------------------------------------------------------------------
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

  } // time_step_loop

  // !-----------------------------------------------------------------------
  //         ! ... Transfer latest solution back to base array
  //         !     and calculate Prod/Loss history buffers
  //         !-----------------------------------------------------------------------

  for (int kk = 0; kk < clscnt4; ++kk) {
    const int jj = clsmap_4[kk];
    const int mm = permute_4[kk];
    // ! ... Transfer latest solution back to base array
    base_sol[jj] = solution[mm];
    // ! ... Prod/Loss history buffers...
    prod_out[kk] = prod[mm] + ind_prd[mm];
    loss_out[kk] = loss[mm];

  } // cls_loop

} // imp_sol

} // namespace gas_chemistry
} // namespace mam4
#endif
