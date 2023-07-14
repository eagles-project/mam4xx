#ifndef MAM4XX_PP_CHEMISTRY_HPP
#define MAM4XX_PP_CHEMISTRY_HPP
// pp_linoz_mam4_resus_mom_soag
// Generated code.
// Authors: Oscar Diaz-Ibarra (odiazib@sandia.gov)
//          Mike Schmidt (mjschm@sandia.gov)
namespace mam4 {
namespace gas_chemistry {
const int rxntot = 7;     // number of total reactions
const int gas_pcnst = 31; // number of gas phase species
const int nzcnt = 32;     // number of non-zero matrix entries
const int clscnt4 = 30;   // number of species in implicit class
const int extcnt = 9;     // number of species with external forcing
const int nfs = 8;        // number of fixed species
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
void setrxt(Real rates[rxntot], const Real temp) {
  rates[2] = 2.9000000000e-12 * haero::exp(-160.000000 / temp);
  rates[4] = 9.6000000000e-12 * haero::exp(-234.000000 / temp);
  rates[6] = 1.9000000000e-13 * haero::exp(520.000000 / temp);
}
KOKKOS_INLINE_FUNCTION
void set_rates(Real rxt_rates[rxntot], Real sol[gas_pcnst]) {
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
void adjrxt(Real rate[rxntot], Real inv[nfs], Real m) {
  rate[2] *= inv[4];
  rate[3] *= inv[4];
  rate[4] *= inv[4];
  rate[5] *= inv[4];
  rate[6] *= inv[5];
  rate[1] *= inv[6] * inv[6] / m;
}
KOKKOS_INLINE_FUNCTION
void exp_prod_loss(Real prod[31], Real loss[31], Real y[31], Real rxt[7],
                   Real het_rates[7]) {
  const Real zero = 0;
  loss[0] = (+het_rates[0]) * (+y[0]);
  prod[0] = zero;
}
KOKKOS_INLINE_FUNCTION
void imp_prod_loss(Real prod[clscnt4], Real loss[clscnt4], Real y[gas_pcnst],
                   const Real rxt[rxntot], const Real het_rates[gas_pcnst]) {
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
void indprd(const int class_id, Real prod[clscnt4], const Real rxt[rxntot],
            const Real extfrc[extcnt]) {
  const Real zero = 0;
  if (class_id == 1) {
    prod[0] = zero;
  } else if (class_id == 4) {
    prod[0] = +rxt[1];
    prod[1] = zero;
    prod[2] = +extfrc[0];
    prod[3] = zero;
    prod[4] = +extfrc[8];
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
void lu_fac(Real lu[nzcnt]) {
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
KOKKOS_INLINE_FUNCTION
void lu_slv(Real lu[nzcnt], Real b[clscnt4]) {
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
void lin_matrix(Real mat[nzcnt], const Real rxt[rxntot],
                const Real het_rates[gas_pcnst]) {
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
void nlnmat(Real mat[nzcnt], const Real lmat[nzcnt], const Real dti) {
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
} // namespace gas_chemistry
} // namespace mam4
#endif
