// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

/*--------------------------------------------------------------------
 linearized ozone chemistry LINOZ
 from Hsu and Prather, JGR, 2008

 written by Jean-Francois Lamarque (September 2008)
 modified by
     24 Oct 2008 -- Francis Vitt
      9 Dec 2008 -- Philip Cameron-Smith, LLNL, -- added ltrop
      4 Jul 2019 -- Qi Tang (LLNL), Juno Hsu (UCI), -- added sfcsink
--------------------------------------------------------------------*/

#ifndef MAM4XX_LIN_STRAT_CHEM_HPP
#define MAM4XX_LIN_STRAT_CHEM_HPP

#include <haero/math.hpp>
#include <mam4xx/aero_config.hpp>

namespace mam4 {
namespace lin_strat_chem {

constexpr Real radians_to_degrees = 180. / haero::Constants::pi;
// number of vertical levels
constexpr int pver = mam4::nlev;

KOKKOS_INLINE_FUNCTION
void psc_activation(const Real lats, const Real temp, const Real pmid,
                    const Real sza, const Real linoz_cariolle_psc,
                    const Real delta_t, // in
                    const bool excess_chlorine,
                    const Real o3_old, // in
                    const Real psc_T,  // PSC ozone loss T (K) threshold
                    const Real chlorine_loading, Real &o3_new,
                    Real &do3_linoz_psc) {

  constexpr Real ninety = 90.0;
  constexpr Real sixteen = 16.0;
  constexpr Real zero = 0;
  constexpr Real one_hundred_k = 1e5;

  // @param[in] psc_T set from namelist input linoz_psc_T
  // @param[in] lats lattitude [degree]
  // @param[in] temp Temperature [K]
  // @param[in] pmid Midpoint Pressure [Pa]
  // @param[in] sza  local solar zenith angle
  // @param[in] linoz_cariolle_psc  Cariolle parameter for PSC loss of ozone
  // [1/s]
  // @param[in] delta_t  timestep size [secs]
  // @param[in] excess_chlorine .TRUE. if chlorine is more than the backgroud
  // chlorine
  // @param[in] o3_old  ozone volume mixing ratio [vmr]
  // @param[in] psc_T
  // @param[in] chlorine_loading

  // intent-outs
  // @param[in] o3_new  ozone volume mixing ratio [vmr]
  // @param[in] do3_linoz_psc [vmr/s]

  // BAD CONSTANT
  constexpr Real lats_threshold = 40.0;
  // BAD CONSTANT
  constexpr Real chlorine_loading_1987 = 2.5977; //    ! EESC value [ppbv]

  // use only if abs(latitude) > lats_threshold
  if (haero::abs(lats) > lats_threshold) {
    if (excess_chlorine) {
      if (temp <= psc_T) {
        // define maximum SZA for PSC loss (= tangent height at sunset)
        const Real max_sza =
            (ninety + haero::sqrt(haero::max(
                          sixteen * haero::log10(one_hundred_k / pmid), zero)));

        if ((sza * radians_to_degrees) <= max_sza) {
          const Real psc_loss = haero::exp(
              -linoz_cariolle_psc *
              haero::square(chlorine_loading / chlorine_loading_1987) *
              delta_t);

          o3_new = o3_old * psc_loss;

          do3_linoz_psc = (o3_new - o3_old) / delta_t; // output diagnostic

        } // end sza * radians_to_degrees) <= max_sza

      } // temp <= psc_T

    } // end excess_chlorine
  }   // end abs(lats) > lats_threshold

} // psc_activation

KOKKOS_INLINE_FUNCTION
void compute_steady_state_ozone(const Real o3_clim, const Real o3col_du,
                                const Real temperature, const Real linoz_t_clim,
                                const Real linoz_o3col_clim, // in
                                const Real linoz_PmL_clim,
                                const Real linoz_dPmL_dT,
                                const Real linoz_dPmL_dO3,
                                const Real linoz_dPmL_dO3col, // in
                                Real &ss_o3) {
  // intent in
  // @param[in] o3_clim           ! ozone (climatology) [vmr]
  // @param[in] o3col_du          ! ozone column above box [DU]
  // @param[in] temperature              ! temperature [K]
  // @param[in] linoz_t_clim      ! temperature (climatology) [K]
  // @param[in] linoz_o3col_clim  ! Column O3 above box (climatology) [Dobson
  // Units or DU]
  // @param[in] linoz_PmL_clim    ! P minus L (climatology) [vmr/s]
  // @param[in] linoz_dPmL_dT     ! Sensitivity of P minus L to T [K]
  // @param[in] linoz_dPmL_dO3    ! Sensitivity of P minus L to O3 [1/s]
  // @param[in] linoz_dPmL_dO3col ! Sensitivity of P minus L to overhead O3
  // column [vmr/DU]

  // @param[out] ss_o3
  // compute differences from climatology
  const Real delta_temperature = temperature - linoz_t_clim;
  const Real delta_o3col = o3col_du - linoz_o3col_clim;

  // steady state ozone
  ss_o3 = o3_clim - (linoz_PmL_clim + delta_o3col * linoz_dPmL_dO3col +
                     delta_temperature * linoz_dPmL_dT) /
                        linoz_dPmL_dO3;
} // compute_steady_state_ozone

KOKKOS_INLINE_FUNCTION
void lin_strat_chem_solve_kk(const Real o3col, const Real temperature,
                             const Real sza, const Real pmid,
                             const Real delta_t, const Real rlats,
                             // ltrop, & !in
                             const Real linoz_o3_clim, const Real linoz_t_clim,
                             const Real linoz_o3col_clim,
                             const Real linoz_PmL_clim,
                             const Real linoz_dPmL_dO3,
                             const Real linoz_dPmL_dT, // in
                             const Real linoz_dPmL_dO3col,
                             const Real linoz_cariolle_psc, // in
                             //
                             const Real chlorine_loading,
                             const Real psc_T, // PSC ozone loss T (K) threshold
                             Real &o3_vmr,
                             // diagnostic variables outputs
                             Real &do3_linoz, Real &do3_linoz_psc, Real &ss_o3,
                             Real &o3col_du_diag, Real &o3clim_linoz_diag,
                             Real &sza_degrees) {
  // @param[in] o3col               ! ozone column above box [mol/cm^2]
  // @param[in] temp                ! temperature [K]
  // @param[in] sza                 ! local solar zenith angle
  // @param[in] pmid                ! midpoint pressure [Pa]
  // @param[in] delta_t             ! timestep size [secs]
  // @param[in] rlats(ncol)         ! column latitudes (radians)
  // @param[in] ltrop               ! vertical index of the tropopause
  // @param[in] linoz_o3_clim       ! ozone (climatology) [vmr]
  // @param[in]linoz_t_clim         ! temperature (climatology) [K]
  // @param[in]linoz_o3col_clim     ! Column O3 above box (climatology) [Dobson
  // Units or DU]
  // @param[in] linoz_PmL_clim      ! P minus L (climatology) [vmr/s]
  // @param[in] linoz_dPmL_dO3      ! Sensitivity of P minus L to O3 [1/s]
  // @param[in] linoz_dPmL_dT       ! Sensitivity of P minus L to T [K]
  // @param[in]linoz_dPmL_dO3col    ! Sensitivity of P minus L to overhead O3
  // column [vmr/DU]
  // @param[in] linoz_cariolle_psc  ! Cariolle parameter for PSC loss of ozone
  // [1/s]

  // intent in-out
  // @param[in/out] o3_vmr              ! ozone volume mixing ratio [vmr]

  // n_ltropl trop(icol)
  // linoz_o3_clim
  // LOOP_COL: do icol = 1, ncol
  constexpr Real chlorine_loading_bgnd =
      0.0000; // EESC value [ppbv] for background conditions
  // BAD CONSTANT
  constexpr Real convert_to_du =
      1.0 / 2.687e16; //      ! convert ozone column from [mol/cm^2] to [DU]
  constexpr Real one = 1;
  constexpr Real zero = 0;
  const Real lats =
      rlats * radians_to_degrees; // ! convert lats from radians to degrees

  // !is there more than the background chlorine?
  const bool excess_chlorine =
      (chlorine_loading - chlorine_loading_bgnd) > zero;

  const Real o3_clim = linoz_o3_clim; // climatological ozone
  o3clim_linoz_diag = o3_clim;        //  diagnostic for output
  const Real o3_old = o3_vmr;         //              old ozone mixing ratio
  const Real o3col_du = o3col * convert_to_du; //  convert o3col from mol/cm2
  o3col_du_diag = o3col_du;                    //       update diagnostic output

  // compute steady state ozone
  compute_steady_state_ozone(o3_clim, o3col_du, temperature,    // in
                             linoz_t_clim, linoz_o3col_clim,    // in
                             linoz_PmL_clim, linoz_dPmL_dT,     // in
                             linoz_dPmL_dO3, linoz_dPmL_dO3col, // in
                             ss_o3);                            // out

  const Real delta_o3 =
      (ss_o3 - o3_old) *
      (one - haero::exp(linoz_dPmL_dO3 * delta_t)); //  ozone change

  Real o3_new = o3_old + delta_o3; // define new ozone mixing ratio

  do3_linoz = delta_o3 / delta_t; // output diagnostic

  // PSC activation (follows Cariolle et al 1990.)
  // used only if abs(latitude) > lats_threshold
  psc_activation(lats, temperature, pmid, sza, linoz_cariolle_psc,
                 delta_t,                                         // in
                 excess_chlorine, o3_old,                         // in
                 psc_T, chlorine_loading, o3_new, do3_linoz_psc); // out

  o3_vmr = o3_new; // update ozone vmr

  sza_degrees = sza * radians_to_degrees;

} // lin_strat_chem_solve_kk

KOKKOS_INLINE_FUNCTION
void lin_strat_chem_solve(
    const ThreadTeam &team, const ColumnView &o3col,
    const ColumnView &temperature, const Real sza, const ColumnView &pmid,
    const Real delta_t, const Real rlats,
    // ltrop, & !in
    const ColumnView &linoz_o3_clim, const ColumnView &linoz_t_clim,
    const ColumnView &linoz_o3col_clim, const ColumnView &linoz_PmL_clim,
    const ColumnView &linoz_dPmL_dO3,
    const ColumnView &linoz_dPmL_dT, // in
    const ColumnView &linoz_dPmL_dO3col,
    const ColumnView &linoz_cariolle_psc, // in
    const int ltrop, const Real chlorine_loading,
    const Real psc_T, // PSC ozone loss T (K) threshold
    const ColumnView &o3_vmr,
    // diagnostic variables outputs
    const ColumnView &do3_linoz, const ColumnView &do3_linoz_psc,
    const ColumnView &ss_o3, const ColumnView &o3col_du_diag,
    const ColumnView &o3clim_linoz_diag, const ColumnView &sza_degrees) {

  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team, ltrop), KOKKOS_LAMBDA(int kk) {
        lin_strat_chem_solve_kk(
            o3col(kk), temperature(kk), sza, pmid(kk), delta_t, rlats,
            // ltrop, & !in
            linoz_o3_clim(kk), linoz_t_clim(kk), linoz_o3col_clim(kk),
            linoz_PmL_clim(kk), linoz_dPmL_dO3(kk),
            linoz_dPmL_dT(kk), // in
            linoz_dPmL_dO3col(kk),
            linoz_cariolle_psc(kk), // in
            chlorine_loading,
            psc_T, // PSC ozone loss T (K) threshold
            o3_vmr(kk),
            // diagnostic variables outputs
            do3_linoz(kk), do3_linoz_psc(kk), ss_o3(kk), o3col_du_diag(kk),
            o3clim_linoz_diag(kk), sza_degrees(kk));
      });

} // lin_strat_chem_solve

KOKKOS_INLINE_FUNCTION
void lin_strat_sfcsink_kk(const Real delta_t, const Real pdel, // in
                          Real &o3l_vmr, const Real o3_sfc, const Real o3_tau,
                          Real &do3mass) {
  constexpr Real one = 1.0;
  // BAD CONSTANT
  constexpr Real mwo3 = 48.; // molecular weight O3
  constexpr Real mwdry = haero::Constants::molec_weight_dry_air *
                         1e3; //     ! molecular weight dry air ~ kg/kmole;//!
  constexpr Real rgrav =
      one / haero::Constants::gravity; // reciprocal of gravit
  const Real efactor =
      (one - haero::exp(-delta_t / o3_tau));     // !compute time scale factor
                                                 //
  const Real mass = pdel * rgrav;                //   air mass in kg/m2
  const Real o3l_old = o3l_vmr;                  // vmr
  const Real do3 = (o3_sfc - o3l_old) * efactor; // vmr
  o3l_vmr = o3l_old + do3;
  do3mass += do3 * mass * mwo3 / mwdry; // loss in kg/m2 summed over boundary
                                        // layers within one time step
}

KOKKOS_INLINE_FUNCTION
void lin_strat_sfcsink(const Real delta_t, const ColumnView &pdel, // in
                       const ColumnView &o3l_vmr, const Real o3_sfc,
                       const int o3_lbl, const Real o3_tau, Real &o3l_sfcsink) {

  // @param[in] delta_t              timestep size [secs]
  // @param[in] pdel(ncol,pver)      pressure delta about midpoints [Pa]

  // !inten in-outs
  // @param[inout]:: o3l_vmr(ncol ,pver)             ! ozone volume mixing ratio
  // [vmr]

  Real do3mass_icol = 0;
  for (int kk = pver - 1; kk > pver - o3_lbl - 1; --kk) {
    lin_strat_sfcsink_kk(delta_t, pdel(kk), o3l_vmr(kk), o3_sfc, o3_tau,
                         do3mass_icol);
  }

  // Two parameters are applied to Linoz O3 for surface sink, O3l is not coupled
  // to real O3
  constexpr Real KgtoTg = 1.0e-9; //
  constexpr Real peryear =
      86400. * 365.0; // ! to multiply to convert per second to per year

  o3l_sfcsink =
      do3mass_icol / delta_t * KgtoTg * peryear; // ! saved in Tg/yr/m2 unit

} // lin_strat_sfcsink

} // namespace lin_strat_chem

} // end namespace mam4

#endif
