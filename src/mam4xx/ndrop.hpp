#ifndef MAM4XX_NDROP_HPP
#define MAM4XX_NDROP_HPP

#include <ekat/util/ekat_math_utils.hpp>

#include <haero/atmosphere.hpp>
#include <haero/math.hpp>

#include <mam4xx/aero_config.hpp>
#include <mam4xx/conversions.hpp>
#include <mam4xx/mam4_types.hpp>
#include <mam4xx/ndrop.hpp>
#include <mam4xx/utils.hpp>
#include <mam4xx/wv_sat_methods.hpp>

namespace mam4 {

namespace ndrop {

const int pver = 72;
const int top_lev = 7;
const int psat = 6; //  ! number of supersaturations to calc ccn concentration
// FIXME: ask about state_q. Is it a prognostic variable?
const int nvars = 40;
const int maxd_aspectype = 14;
// BAD CONSTANT
const Real t0 = 273;       // reference temperature [K]
const Real p0 = 1013.25e2; //  ! reference pressure [Pa]
const int nvar_ptend_q = 40;
// FIXME; surften is defined in ndrop_init
// BAD CONSTANT
const Real surften = 0.076;
const Real sq2 = haero::sqrt(2.);
const Real smcoefcoef = 2. / haero::sqrt(27.);
const int ncnst_tot = 25;
//// const Real percent_to_fraction = 0.01;
// super(:)=supersat(:)*percent_to_fraction
// supersaturation [fraction]
const Real super[psat] = {
    0.0002, 0.0005, 0.001, 0.002,
    0.005,  0.01}; //& ! supersaturation (%) to determine ccn concentration
//  [m-K]
const int nspec_max = 8;

KOKKOS_INLINE_FUNCTION
void get_aer_mmr_sum(
    const int imode, const int nspec, const Real state_q[nvars],
    const Real qcldbrn1d[maxd_aspectype],
    const int lspectype_amode[maxd_aspectype][AeroConfig::num_modes()],
    const Real specdens_amode[maxd_aspectype],
    const Real spechygro[maxd_aspectype],
    const int lmassptr_amode[maxd_aspectype][AeroConfig::num_modes()],
    Real &vaerosolsum_icol, Real &hygrosum_icol) {
  // add these for direct access to mmr (in state_q array), density and
  // hygroscopicity use modal_aero_data,   only: lspectype_amode,
  // specdens_amode, spechygro, lmassptr_amode
  const Real zero = 0;
  // ! sum to find volume conc [m3/kg]
  vaerosolsum_icol = zero;
  // ! sum to bulk hygroscopicity of mode [m3/kg]
  hygrosum_icol = zero;
  // @param[in] imode  mode index
  // @param[in] nspec  total # of species in mode imode
  // @param[in] state_q interstitial aerosol mass mixing ratios [kg/kg]
  // @param[in] qcldbrn1d ! cloud-borne aerosol mass mixing ratios [kg/kg]

  // !Start to compute bulk volume conc / hygroscopicity by summing over species
  // per mode.
  for (int lspec = 0; lspec < nspec; ++lspec) {
    // Fortran indexing to C++
    const int type_idx = lspectype_amode[lspec][imode] - 1;
    // density at species / mode indices [kg/m3]
    const Real density_sp = specdens_amode[type_idx]; //! species density
    // hygroscopicity at species / mode indices [dimensionless]
    const Real hygro_sp = spechygro[type_idx]; // !species hygroscopicity
    // Fortran indexing to C++
    const int spc_idx =
        lmassptr_amode[lspec][imode] - 1; //! index of species in state_q array
    // !aerosol volume mixing ratio [m3/kg]
    const Real vol = haero::max(state_q[spc_idx] + qcldbrn1d[lspec], zero) /
                     density_sp; // !volume = mmr/density
    vaerosolsum_icol += vol;
    hygrosum_icol += vol * hygro_sp; // !bulk hygroscopicity
  }                                  // end

} // end get_aer_mmr_sum

KOKKOS_INLINE_FUNCTION
void get_aer_num(const Real voltonumbhi_amode, const Real voltonumblo_amode,
                 const int num_idx, const Real state_q[nvars],
                 const Real air_density, const Real vaerosol,
                 const Real qcldbrn1d_num, Real &naerosol) {

  // input arguments
  // @param[in] imode        ! mode index
  // @param[in] state_q(:,:) ! interstitial aerosol number mixing ratios [#/kg]
  // @param[in] air_density        ! air density [kg/m3]
  // @param[in] vaerosol(:)  ! volume conc [m3/m3]
  // @param[in] qcldbrn1d_num(:) ! cloud-borne aerosol number mixing ratios
  // [#/kg]
  // @param[out] naerosol(:)  ! number conc [#/m3]

  // convert number mixing ratios to number concentrations
  // Use bulk volume conc found previously to bound value
  // or contains both species and modes concentrations ?
  naerosol = (state_q[num_idx] + qcldbrn1d_num) * air_density;

  //! adjust number so that dgnumlo < dgnum < dgnumhi
  naerosol = utils::min_max_bound(vaerosol * voltonumbhi_amode,
                                  vaerosol * voltonumblo_amode, naerosol);

} // end get_aer_num

// calculates maximum supersaturation for multiple
// competing aerosol modes.
// Abdul-Razzak and Ghan, A parameterization of aerosol activation.
// 2. Multiple aerosol types. J. Geophys. Res., 105, 6837-6844.
KOKKOS_INLINE_FUNCTION
void maxsat(
    const Real zeta,                         // [dimensionless]
    const Real eta[AeroConfig::num_modes()], // [dimensionless]
    const Real nmode,                        // number of modes
    const Real smc[AeroConfig::num_modes()], // critical supersaturation for
                                             // number mode radius [fraction]
    Real &smax // maximum supersaturation [fraction] (output)
) {
  // abdul-razzak functions of width
  Real f1[AeroConfig::num_modes()];
  Real f2[AeroConfig::num_modes()];

  Real const small = 1e-20;     /*FIXME: BAD CONSTANT*/
  Real const mid = 1e5;         /*FIXME: BAD CONSTANT*/
  Real const big = 1.0 / small; /*FIXME: BAD CONSTANT*/
  Real sum = 0;
  Real g1, g2;
  bool weak_forcing = true; // whether forcing is sufficiently weak or not

  for (int m = 0; m < nmode; m++) {
    if (zeta > mid * eta[m] || smc[m] * smc[m] > mid * eta[m]) {
      // weak forcing. essentially none activated
      smax = small;
    } else {
      // significant activation of this mode. calc activation of all modes.
      weak_forcing = false;
      break;
    }
  }

  // if the forcing is weak, return
  if (weak_forcing)
    return;

  for (int m = 0; m < nmode; m++) {
    f1[m] = 0.5 *
            haero::exp(2.5 * haero::square(haero::log(modes(m).mean_std_dev)));
    f2[m] = 1.0 + 0.25 * haero::log(modes(m).mean_std_dev);
    if (eta[m] > small) {
      g1 = (zeta / eta[m]) * haero::sqrt(zeta / eta[m]);
      g2 = (smc[m] / haero::sqrt(eta[m] + 3.0 * zeta)) *
           haero::sqrt(smc[m] / haero::sqrt(eta[m] + 3.0 * zeta));
      sum += (f1[m] * g1 + f2[m] * g2) / (smc[m] * smc[m]);
    } else {
      sum = big;
    }
  }
  smax = 1.0 / haero::sqrt(sum);
  return;
} // end maxsat

KOKKOS_INLINE_FUNCTION
void loadaer(const Real state_q[nvars],
             const int nspec_amode[AeroConfig::num_modes()], Real air_density,
             const int phase,
             const int lspectype_amode[maxd_aspectype][AeroConfig::num_modes()],
             const Real specdens_amode[maxd_aspectype],
             const Real spechygro[maxd_aspectype],
             const int lmassptr_amode[maxd_aspectype][AeroConfig::num_modes()],
             const Real voltonumbhi_amode[AeroConfig::num_modes()],
             const Real voltonumblo_amode[AeroConfig::num_modes()],
             const int numptr_amode[AeroConfig::num_modes()],
             const Real qcldbrn1d[maxd_aspectype][AeroConfig::num_modes()],
             const Real qcldbrn1d_num[AeroConfig::num_modes()],
             Real naerosol[AeroConfig::num_modes()],
             Real vaerosol[AeroConfig::num_modes()],
             Real hygro[AeroConfig::num_modes()]) {
  // aerosol number, volume concentrations, and bulk hygroscopicity at one
  // specific column and level aerosol mmrs [kg/kg]

  // input arguments
  // @param [in] state_q(:)         aerosol mmrs [kg/kg]
  // @param [in] air_density        air density [kg/m3]
  // @param [in]  phase     phase of aerosol: 1 for interstitial, 2 for
  // cloud-borne, 3 for sum

  // output arguments
  // @param [out]  naerosol(ntot_amode)  ! number conc [#/m3]
  // @param [out]  vaerosol(ntot_amode)  ! volume conc [m3/m3]
  // @param [out]  hygro(ntot_amode)     ! bulk hygroscopicity of mode
  // [dimensionless]

  // optional input arguments; we assumed that this inputs is not optional
  // @param [in] qcldbrn1d(:,:), qcldbrn1d_num(:)  cloud-borne aerosol mass /
  // number mixing ratios [kg/kg or #/kg]
  // !Currenly supports only phase 1 (interstitial) and 3 (interstitial+cldbrn)

  if (phase != 1 && phase != 3) {
    // write(iulog,*)'phase=',phase,' in loadaer'
    Kokkos::abort("phase error in loadaer");
  }

  const Real nmodes = AeroConfig::num_modes();
  // BAD CONSTANT
  const Real even_smaller_val = 1.0e-30;

  // assume is present
  // qcldbrn_local(:,:nspec) = qcldbrn1d(:,:nspec)

  // Sum over all species within imode to get bulk hygroscopicity and volume
  // conc phase == 1 is interstitial only. phase == 3 is interstitial + cldborne
  // Assumes iphase =1 or 3, so interstitial is always summed, added with cldbrn
  // when present iphase = 2 would require alternate logic from following
  // subroutine

  const Real zero = 0;
  // FIXME mam4 arrays shape
  // const Real nspec = AeroConfig::num_aerosol_ids();
  Real qcldbrn1d_imode[AeroConfig::num_aerosol_ids()] = {};
  // FIXME number of species
  for (int imode = 0; imode < nmodes; ++imode) {
    Real vaerosolsum = zero;
    Real hygrosum = zero;
    const Real nspec = nspec_amode[imode];

    for (int ispec = 0; ispec < nspec; ++ispec) {
      qcldbrn1d_imode[ispec] = qcldbrn1d[ispec][imode];
    }

    get_aer_mmr_sum(imode, nspec, state_q, qcldbrn1d_imode, lspectype_amode,
                    specdens_amode, spechygro, lmassptr_amode, vaerosolsum,
                    hygrosum);

    //  Finalize computation of bulk hygrospopicity and volume conc
    if (vaerosolsum > even_smaller_val) {
      hygro[imode] = hygrosum / vaerosolsum;
      vaerosol[imode] = vaerosolsum * air_density;
    } else {
      hygro[imode] = zero;
      vaerosol[imode] = zero;
    } //

    // ! Compute aerosol number concentration
    // Fortran indexing to C++
    const int num_idx = numptr_amode[imode] - 1;
    get_aer_num(voltonumbhi_amode[imode], voltonumblo_amode[imode], num_idx,
                state_q, air_density, vaerosol[imode], qcldbrn1d_num[imode],
                naerosol[imode]);

  } // end imode

} // loadaer

KOKKOS_INLINE_FUNCTION
void ccncalc(const Real state_q[nvars], const Real tair,
             const Real qcldbrn[maxd_aspectype][AeroConfig::num_modes()],
             const Real qcldbrn_num[AeroConfig::num_modes()],
             const Real air_density,
             const int lspectype_amode[maxd_aspectype][AeroConfig::num_modes()],
             const Real specdens_amode[maxd_aspectype],
             const Real spechygro[maxd_aspectype],
             const int lmassptr_amode[maxd_aspectype][AeroConfig::num_modes()],
             const Real voltonumbhi_amode[AeroConfig::num_modes()],
             const Real voltonumblo_amode[AeroConfig::num_modes()],
             const int numptr_amode[AeroConfig::num_modes()],
             const int nspec_amode[AeroConfig::num_modes()],
             const Real exp45logsig[AeroConfig::num_modes()],
             const Real alogsig[AeroConfig::num_modes()], Real ccn[psat]) {

  // calculates number concentration of aerosols activated as CCN at
  // supersaturation supersat.
  // assumes an internal mixture of a multiple externally-mixed aerosol modes
  // cgs units

  // Ghan et al., Atmos. Res., 1993, 198-221.

  // input arguments
  // @param [in] state_q(:,:,:) ! aerosol mmrs [kg/kg]
  // @param [in] tair(:,:)     ! air temperature [K]
  // @param [in] qcldbrn(:,:,:,:), qcldbrn_num(:,:,:) ! cloud-borne aerosol mass
  // / number  mixing ratios [kg/kg or #/kg]
  // @param [in] air_density(pcols,pver)       ! air density [kg/m3]

  // output arguments
  // ccn(pcols,pver,psat) ! number conc of aerosols activated at supersat [#/m3]
  // qcldbrn  1) icol 2) ispec 3) kk 4)imode
  // state_q 1) icol 2) kk 3) imode and ispec
  const Real zero = 0;
  // BAD CONSTANT
  const Real nconc_thresh = 1.e-3;
  const Real twothird = 2. / 3.;
  const Real two = 2.;
  const Real three_fourths = 3. / 4.;
  const Real half = 0.5;
  const Real one = 1;

  const Real per_m3_to_per_cm3 = 1.e-6;
  // phase of aerosol
  const int phase = 3; // ! interstitial+cloudborne
  const int nmodes = AeroConfig::num_modes();

  const Real mwh2o = haero::Constants::molec_weight_h2o * 1e3; // [kg/kmol]
  const Real r_universal = haero::Constants::r_gas * 1e3;      //[J/K/kmole]
  const Real rhoh2o = haero::Constants::density_h2o;
  const Real pi = haero::Constants::pi;

  const Real surften_coef = two * mwh2o * surften / (r_universal * rhoh2o);

  // surface tension parameter  [m]
  const Real aparam = surften_coef / tair;
  const Real smcoef = smcoefcoef * aparam * haero::sqrt(aparam); //[m^(3/2)]

  Real naerosol[nmodes] = {
      zero}; // ! interstit+activated aerosol number conc [#/m3]
  Real vaerosol[nmodes] = {
      zero}; // ! interstit+activated aerosol volume conc [m3/m3]
  Real hygro[nmodes] = {zero};

  loadaer(state_q, nspec_amode, air_density, phase, lspectype_amode,
          specdens_amode, spechygro, lmassptr_amode, voltonumbhi_amode,
          voltonumblo_amode, numptr_amode, qcldbrn, qcldbrn_num, naerosol,
          vaerosol, hygro);

  for (int lsat = 0; lsat < psat; ++lsat) {
    ccn[lsat] = {zero};
  }
  for (int imode = 0; imode < nmodes; ++imode) {
    // here we assume that value shouldn't matter much since naerosol is small
    // endwhere
    Real sm = 1; // critical supersaturation at mode radius [fraction]
    if (naerosol[imode] > nconc_thresh) {
      // [dimensionless]
      const Real amcubecoef_imode = three_fourths / (pi * exp45logsig[imode]);
      // [m3]
      const Real amcube = amcubecoef_imode * vaerosol[imode] / naerosol[imode];
      sm = smcoef /
           haero::sqrt(hygro[imode] * amcube); // ! critical supersaturation
    }

    const Real argfactor_imode = twothird / (sq2 * alogsig[imode]);
    for (int lsat = 0; lsat < psat; ++lsat) {
      // [dimensionless]
      const Real arg_erf_ccn = argfactor_imode * haero::log(sm / super[lsat]);
      ccn[lsat] += naerosol[imode] * half * (one - haero::erf(arg_erf_ccn));
    }

  } // imode end

  for (int lsat = 0; lsat < psat; ++lsat) {
    ccn[lsat] *= per_m3_to_per_cm3; // ! convert from #/m3 to #/cm3
  }                                 // lsat

} /// ccncalc

KOKKOS_INLINE_FUNCTION
void qsat(const Real t, const Real p, Real &es, Real &qs) {
  //  ------------------------------------------------------------------!
  // ! Purpose:                                                         !
  // !   Look up and return saturation vapor pressure from precomputed  !
  // !   table, then calculate and return saturation specific humidity. !
  // !   Optionally return various temperature derivatives or enthalpy  !
  // !   at saturation.                                                 !
  // !------------------------------------------------------------------!
  // Inputs
  // @param [in] t    ! Temperature
  // @param [in] p    ! Pressure
  // Outputs
  // @param [out] es  ! Saturation vapor pressure
  // @param [out] qs  ! Saturation specific humidity

  // Note. Fortran code uses a table lookup. In C++ version, we compute directly
  // from the function.
  es = wv_sat_methods::wv_sat_svp_trans(t);
  qs = wv_sat_methods::wv_sat_svp_to_qsat(es, p);
  // Ensures returned es is consistent with limiters on qs.
  es = haero::min(es, p);
} // qsat

inline void ndrop_int(Real exp45logsig[AeroConfig::num_modes()],
                      Real alogsig[AeroConfig::num_modes()], Real &aten,
                      Real num2vol_ratio_min_nmodes[AeroConfig::num_modes()],
                      Real num2vol_ratio_max_nmodes[AeroConfig::num_modes()]) {
  const Real one = 1;
  const Real two = 2;
  const Real one_thousand = 1e3;
  for (int imode = 0; imode < AeroConfig::num_modes(); ++imode) {
    alogsig[imode] = haero::log(modes(imode).mean_std_dev);
    exp45logsig[imode] = haero::exp(4.5 * alogsig[imode] * alogsig[imode]);

    // voltonumbhi_amode
    num2vol_ratio_min_nmodes[imode] =
        one / conversions::mean_particle_volume_from_diameter(
                  modes(imode).max_diameter, modes(imode).mean_std_dev);
    // voltonumblo_amode
    num2vol_ratio_max_nmodes[imode] =
        one / conversions::mean_particle_volume_from_diameter(
                  modes(imode).min_diameter, modes(imode).mean_std_dev);

  } // imode

  // SHR_CONST_RHOFW   = 1.000e3_R8      ! density of fresh water     ~ kg/m^3
  const Real rhoh2o = haero::Constants::density_h2o;
  const Real r_universal = haero::Constants::r_gas * one_thousand; //[J/K/kmole]
  const Real mwh2o =
      haero::Constants::molec_weight_h2o * one_thousand; // [kg/kmol]
  // BAD CONSTANT
  aten = two * mwh2o * surften / (r_universal * t0 * rhoh2o);

} // end ndrop_int

KOKKOS_INLINE_FUNCTION
void activate_modal(const Real w_in, const Real wmaxf, const Real tair,
                    const Real rhoair, Real na[AeroConfig::num_modes()],
                    const Real volume[AeroConfig::num_modes()],
                    const Real hygro[AeroConfig::num_modes()],
                    const Real exp45logsig[AeroConfig::num_modes()],
                    const Real alogsig[AeroConfig::num_modes()],
                    const Real aten, Real fn[AeroConfig::num_modes()],
                    Real fm[AeroConfig::num_modes()],
                    Real fluxn[AeroConfig::num_modes()],
                    Real fluxm[AeroConfig::num_modes()], Real &flux_fullact) {
  // 	  !---------------------------------------------------------------------------------
  // !Calculates number, surface, and mass fraction of aerosols activated as CCN
  // !calculates flux of cloud droplets, surface area, and aerosol mass into
  // cloud !assumes an internal mixture within each of up to nmode multiple
  // aerosol modes !a gaussiam spectrum of updrafts can be treated.
  // !
  // !Units: SI (MKS)
  // !
  // !Reference: Abdul-Razzak and Ghan, A parameterization of aerosol
  // activation. !      2. Multiple aerosol types. J. Geophys. Res., 105,
  // 6837-6844.
  // !---------------------------------------------------------------------------------

  // input
  // @param [in] w_in      ! vertical velocity [m/s]
  // @param [in] wmaxf     ! maximum updraft velocity for integration [m/s]
  // @param [in] tair      ! air temperature [K]
  // @param [in] rhoair    ! air density [kg/m3]
  // @param [in] na(:)     ! aerosol number concentration [#/m3]
  // @param [in] nmode     ! number of aerosol modes
  // @param [in] volume(:) ! aerosol volume concentration [m3/m3]
  // @param [in] hygro(:)  ! hygroscopicity of aerosol mode [dimensionless]
  // @param [in] smax_prescribed  ! prescribed max. supersaturation for
  // secondary activation [fraction]

  // !output
  // @param [out] fn(:)        ! number fraction of aerosols activated
  // [fraction]
  // @param [out] fm(:)        ! mass fraction of aerosols activated [fraction]
  // @param [out] fluxn(:)     ! flux of activated aerosol number fraction into
  // cloud [m/s]
  // @param [out] fluxm(:)     ! flux of activated aerosol mass fraction into
  // cloud [m/s]
  // @param [out] flux_fullact ! flux of activated aerosol fraction assuming
  // 100% activation [m/s]

  // !---------------------------------------------------------------------------------
  // ! flux_fullact is used for consistency check -- this should match
  // (ekd(k)*zs(k)) ! also, fluxm/flux_fullact gives fraction of aerosol mass
  // flux ! that is activated
  // !---------------------------------------------------------------------------------
  const Real zero = 0;
  const Real one = 1;
  const Real sq2 = haero::sqrt(2.);
  const Real two = 2;
  const Real three_fourths = 3. / 4.;
  const Real twothird = 2. / 3.;
  const Real half = 0.5;
  const Real small = 1.e-39;

  constexpr int nmode = AeroConfig::num_modes();
  // BAD CONSTANT
  // return if aerosol number is negligible in the accumulation mode
  // FIXME use index of accumulation mode
  if (na[0] < 1.e-20) {
    return;
  }

  // return if vertical velocity is 0 or negative
  if (w_in <= zero) {
    return;
  }

  //
  // !return if max supersaturation is 0 or negative
  // if (smax_prescribed <= zero) {return;};

  // BAD CONSTANT
  //  FIXME look for constant in ahero
  //  const Real SHR_CONST_MWDAIR  = 28.966;//       ! molecular weight dry air
  //  ~ kg/kmole const Real SHR_CONST_MWWV    = 18.016;//       ! molecular
  //  weight water vapor const Real  rair = SHR_CONST_RGAS/SHR_CONST_MWDAIR;//
  //  ! Dry air gas constant     ~ J/K/kg

  const Real rair = haero::Constants::r_gas_dry_air;
  const Real rh2o = haero::Constants::r_gas_h2o_vapor;
  const Real latvap =
      haero::Constants::latent_heat_evap; //      ! latent heat of evaporation ~
                                          //      J/kg
  const Real cpair =
      haero::Constants::cp_dry_air; //    ! specific heat of dry air   ~ J/kg/K
  const Real gravit =
      haero::Constants::gravity; //      ! acceleration of gravity ~ m/s^2
  // SHR_CONST_RHOFW   = 1.000e3_R8      ! density of fresh water     ~ kg/m^3
  const Real rhoh2o = haero::Constants::density_h2o;
  const Real pi = haero::Constants::pi;

  const Real pres = rair * rhoair * tair; // pressure [Pa]
  // Obtain Saturation vapor pressure (es) and saturation specific humidity (qs)
  //  FIXME: check if we have implemented qsat
  //  water vapor saturation specific humidity [kg/kg]
  Real qs = zero;
  // ! saturation vapor pressure [Pa]
  Real es = zero;           //
  qsat(tair, pres, es, qs); // !es and qs are the outputs
  // ! change in qs with temperature  [(kg/kg)/T]
  const Real dqsdt = latvap / (rh2o * tair * tair) * qs;
  // [/m]
  const Real alpha =
      gravit * (latvap / (cpair * rh2o * tair * tair) - one / (rair * tair));
  // [m3/kg]
  const Real gamma = (one + latvap / cpair * dqsdt) / (rhoair * qs);
  // [s^(3/2)]
  // BAD CONSTANT
  const Real etafactor2max =
      1.e10 / haero::pow((alpha * wmaxf),
                         1.5); // !this should make eta big if na is very small.
  // vapor diffusivity [m2/s]
  const Real diff0 = 0.211e-4 * (p0 / pres) * haero::pow(tair / t0, 1.94);
  // ! thermal conductivity [J / (m-s-K)]
  const Real conduct0 =
      (5.69 + 0.017 * (tair - t0)) * 4.186e2 * 1.e-5; // !convert to J/m/s/deg
  // thermodynamic function [m2/s]
  const Real gthermfac =
      one /
      (rhoh2o / (diff0 * rhoair * qs) +
       latvap * rhoh2o / (conduct0 * tair) *
           (latvap / (rh2o * tair) - one)); // gthermfac is same for all modes
  const Real beta = two * pi * rhoh2o * gthermfac * gamma; //[m2/s]
  // nucleation w, but = w_in if wdiab == 0 [m/s]
  const Real wnuc = w_in;
  const Real alw = alpha * wnuc;                  // [/s]
  const Real etafactor1 = alw * haero::sqrt(alw); // [/ s^(3/2)]
  // [unitless]
  const Real zeta = twothird * haero::sqrt(alw) * aten / haero::sqrt(gthermfac);

  Real amcube[nmode] = {}; // ! cube of dry mode radius [m3]

  Real etafactor2[nmode] = {};
  Real lnsm[nmode] = {};

  // critical supersaturation for number mode radius [fraction]
  Real smc[nmode] = {};

  Real eta[nmode] = {};
  // !Here compute smc, eta for all modes for maxsat calculation
  for (int imode = 0; imode < nmode; ++imode) {
    // BAD CONSTANT

    if (volume[imode] > small && na[imode] > small) {
      // !number mode radius (m)
      amcube[imode] =
          three_fourths * volume[imode] /
          (pi * exp45logsig[imode] * na[imode]); // ! only if variable size dist
      // !Growth coefficent Abdul-Razzak & Ghan 1998 eqn 16
      // !should depend on mean radius of mode to account for gas kinetic
      // effects !see Fountoukis and Nenes, JGR2005 and Meskhidze et al.,
      // JGR2006 !for approriate size to use for effective diffusivity.
      etafactor2[imode] = one / (na[imode] * beta * haero::sqrt(gthermfac));
      // BAD CONSTANT
      if (hygro[imode] > 1.e-10) {
        smc[imode] =
            two * aten *
            haero::sqrt(aten / (27. * hygro[imode] *
                                amcube[imode])); // ! only if variable size dist
      } else {
        // BAD CONSTANT
        smc[imode] = 100.;
      } // hygro
    } else {
      smc[imode] = one;
      etafactor2[imode] =
          etafactor2max; // ! this should make eta big if na is very small.
    }                    // volumne
    lnsm[imode] = haero::log(smc[imode]); // ! only if variable size dist
    eta[imode] = etafactor1 * etafactor2[imode];
  } // end imode

  // Find maximum supersaturation
  // Use smax_prescribed if it is present; otherwise get smax from subr maxsat
  // if ( present( smax_prescribed ) ) then
  //  maximum supersaturation [fraction]
  // const Real smax = smax_prescribed;
  // else
  Real smax = zero;

  maxsat(zeta, eta, nmode, smc, smax);
  // FIXME [unitless] ? lnsmax maybe has units of log(unit of smax ([fraction]))
  const Real lnsmax = haero::log(smax);

  // !Use maximum supersaturation to calculate aerosol activation output
  for (int imode = 0; imode < nmode; ++imode) {
    // ! [unitless]
    const Real arg_erf_n =
        twothird * (lnsm[imode] - lnsmax) / (sq2 * alogsig[imode]);

    fn[imode] = half * (one - haero::erf(arg_erf_n)); //! activated number

    const Real arg_erf_m = arg_erf_n - 1.5 * sq2 * alogsig[imode];
    fm[imode] = half * (one - haero::erf(arg_erf_m)); // !activated mass
    fluxn[imode] = fn[imode] * w_in; // !activated aerosol number flux
    fluxm[imode] = fm[imode] * w_in; // !activated aerosol mass flux
  }
  // FIXME: what is this??
  // is vertical velocity equal to flux of activated aerosol fraction assuming
  // 100% activation [m/s]?
  flux_fullact = w_in;

} // activate_modal

KOKKOS_INLINE_FUNCTION
void get_activate_frac(
    const Real state_q_kload[nvars], const Real air_density_kload,
    const Real air_density_kk, const Real wtke,
    const Real tair, // in
    const int lspectype_amode[maxd_aspectype][AeroConfig::num_modes()],
    const Real specdens_amode[maxd_aspectype],
    const Real spechygro[maxd_aspectype],
    const int lmassptr_amode[maxd_aspectype][AeroConfig::num_modes()],
    const Real voltonumbhi_amode[AeroConfig::num_modes()],
    const Real voltonumblo_amode[AeroConfig::num_modes()],
    const int numptr_amode[AeroConfig::num_modes()],
    const int nspec_amode[maxd_aspectype],
    const Real exp45logsig[AeroConfig::num_modes()],
    const Real alogsig[AeroConfig::num_modes()], const Real aten,
    Real fn[AeroConfig::num_modes()], Real fm[AeroConfig::num_modes()],
    Real fluxn[AeroConfig::num_modes()], Real fluxm[AeroConfig::num_modes()],
    Real &flux_fullact) {

  // input arguments
  //  @param [in] state_q_kload(:)         aerosol mmrs at level from which to
  //  load aerosol [kg/kg]
  //  @param [in] cs_kload     air density at level from which to load aerosol
  //  [kg/m3]
  //  @param [in] cs_kk        air density at actual vertical level [kg/m3]
  //  @param [in] wtke        subgrid vertical velocity [m/s]
  //  @param [in]  tair        ! air temperature [K]

  // output arguments
  // @param [out]  fn(:)        ! number fraction of aerosols activated
  // [fraction]
  // @param [out]   fm(:)        ! mass fraction of aerosols activated
  // [fraction]
  // @param [out]   fluxn(:)     ! flux of activated aerosol number fraction
  // into cloud [m/s]
  // @param [out]   fluxm(:)     ! flux of activated aerosol mass fraction into
  // cloud [m/s]
  // @param [out]   flux_fullact ! flux of activated aerosol fraction assuming
  // 100% activation [m/s]
  const Real zero = 0;
  const int nmodes = AeroConfig::num_modes();
  const int phase = 1; // ! interstitial

  const Real qcldbrn[maxd_aspectype][AeroConfig::num_modes()] = {{zero}};
  const Real qcldbrn_num[AeroConfig::num_modes()] = {zero};

  Real naermod[nmodes] = {zero};  // aerosol number concentration [#/m^3]
  Real vaerosol[nmodes] = {zero}; // aerosol volume conc [m^3/m^3]
  Real hygro[nmodes] = {zero}; // hygroscopicity of aerosol mode [dimensionless]

  // load aerosol properties, assuming external mixtures
  loadaer(state_q_kload, nspec_amode, air_density_kload, phase, lspectype_amode,
          specdens_amode, spechygro, lmassptr_amode, voltonumbhi_amode,
          voltonumblo_amode, numptr_amode, qcldbrn, qcldbrn_num, naermod,
          vaerosol, hygro);
  //
  // Below is to avoid warning about not assigning value to intent(out)
  //  the assignment should have no affect because flux_fullact is intent(out)
  //  in activate_modal (and in that subroutine is initialized to zero anyway).
  // BAD CONSTANT
  const Real wmax = 10.;
  activate_modal(wtke, wmax, tair, air_density_kk, //   ! in
                 naermod, vaerosol, hygro,         //  ! in
                 exp45logsig, alogsig, aten, fn, fm, fluxn, fluxm,
                 flux_fullact); // out

} // get_activate_frac

KOKKOS_INLINE_FUNCTION
void update_from_cldn_profile(
    const Real cldn_col_in, const Real cldn_col_in_kp1, const Real dtinv,
    const Real wtke_col_in, const Real zs,
    const Real dz, // ! in
    const Real temp_col_in, const Real air_density, const Real air_density_kp1,
    const Real csbot_cscen,
    const Real state_q_col_in_kp1[nvars], // ! in
    const int lspectype_amode[maxd_aspectype][AeroConfig::num_modes()],
    const Real specdens_amode[maxd_aspectype],
    const Real spechygro[maxd_aspectype],
    const int lmassptr_amode[maxd_aspectype][AeroConfig::num_modes()],
    const Real voltonumbhi_amode[AeroConfig::num_modes()],
    const Real voltonumblo_amode[AeroConfig::num_modes()],
    const int numptr_amode[AeroConfig::num_modes()],
    const int nspec_amode[maxd_aspectype],
    const Real exp45logsig[AeroConfig::num_modes()],
    const Real alogsig[AeroConfig::num_modes()], const Real aten,
    const int mam_idx[AeroConfig::num_modes()][nspec_max],
    Real raercol_nsav[ncnst_tot], const Real raercol_nsav_kp1[ncnst_tot],
    Real raercol_cw_nsav[ncnst_tot],
    Real &nsource_col, // inout
    Real &qcld, Real factnum_col[AeroConfig::num_modes()],
    Real &ekd, // out
    Real nact[AeroConfig::num_modes()], Real mact[AeroConfig::num_modes()]) {
  // input arguments
  // cldn_col_in(:)   ! cloud fraction [fraction] at kk
  // cldn_col_in_kp1(:)   ! cloud fraction [fraction] at min0(kk+1, pver);
  // dtinv      ! inverse time step for microphysics [s^{-1}]
  // wtke_col_in(:)   ! subgrid vertical velocity [m/s] at kk
  // zs(:)            ! inverse of distance between levels [m^-1]
  // dz(:)         ! geometric thickness of layers [m]
  // temp_col_in(:)   ! temperature [K]
  // air_density(:)     ! air density [kg/m^3] at kk
  // air_density_kp1  ! air density [kg/m^3] at min0(kk+1, pver);
  // csbot_cscen(:)   ! inverse normalized air density csbot(i)/cs(i,k)
  // [dimensionless] [dimensionless] state_q_col_in(:,:)    ! aerosol mmrs
  // [kg/kg]

  // raercol_nsav(:,:)    ! single column of saved aerosol mass, number mixing
  // ratios [#/kg or kg/kg] raercol_cw_nsav(:,:) ! same as raercol but for
  // cloud-borne phase [#/kg or kg/kg] nsource_col(:)   ! droplet number mixing
  // ratio source tendency [#/kg/s] qcld(:)  ! cloud droplet number mixing ratio
  // [#/kg] factnum_col(:,:)  ! activation fraction for aerosol number
  // [fraction] ekd(:)     ! diffusivity for droplets [m^2/s] nact(:,:)  !
  // fractional aero. number  activation rate [/s] mact(:,:)  ! fractional aero.
  // mass    activation rate [/s]

  // ! ......................................................................
  // ! start of k-loop for calc of old cloud activation tendencies ..........
  // !
  // ! rce-comment
  // !    changed this part of code to use current cloud fraction (cldn)
  // exclusively !    consider case of cldo(:)=0, cldn(k)=1, cldn(k+1)=0 !
  // previous code (which used cldo below here) would have no cloud-base
  // activation !       into layer k.  however, activated particles in k mix out
  // to k+1, !       so they are incorrectly depleted with no replacement
  // BAD CONSTANT
  const Real cld_thresh = 0.01; //   !  threshold cloud fraction [fraction]
  // kp1 = min0(kk+1, pver);
  const int ntot_amode = AeroConfig::num_modes();
  const Real zero = 0;

  if (cldn_col_in > cld_thresh) {

    if (cldn_col_in - cldn_col_in_kp1 > cld_thresh) {

      // cloud base

      // ! rce-comments
      // !   first, should probably have 1/zs(k) here rather than dz(i,k)
      // because !      the turbulent flux is proportional to ekd(k)*zs(k), !
      // while the dz(i,k) is used to get flux divergences !      and mixing
      // ratio tendency/change !   second and more importantly, using a single
      // updraft velocity here !      means having monodisperse turbulent
      // updraft and downdrafts. !      The sq2pi factor assumes a normal draft
      // spectrum. !      The fluxn/fluxm from activate must be consistent with
      // the !      fluxes calculated in explmix.
      ekd = wtke_col_in / zs;
      // ! rce-comment - use kp1 here as old-cloud activation involves
      // !   aerosol from layer below

      // Real fn[ntot_amode] = {};// activation fraction for aerosol number
      // [fraction]
      Real fm[ntot_amode] =
          {}; // activation fraction for aerosol mass [fraction]
      Real fluxm[ntot_amode] =
          {}; // flux of activated aerosol mass fraction into cloud [m/s]
      Real fluxn[ntot_amode] =
          {}; // flux of activated aerosol number fraction into cloud [m/s]
      Real flux_fullact = zero;
      ; // flux of activated aerosol fraction assuming 100% activation [m/s]
      get_activate_frac(
          state_q_col_in_kp1, air_density_kp1, air_density, wtke_col_in,
          temp_col_in, // in
          lspectype_amode, specdens_amode, spechygro, lmassptr_amode,
          voltonumbhi_amode, voltonumblo_amode, numptr_amode, nspec_amode,
          exp45logsig, alogsig, aten, factnum_col, fm, fluxn, fluxm, // out
          flux_fullact);

      //
      //  store for output activation fraction of aerosol
      // factnum_col(kk,:) = fn
      // vertical change in cloud raction [fraction]
      const Real delz_cld = cldn_col_in - cldn_col_in_kp1;

      // flux of activated aerosol number into cloud [#/m^2/s]
      Real fluxntot = zero;

      // ! rce-comment 1
      // !    flux of activated mass into layer k (in kg/m2/s)
      // !       = "actmassflux" = dumc*fluxm*raercol(kp1,lmass)*csbot(k)
      // !    source of activated mass (in kg/kg/s) = flux divergence
      // !       = actmassflux/(cs(i,k)*dz(i,k))
      // !    so need factor of csbot_cscen = csbot(k)/cs(i,k)
      // !                   dum=1./(dz(i,k))
      // conversion factor from flux to rate [m^{-1}]
      const Real crdz = csbot_cscen / dz;

      // ! rce-comment 2
      // !    code for k=pver was changed to use the following conceptual model
      // !    in k=pver, there can be no cloud-base activation unless one
      // considers !       a scenario such as the layer being partially cloudy,
      // !       with clear air at bottom and cloudy air at top
      // !    assume this scenario, and that the clear/cloudy portions mix with
      // !       a timescale taumix_internal = dz(i,pver)/wtke_cen(i,pver)
      // !    in the absence of other sources/sinks, qact (the activated
      // particle !       mixratio) attains a steady state value given by !
      // qact_ss = fcloud*fact*qtot !       where fcloud is cloud fraction, fact
      // is activation fraction, !       qtot=qact+qint, qint is interstitial
      // particle mixratio !    the activation rate (from mixing within the
      // layer) can now be !       written as !          d(qact)/dt = (qact_ss -
      // qact)/taumix_internal !                     =
      // qtot*(fcloud*fact*wtke/dz) - qact*(wtke/dz) !    note that
      // (fcloud*fact*wtke/dz) is equal to the nact/mact !    also, d(qact)/dt
      // can be negative.  in the code below !       it is forced to be >= 0
      // !
      // ! steve --
      // !    you will likely want to change this.  i did not really understand
      // !       what was previously being done in k=pver
      // !    in the cam3_5_3 code, wtke(i,pver) appears to be equal to the
      // !       droplet deposition velocity which is quite small
      // !    in the cam3_5_37 version, wtke is done differently and is much
      // !       larger in k=pver, so the activation is stronger there
      // !
      for (int imode = 0; imode < ntot_amode; ++imode) {
        // local array index for MAM number, species
        // Fortran indexing to C++ indexing
        const int mm = mam_idx[imode][0] - 1;
        nact[imode] += fluxn[imode] * crdz * delz_cld;
        mact[imode] += fluxm[imode] * crdz * delz_cld;
        // ! note that kp1 is used here
        fluxntot +=
            fluxn[imode] * delz_cld * raercol_nsav_kp1[mm] * air_density;
      } // end imode

      nsource_col += fluxntot / (air_density * dz);
    } // end cldn_col_in(kk) - cldn_col_in(kp1) > cld_thresh
  } else {

    //         !  if cldn < 0.01_r8 at any level except kk=pver, deplete qcld,
    //         turn all raercol_cw to raercol, put appropriate tendency
    // !  in nsource
    // !  Note that if cldn(kk) >= 0.01_r8 but cldn(kk) - cldn(kp1)  <= 0.01,
    // nothing is done.

    // ! no cloud

    nsource_col -= qcld * dtinv;
    qcld = zero;

    // convert activated aerosol to interstitial in decaying cloud

    for (int imode = 0; imode < ntot_amode; ++imode) {
      // local array index for MAM number, species
      // Fortran indexing to C++ indexing
      int mm = mam_idx[imode][0] - 1;
      raercol_nsav[mm] += raercol_cw_nsav[mm]; // ! cloud-borne aerosol
      raercol_cw_nsav[mm] = zero;

      for (int lspec = 1; lspec < nspec_amode[imode] + 1; ++lspec) {
        mm = mam_idx[imode][lspec] - 1;
        raercol_nsav[mm] += raercol_cw_nsav[mm]; // cloud-borne aerosol
        raercol_cw_nsav[mm] = zero;
      }
    }

  } // end cldn_col_in(kk) > cld_thresh

} // end update_from_cldn_profile

KOKKOS_INLINE_FUNCTION
void update_from_newcld(
    const Real cldn_col_in, const Real cldo_col_in,
    const Real dtinv, //& ! in
    const Real wtke_col_in, const Real temp_col_in, const Real air_density,
    const Real state_q_col_in[nvars], //& ! in
    const int lspectype_amode[maxd_aspectype][AeroConfig::num_modes()],
    const Real specdens_amode[maxd_aspectype],
    const Real spechygro[maxd_aspectype],
    const int lmassptr_amode[maxd_aspectype][AeroConfig::num_modes()],
    const Real voltonumbhi_amode[AeroConfig::num_modes()],
    const Real voltonumblo_amode[AeroConfig::num_modes()],
    const int numptr_amode[AeroConfig::num_modes()],
    const int nspec_amode[maxd_aspectype],
    const Real exp45logsig[AeroConfig::num_modes()],
    const Real alogsig[AeroConfig::num_modes()], const Real aten,
    const int mam_idx[AeroConfig::num_modes()][nspec_max], Real &qcld,
    Real raercol_nsav[ncnst_tot],
    Real raercol_cw_nsav[ncnst_tot], //&      ! inout
    Real &nsource_col_out, Real factnum_col_out[AeroConfig::num_modes()]) {

  // // input arguments
  // real(r8), intent(in) :: cldn_col_in(:)   ! cloud fraction [fraction]
  // real(r8), intent(in) :: cldo_col_in(:)   ! cloud fraction on previous time
  // step [fraction] real(r8), intent(in) :: dtinv     ! inverse time step for
  // microphysics [s^{-1}] real(r8), intent(in) :: wtke_col_in(:)   ! subgrid
  // vertical velocity [m/s] real(r8), intent(in) :: temp_col_in(:)   !
  // temperature [K] real(r8), intent(in) :: cs_col_in(:)     ! air density at
  // actual level kk [kg/m^3] real(r8), intent(in) :: state_q_col_in(:,:) !
  // aerosol mmrs [kg/kg]

  // real(r8), intent(inout) :: qcld(:)  ! cloud droplet number mixing ratio
  // [#/kg] real(r8), intent(inout) :: nsource_col_out(:)   ! droplet number
  // mixing ratio source tendency [#/kg/s] real(r8), intent(inout) ::
  // raercol_nsav(:,:)   ! single column of saved aerosol mass, number mixing
  // ratios [#/kg or kg/kg] real(r8), intent(inout) :: raercol_cw_nsav(:,:)  !
  // same as raercol_nsav but for cloud-borne phase [#/kg or kg/kg] real(r8),
  // intent(inout) :: factnum_col_out(:,:)  ! activation fraction for aerosol
  // number [fraction] ! k-loop for growing/shrinking cloud calcs
  // .............................
  const Real zero = 0;
  const Real one = 1;
  const int ntot_amode = AeroConfig::num_modes();
  // BAD CONSTANT
  const Real grow_cld_thresh =
      0.01; //   !  threshold cloud fraction growth [fraction]

  // new - old cloud fraction [fraction]
  const Real delt_cld = cldn_col_in - cldo_col_in;

  // ! shrinking cloud ......................................................
  // !    treat the reduction of cloud fraction from when cldn(i,k) < cldo(i,k)
  // !    and also dissipate the portion of the cloud that will be regenerated

  if (cldn_col_in < cldo_col_in) {
    // !  droplet loss in decaying cloud
    // !++ sungsup
    nsource_col_out += qcld * (cldn_col_in - cldo_col_in) / cldo_col_in * dtinv;

    qcld *= (one + (cldn_col_in - cldo_col_in) / cldo_col_in);
    // !-- sungsup
    // ! convert activated aerosol to interstitial in decaying cloud
    // fractional change in cloud fraction [fraction]
    const Real frac_delt_cld = (cldn_col_in - cldo_col_in) / cldo_col_in;

    for (int imode = 0; imode < ntot_amode; ++imode) {
      // Fortran indexing to C++ indexing
      const int mm = mam_idx[imode][0] - 1;
      // cloud-borne aerosol tendency due to cloud frac tendency [#/kg or
      // kg/kg]
      const Real dact = raercol_cw_nsav[mm] * frac_delt_cld;
      raercol_cw_nsav[mm] += dact; //   ! cloud-borne aerosol
      raercol_nsav[mm] -= dact;
      for (int lspec = 1; lspec < nspec_amode[imode] + 1; ++lspec) {
        // Fortran indexing to C++ indexing
        const int mm = mam_idx[imode][lspec] - 1;
        const Real dact = raercol_cw_nsav[mm] * frac_delt_cld;
        raercol_cw_nsav[mm] += dact; //  ! cloud-borne aerosol
        raercol_nsav[mm] -= dact;
      } // lspec
    }   // imode

  } // cldn_col_in < cldo_col_in

  // ! growing cloud ......................................................
  //        !    treat the increase of cloud fraction from when cldn(i,k) >
  //        cldo(i,k) !    and also regenerate part of the cloud

  if (cldn_col_in - cldo_col_in > grow_cld_thresh) {
    Real fm[ntot_amode] = {}; // activation fraction for aerosol mass [fraction]
    Real fluxm[ntot_amode] =
        {}; // flux of activated aerosol mass fraction into cloud [m/s]
    Real fluxn[ntot_amode] =
        {}; // flux of activated aerosol number fraction into cloud [m/s]
    Real flux_fullact = zero;
    ; // flux of activated aerosol fraction assuming 100% activation [m/s]

    get_activate_frac(state_q_col_in, air_density, air_density, wtke_col_in,
                      temp_col_in, // in
                      lspectype_amode, specdens_amode, spechygro,
                      lmassptr_amode, voltonumbhi_amode, voltonumblo_amode,
                      numptr_amode, nspec_amode, exp45logsig, alogsig, aten,
                      factnum_col_out, fm, fluxn, fluxm, // out
                      flux_fullact);
    //
    // !  store for output activation fraction of aerosol
    // factnum_col_out(kk,:) = fn

    for (int imode = 0; imode < ntot_amode; ++imode) {
      // Fortran indexing to C++ indexing
      const int mm = mam_idx[imode][0] - 1;
      // Fortran indexing to C++ indexing
      const int num_idx = numptr_amode[imode] - 1;
      const Real dact = delt_cld * factnum_col_out[imode] *
                        state_q_col_in[num_idx]; // ! interstitial only
      qcld += dact;
      nsource_col_out += dact * dtinv;
      raercol_cw_nsav[mm] += dact; //  ! cloud-borne aerosol
      raercol_nsav[mm] -= dact;
      // fm change from fractional change in cloud fraction [fraction]
      const Real fm_delt_cld = delt_cld * fm[imode];

      for (int lspec = 1; lspec < nspec_amode[imode] + 1; ++lspec) {
        // Fortran indexing to C++ indexing
        const int mm = mam_idx[imode][lspec] - 1;
        // Fortran indexing to C++ indexing
        const int spc_idx = lmassptr_amode[lspec - 1][imode] - 1;
        const Real dact =
            fm_delt_cld * state_q_col_in[spc_idx]; // ! interstitial only
        raercol_cw_nsav[mm] += dact;               //  ! cloud-borne aerosol
        raercol_nsav[mm] -= dact;

      } // lspec
    }   // imode
  }     // cldn_col_in - cldo_col_in  > grow_cld_thresh

} // update_from_newcld

KOKKOS_INLINE_FUNCTION
void explmix(
    const Real qold_km1, // number / mass mixing ratio from previous time step
                         // at level k-1 [# or kg / kg]
    const Real qold_k, // number / mass mixing ratio from previous time step at
                       // level k [# or kg / kg]
    const Real qold_kp1, // number / mass mixing ratio from previous time step
                         // at level k+1 [# or kg / kg]
    Real &
        qnew, // OUTPUT, number / mass mixing ratio to be updated [# or kg / kg]
    const Real src, // source due to activation/nucleation at level k [# or kg /
                    // (kg-s)]
    const Real ekkp,     // zn*zs*density*diffusivity (kg/m3 m2/s) at interface
                         // [/s]; below layer k  (k,k+1 interface)
    const Real ekkm,     // zn*zs*density*diffusivity (kg/m3 m2/s) at interface
                         // [/s]; above layer k  (k,k+1 interface)
    const Real overlapp, // cloud overlap below [fraction]
    const Real overlapm, // cloud overlap above [fraction]
    const Real dtmix     // time step [s]
) {

  qnew = qold_k + dtmix * (src + ekkp * (overlapp * qold_kp1 - qold_k) +
                           ekkm * (overlapm * qold_km1 - qold_k));

  // force to non-negative
  qnew = haero::max(qnew, 0);

} // end explmix

KOKKOS_INLINE_FUNCTION
void explmix(
    const Real qold_km1, // number / mass mixing ratio from previous time step
                         // at level k-1 [# or kg / kg]
    const Real qold_k, // number / mass mixing ratio from previous time step at
                       // level k [# or kg / kg]
    const Real qold_kp1, // number / mass mixing ratio from previous time step
                         // at level k+1 [# or kg / kg]
    Real &
        qnew, // OUTPUT, number / mass mixing ratio to be updated [# or kg / kg]
    const Real src, // source due to activation/nucleation at level k [# or kg /
                    // (kg-s)]
    const Real ekkp,     // zn*zs*density*diffusivity (kg/m3 m2/s) at interface
                         // [/s]; below layer k  (k,k+1 interface)
    const Real ekkm,     // zn*zs*density*diffusivity (kg/m3 m2/s) at interface
                         // [/s]; above layer k  (k,k+1 interface)
    const Real overlapp, // cloud overlap below [fraction]
    const Real overlapm, // cloud overlap above [fraction]
    const Real dtmix,    // time step [s]
    const Real qactold_km1,
    // optional: number / mass mixing ratio of ACTIVATED species
    // from previous step at level k-1 *** this should only be present if
    // the current species is unactivated number/sfc/mass
    const Real qactold_kp1
    // optional: number / mass mixing ratio of ACTIVATED species
    // from previous step at level k+1 *** this should only be present if
    // the current species is unactivated number/sfc/mass
) {

  // the qactold*(1-overlap) terms are resuspension of activated material
  const Real one = 1.0;
  qnew = qold_k +
         dtmix * (-src +
                  ekkp * (qold_kp1 - qold_k + qactold_kp1 * (one - overlapp)) +
                  ekkm * (qold_km1 - qold_k + qactold_km1 * (one - overlapm)));

  // force to non-negative
  qnew = haero::max(qnew, 0);
} // end explmix

KOKKOS_INLINE_FUNCTION
void update_from_explmix(
    const Real dtmicro, // time step for microphysics [s]
    int top_lev,        // top level
    int pver,           // number of levels
    ColumnView csbot,   // air density at bottom (interface) of layer [kg/m^3]
    ColumnView cldn,    // cloud fraction [fraction]
    ColumnView zn,      // g/pdel for layer [m^2/kg]
    ColumnView zs,      // inverse of distance between levels [m^-1]
    ColumnView ekd,     // diffusivity for droplets [m^2/s]
    ColumnView nact[pver], // fractional aero. number
                                              // activation rate [/s]
    ColumnView mact[pver], // fractional aero. mass
                                              // activation rate [/s]
    ColumnView qcld, // cloud droplet number mixing ratio [#/kg]
    ColumnView raercol[pver][2],    // single column of saved aerosol mass,
                                         // number mixing ratios [#/kg or kg/kg]
    ColumnView raercol_cw[pver][2], // same as raercol but for cloud-borne
                                         // phase [#/kg or kg/kg]
    int &nsav, // indices for old, new time levels in substepping
    int &nnew, // indices for old, new time levels in substepping
    const int nspec_amode[AeroConfig::num_modes()],
    const int mam_idx[AeroConfig::num_modes()][nspec_max],
    // work vars
    ColumnView overlapp, // cloud overlap involving level kk+1 [fraction]
    ColumnView overlapm, // cloud overlap involving level kk-1 [fraction]
    ColumnView ekkp,     // zn*zs*density*diffusivity [/s]
    ColumnView ekkm,     // zn*zs*density*diffusivity   [/s]
    ColumnView qncld,    // updated cloud droplet number mixing ratio [#/kg]
    ColumnView srcn,     // droplet source rate [/s]
    ColumnView source //  source rate for activated number or species mass [/s]
) {


  // BAD CONSTANT
  const Real overlap_cld_thresh =
      1e-10; //  threshold cloud fraction to compute overlap [fraction]
  const Real zero = 0.0;
  const Real one = 1.0;

  Real tmpa = zero; //  temporary aerosol tendency variable [/s]

  const int ntot_amode = AeroConfig::num_modes();
  // load new droplets in layers above, below clouds
  Real dtmin = dtmicro;
  // rce-comment -- ekd(k) is eddy-diffusivity at k/k+1 interface
  //   want ekk(k) = ekd(k) * (density at k/k+1 interface)
  //   so use pint(i,k+1) as pint is 1:pverp
  //           ekk(k)=ekd(k)*2.*pint(i,k)/(rair*(temp(i,k)+temp(i,k+1)))
  //           ekk(k)=ekd(k)*2.*pint(i,k+1)/(rair*(temp(i,k)+temp(i,k+1)))

  // start k for loop here. for k = top_lev to pver
  // cldn will be columnviews of length pver,
  // overlaps also to columnview pass as parameter so it is allocated elsewhwere
  for (int k = top_lev - 1; k < pver; k++) {
    const int kp1 = haero::min(k + 1, pver - 1);
    const int km1 = haero::max(k - 1, top_lev - 1);
    // maximum overlap assumption
    if (cldn(kp1) > overlap_cld_thresh) {
      overlapp(k) = haero::min(cldn(k) / cldn(kp1), one);
    } else {
      overlapp(k) = one;
    }

    if (cldn(km1) > overlap_cld_thresh) {
      overlapm(k) = haero::min(cldn(k) / cldn(km1), one);
    } else {
      overlapm(k) = one;
    }

    ekkp(k) = zn(k) * ekd(k) * csbot(k) * zs(k);
    // NOTE: ekk uses k-1 while sz uses km1.
    ekkm(k) = zn(k) * ekd(k-1) * csbot(k-1) * zs(km1);
    const Real tinv = ekkp(k) + ekkm(k);

      // rce-comment
  //    the activation source(k) = mact(k,m)*raercol(kp1,lmass)
  //       should not exceed the rate of transfer of unactivated particles
  //       from kp1 to k which = ekkp(k)*raercol(kp1,lmass)
  //    however it might if things are not "just right" in subr activate
  //    the following is a safety measure to avoid negatives in explmix

  for (int imode = 0; imode < ntot_amode; imode++) {
      nact[k](imode) = haero::min(nact[k](imode), ekkp(k));
      mact[k](imode) = haero::min(mact[k](imode), ekkp(k));
  }
  
    // rce-comment -- tinv is the sum of all first-order-loss-rates
    //    for the layer.  for most layers, the activation loss rate
    //    (for interstitial particles) is accounted for by the loss by
    //    turb-transfer to the layer above.
    //    k=pver is special, and the loss rate for activation within
    //    the layer must be added to tinv.  if not, the time step
    //    can be too big, and explmix can produce negative values.
    //    the negative values are reset to zero, resulting in an
    //    artificial source.

    // FIXME: BAD CONSTANT
    if (tinv > 1e-6) {
      dtmin = haero::min(dtmin, one / tinv);
    }
  }
  // TODO
  //    fix dtmin section
  //    pass arrs as columnviews and k len instead of single values
  //    loop over k for calls to explmis
  //    for things like src, qcld, loop over to assign the values
  //    pass overlapp etc as params as work vars so they are allocated elsewhere

  // timescale for subloop [s]
  //  BAD CONSTANT
  Real dtmix = 0.9 * dtmin;
  // number of substeps and bound
  const int nsubmix = dtmicro / dtmix + 1;
  // FIXME: nsubmix_bnd is used in the code. Ask Fortran team.
  //  re: nsubmix_bnd isn't used in the code?
  //  if (nsubmix > 100) {
  //     nsubmix_bnd = 100;
  //  } else {
  //     nsubmix_bnd = nsubmix;
  //  }

  dtmix = dtmicro / nsubmix;

  // old_cloud_nsubmix_loop
  //  Note:  each pass in submix loop stores updated aerosol values at index
  //  nnew, current values at index nsav.  At the start of each pass, nnew
  //  values are copied to nsav.  However, this is accomplished by switching the
  //  values of nsav and nnew rather than a physical copying.  At end of loop
  //  nnew stores index of most recent updated values (either 1 or 2).

  for (int isub = 0; isub < nsubmix; isub++) {
    for (int k = top_lev - 1; k < pver; k++) {
      qncld(k) = qcld(k);
      srcn(k) = zero;
    } // end kk
    // after first pass, switch nsav, nnew so that nsav is the recently updated
    // aerosol
    if (isub > 0) {
      const int ntemp = nsav;
      nsav = nnew;
      nnew = ntemp;
    } // end if

    for (int imode = 0; imode < ntot_amode; imode++) {
      const int mm = mam_idx[imode][0] - 1;

      // update droplet source

      // rce-comment- activation source in layer k involves particles from k+1
      //         srcn(:)=srcn(:)+nact(:,m)*(raercol(:,mm,nsav))
      for (int k = top_lev - 1; k < pver - 1; k++) {
        const int kp1 = haero::min(k + 1, pver - 1);
        srcn(k) += nact[k](imode) * raercol[kp1][nsav](mm);
      } // kk

      // rce-comment- new formulation for k=pver
      //             srcn(  pver  )=srcn(  pver  )+nact(  pver  ,m)*(raercol(
      //             pver,mm,nsav))
      tmpa = raercol[pver - 1][nsav](mm) * nact[pver - 1](imode) +
             raercol_cw[pver - 1][nsav](mm) * nact[pver - 1](imode);
      srcn(pver - 1) += haero::max(zero, tmpa);

    } // end imode

    // qcld == qold
    // qncld == qnew
    for (int k = top_lev - 1; k < pver; k++) {
      const int kp1 = haero::min(k + 1, pver - 1);
      const int km1 = haero::max(k - 1, top_lev - 1);
      explmix(qncld(km1), qncld(k), qncld(kp1),
              qcld(k), // output FIXME: move to last position
              srcn(k), ekkp(k), ekkm(k), overlapp(k), overlapm(k), dtmix);
    } // end kk

    // update aerosol number
    // rce-comment
    //    the interstitial particle mixratio is different in clear/cloudy
    //    portions of a layer, and generally higher in the clear portion.  (we
    //    have/had a method for diagnosing the the clear/cloudy mixratios.)  the
    //    activation source terms involve clear air (from below) moving into
    //    cloudy air (above). in theory, the clear-portion mixratio should be
    //    used when calculating source terms
    for (int imode = 0; imode < ntot_amode; imode++) {
      const int mm = mam_idx[imode][0] - 1;
      // rce-comment -   activation source in layer k involves particles from
      // k+1
      //                source(:)= nact(:,m)*(raercol(:,mm,nsav))
      for (int k = top_lev - 1; k < pver - 1; k++) {
        const int kp1 = haero::min(k + 1, pver - 1);
        // const int km1 = haero::max(k-1, top_lev);
        source(k) = nact[k](imode) * raercol[kp1][nsav](mm);
      } // end k

      tmpa = raercol[pver - 1][nsav](mm) * nact[pver - 1](imode) +
             raercol_cw[pver - 1][nsav](mm) * nact[pver - 1](imode);
      source(pver - 1) = haero::max(zero, tmpa);

      // raercol_cw[mm][nnew] == qold
      // raercol_cw[mm][nsav] == qnew

      for (int k = top_lev - 1; k < pver; k++) {
        const int kp1 = haero::min(k + 1, pver - 1);
        const int km1 = haero::max(k - 1, top_lev - 1);

        explmix(raercol_cw[km1][nsav](mm), raercol_cw[k][nsav](mm),
                raercol_cw[kp1][nsav](mm),
                raercol_cw[k][nnew](mm), // output FIXME: move to last position
                source(k), ekkp(k), ekkm(k), overlapp(k), overlapm(k), dtmix);
        // raercol[mm][nnew] == qold
        // raercol[mm][nsav] == qnew
        // raercol_cw[mm][nsav] == qactold

        explmix(raercol[km1][nsav](mm), raercol[k][nsav](mm),
                raercol[kp1][nsav](mm),
                raercol[k][nnew](mm), //// output FIXME: move to last position
                source(k), ekkp(k), ekkm(k), overlapp(k), overlapm(k), dtmix,
                raercol_cw[km1][nsav](mm),
                raercol_cw[kp1][nsav](mm)); // optional in

      } // end kk

      // update aerosol species mass
      for (int lspec = 1; lspec < nspec_amode[imode] + 1; lspec++) {
        const int mm = mam_idx[imode][lspec] - 1;
        // rce-comment -   activation source in layer k involves particles from
        // k+1
        //            source(:)= mact(:,m)*(raercol(:,mm,nsav))
        for (int k = top_lev - 1; k < pver - 1; k++) {
          const int kp1 = haero::min(k + 1, pver - 1);
          source(k) = mact[k](imode) * raercol[kp1][nsav](mm);

        } // end k
        tmpa = raercol[pver - 1][nsav](mm)* nact[pver - 1](imode) +
               raercol_cw[pver - 1][nsav](mm)* nact[pver - 1](imode);
        source(pver - 1) = haero::max(zero, tmpa);

        // raercol_cw[mm][nnew] == qold
        // raercol_cw[mm][nsav] == qnew
        for (int k = top_lev - 1; k < pver; k++) {
          const int kp1 = haero::min(k + 1, pver - 1);
          const int km1 = haero::max(k - 1, top_lev - 1);
          explmix(
              raercol_cw[km1][nsav](mm), raercol_cw[k][nsav](mm),
              raercol_cw[kp1][nsav](mm),
              raercol_cw[k][nnew](mm), //// output FIXME: move to last position
              source(k), ekkp(k), ekkm(k), overlapp(k), overlapm(k), dtmix);

          // raercol[mm][nnew] == qold
          // raercol[mm][nsav] == qnew
          // raercol_cw[mm][nsav] == qactold
          explmix(raercol[km1][nsav](mm), raercol[k][nsav](mm),
                  raercol[kp1][nsav](mm),
                  raercol[k][nnew](mm), //// output FIXME: move to last position
                  source(k), ekkp(k), ekkm(k), overlapp(k), overlapm(k), dtmix,
                  raercol_cw[km1][nsav](mm),
                  raercol_cw[kp1][nsav](mm)); // optional in
        }                                     // end kk

      } // lspec loop
    }   //  imode loop

  } // old_cloud_nsubmix_loop

  // evaporate particles again if no cloud
  for (int k = top_lev - 1; k < pver; k++) {
    if (cldn(k) == zero) {
      // no cloud
      qcld(k) = zero;

      // convert activated aerosol to interstitial in decaying cloud
      for (int imode = 0; imode < ntot_amode; imode++) {
        const int mm = mam_idx[imode][0] - 1;
        raercol[k][nnew](mm) += raercol_cw[k][nnew](mm);
        raercol_cw[k][nnew](mm) = zero;

        for (int lspec = 1; lspec < nspec_amode[imode] + 1; lspec++) {
          const int mm = mam_idx[imode][lspec] - 1;
          raercol[k][nnew](mm) += raercol_cw[k][nnew](mm);
          raercol_cw[k][nnew](mm) = zero;
        } // lspec
      }   // imode
    }     // if cldn(k) == 0
  }       // kk

} // end update_from_explmix

KOKKOS_INLINE_FUNCTION
void dropmixnuc(
    const Real dtmicro, 
    ColumnView temp,
    ColumnView pmid,
    ColumnView pint,
    ColumnView pdel,
    ColumnView rpdel,
    ColumnView zm,
    ColumnView state_q[pver],
    ColumnView ncldwtr,
    ColumnView kvh, // kvh[kk+1]
    ColumnView cldn,
    const int lspectype_amode[maxd_aspectype][AeroConfig::num_modes()],
    const Real specdens_amode[maxd_aspectype],
    const Real spechygro[maxd_aspectype],
    const int lmassptr_amode[maxd_aspectype][AeroConfig::num_modes()],
    const Real voltonumbhi_amode[AeroConfig::num_modes()],
    const Real voltonumblo_amode[AeroConfig::num_modes()],
    const int numptr_amode[AeroConfig::num_modes()],
    const int nspec_amode[maxd_aspectype],
    const Real exp45logsig[AeroConfig::num_modes()],
    const Real alogsig[AeroConfig::num_modes()], const Real aten,
    const int mam_idx[AeroConfig::num_modes()][nspec_max],
    const int mam_cnst_idx[AeroConfig::num_modes()][nspec_max], ColumnView qcld,
    ColumnView wsub,
    ColumnView cldo,                // in
    ColumnView qqcw_fld[ncnst_tot], // inout
    ColumnView ptend_q[nvar_ptend_q], ColumnView tendnd,
    ColumnView factnum[pver], ColumnView ndropcol,
    ColumnView ndropmix, ColumnView nsource, ColumnView wtke,
    ColumnView ccn[pver], ColumnView coltend[ncnst_tot],
    ColumnView coltend_cw[ncnst_tot],
    // work arrays
    ColumnView raercol_cw[pver][2], ColumnView raercol[pver][2],
    ColumnView nact[pver],
    ColumnView mact[pver],
    ColumnView ekd, ColumnView zn,
    ColumnView csbot, ColumnView zs, ColumnView overlapp, ColumnView overlapm,
    ColumnView ekkp, ColumnView ekkm, ColumnView qncld,
    ColumnView srcn, ColumnView source) {
  // vertical diffusion and nucleation of cloud droplets
  // assume cloud presence controlled by cloud fraction
  // doesn't distinguish between warm, cold clouds

  // input arguments
  // lchnk               ! chunk identifier
  // ncol                ! number of columns
  // psetcols            ! maximum number of columns
  // dtmicro     ! time step for microphysics [s]
  // temp(:,:)    ! temperature [K]
  // pmid(:,:)    ! mid-level pressure [Pa]
  // pint(:,:)    ! pressure at layer interfaces [Pa]
  // pdel(:,:)    ! pressure thickess of layer [Pa]
  // rpdel(:,:)   ! inverse of pressure thickess of layer [/Pa]
  // zm(:,:)      ! geopotential height of level [m]
  // state_q(:,:,:) ! aerosol mmrs [kg/kg]
  // ncldwtr(:,:) ! initial droplet number mixing ratio [#/kg]
  // kvh(:,:)     ! vertical diffusivity [m^2/s]
  // wsub(pcols,pver)    ! subgrid vertical velocity [m/s]
  // cldn(pcols,pver)    ! cloud fraction [fraction]
  // cldo(pcols,pver)    ! cloud fraction on previous time step [fraction]

  // inout arguments
  //  qqcw(:)     ! cloud-borne aerosol mass, number mixing ratios [#/kg or
  //  kg/kg]

  // output arguments
  //   ptend
  // tendnd(pcols,pver) ! tendency in droplet number mixing ratio [#/kg/s]
  // factnum(:,:,:)     ! activation fraction for aerosol number [fraction]

  // nsource droplet number mixing ratio source tendency [#/kg/s]

  // ndropmix droplet number mixing ratio tendency due to mixing [#/kg/s]
  // ccn number conc of aerosols activated at supersat [#/m^3]

  //  !     note:  activation fraction fluxes are defined as
  // !     fluxn = [flux of activated aero. number into cloud [#/m^2/s]]
  // !           / [aero. number conc. in updraft, just below cloudbase [#/m^3]]

  // coltend(:,:)       ! column tendency for diagnostic output
  // coltend_cw(:,:)    ! column tendency

  // BAD CONSTANT
  const Real zkmin = 0.01;
  const Real zkmax = 100; //  ! min, max vertical diffusivity [m^2/s]
  const Real wmixmin =
      0.1; //        ! minimum turbulence vertical velocity [m/s]

  const Real zero = 0;
  const Real one = 1;
  const Real two = 2;

  /// inverse time step for microphysics [s^-1]
  const Real dtinv = one / dtmicro;
  const int ntot_amode = AeroConfig::num_modes();
  // THIS constant depends on the chem mechanism

  // initialize variables to zero
  //  ndropmix(:,:) = 0._r8
  //  nsource(:,:) = 0._r8
  //  wtke(:,:)    = 0._r8
  //  factnum(:,:,:) = 0._r8

  //! NOTE FOR C++ PORT: Get the cloud borne MMRs from AD in variable qcldbrn,
  //! do not port the code before END NOTE
  // qcldbrn(:,:,:,:) = huge(qcldbrn) !store invalid values
  //! END NOTE FOR C++ PORT
  const Real gravit = haero::Constants::gravity;
  const Real rair = haero::Constants::r_gas_dry_air;

  // const Real air_density = pmid/(rair*temp);//        ! air density (kg/m3)
  // geometric thickness of layers [m]
  // fractional aero. number  activation rate [/s]
  // Real raercol[nlevels][2][ncnst_tot] = {};
  // same as raercol but for cloud-borne phase [#/kg or kg/kg]
  // Real raercol_cw[nlevels][2][ncnst_tot] = {};

  // Initialize 1D (in space) versions of interstitial and cloud borne aerosol
  int nsav = 0;
  Kokkos::parallel_for(
      "update_from_newcld", pver - top_lev + 1, KOKKOS_LAMBDA(int kk) {
        // // ! g/pdel for layer [m^2/kg]
        // const Real zn = gravit * rpdel;
        // turbulent vertical velocity at base of layer k [m/s]
        const int k = kk + top_lev - 1;

        wtke(k) = haero::max(wsub(k), wmixmin);
        // qcld(:)  = ncldwtr(icol,:)
        // cloud droplet number mixing ratio [#/kg]
        qcld(k) = ncldwtr(k);

        for (int imode = 0; imode < ntot_amode; ++imode) {
          // Fortran indexing to C++ indexing

          const int mm = mam_idx[imode][0] - 1;
          raercol_cw[k][nsav](mm) = qqcw_fld[mm](k);
          // Fortran indexing to C++ indexing
          const int num_idx = numptr_amode[imode] - 1;
          raercol[k][nsav][mm] = state_q[k][num_idx];
          for (int lspec = 1; lspec < nspec_amode[imode] + 1; ++lspec) {
            // Fortran indexing to C++ indexing
            const int mm = mam_idx[imode][lspec] - 1;

            raercol_cw[k][nsav](mm) = qqcw_fld[mm](k);
            // Fortran indexing to C++ indexing
            const int spc_idx = lmassptr_amode[lspec - 1][imode] - 1;
            raercol[k][nsav][mm] = state_q[k][spc_idx];
          } // lspec
        }   // imode

        // PART I:  changes of aerosol and cloud water from temporal changes in
        // cloud
        // fraction droplet nucleation/aerosol activation
        nsource(k) = zero;

        const Real air_density =
            conversions::density_of_ideal_gas(temp(k), pmid(k));

        update_from_newcld(cldn(k), cldo(k), dtinv, //& ! in
                           wtke(k), temp(k), air_density,
                           state_q[k].data(), //& ! in
                           lspectype_amode, specdens_amode, spechygro,
                           lmassptr_amode, voltonumbhi_amode, voltonumblo_amode,
                           numptr_amode, nspec_amode, exp45logsig, alogsig,
                           aten, mam_idx, qcld(k),
                           raercol[k][nsav].data(),            // inout
                           raercol_cw[k][nsav].data(),         //&      ! inout
                           nsource(k), factnum[k].data()); // inout
      }); // end k

  // NOTE: update_from_cldn_profile loop from 7 to 71 in fortran code.
  Kokkos::parallel_for(
      "update_from_cldn_profile", pver - top_lev, KOKKOS_LAMBDA(int kk) {
        const int k = kk + top_lev - 1;
        const int kp1 = haero::min(k + 1, pver - 1);

        const Real air_density =
            conversions::density_of_ideal_gas(temp(k), pmid(k));

        const Real air_density_kp1 =
            conversions::density_of_ideal_gas(temp(kp1), pmid(kp1));

        // PART II: changes in aerosol and cloud water from vertical profile of
        // new cloud fraction

        Real delta_zm, csbot, csbot_cscen = zero;
        if (k >= top_lev - 1 && k < pver - 1) {
          delta_zm = zm(k) - zm(k + 1);
          csbot = two * pint(k + 1) / (rair * (temp(k) + temp(k + 1)));
          csbot_cscen = csbot / air_density;

          ekd(k) = utils::min_max_bound(zkmin, zkmax, kvh(k + 1));
        } else {
          delta_zm = zm(k - 1) - zm(k);
          csbot_cscen = one;
          // FIXME: which density k or kp1?
          csbot = air_density;
        }
        const Real zs = one / delta_zm;

        const Real dz =
            one / (air_density * gravit * rpdel(k)); // ! layer thickness in m

        update_from_cldn_profile(
            cldn(k), cldn(kp1), dtinv, wtke(k), zs, dz, // ! in
            temp(k), air_density, air_density_kp1, csbot_cscen,
            state_q[kp1].data(), // ! in
            lspectype_amode, specdens_amode, spechygro, lmassptr_amode,
            voltonumbhi_amode, voltonumblo_amode, numptr_amode, nspec_amode,
            exp45logsig, alogsig, aten, mam_idx, raercol[k][nsav].data(), raercol[kp1][nsav].data(),
            raercol_cw[k][nsav].data(),
            nsource(k), // inout
            qcld(k), factnum[k].data(),
            ekd(k), // out
            nact[k].data(), mact[k].data());
      });

  // PART III:  perform explict integration of droplet/aerosol mixing using
  // substepping

  int nnew = 1;
  // // single column of aerosol mass, number mixing ratios [#/kg or kg/kg]
  // Real raercol_2[ncnst_tot] = {};
  // // same as raercol but for cloud-borne phase [#/kg or kg/kg]
  // Real raercol_cw_2[ncnst_tot] = {};

  Kokkos::parallel_for(
      "pre-update_from_explmix", pver, KOKKOS_LAMBDA(int k) {
        zn(k) = gravit * rpdel[k];
        Real delta_zm = zero;
        // Real csbot_km1 = zero;
        if (k >= top_lev - 1 && k < pver - 1) {
          csbot(k) = two * pint(k + 1) / (rair * (temp(k) + temp(k + 1)));
          delta_zm = zm(k) - zm(k + 1);

        } else {
          // FIXME: which density k or kp1?
          const Real air_density =
              conversions::density_of_ideal_gas(temp(k), pmid(k));
          csbot(k) = air_density;
          // FIXME ; check this
          // csbot_km1 = two * pint(k - 1) / (rair * (temp(k - 2) + temp(k -
          // 1)));
          delta_zm = zm(k - 1) - zm(k);
        }

        zs(k) = one / delta_zm;
      }); // end k

  update_from_explmix(dtmicro, top_lev, pver, csbot, cldn, zn, zs, ekd, nact,
                      mact, qcld, raercol, raercol_cw, nsav, nnew, nspec_amode,
                      mam_idx,
                      // work vars
                      overlapp, overlapm, ekkp, ekkm, qncld,
                      srcn, // droplet source rate [/s]
                      source);

  Kokkos::parallel_for(
      "qqcw_fld initialization", top_lev - 1, KOKKOS_LAMBDA(int kk) {
        for (int i = 0; i < ncnst_tot; ++i) {
          qqcw_fld[i](kk) = zero;
        }
      });

  Kokkos::parallel_for(
      "ccncalc", pver - top_lev + 1, KOKKOS_LAMBDA(int kk) {
        const int k = kk + top_lev - 1;
        // droplet number mixing ratio tendency due to mixing [#/kg/s]
        ndropmix(k) = (qcld(k) - ncldwtr(k)) * dtinv - nsource(k);
        // BAD CONSTANT
        tendnd(k) = (haero::max(qcld(k), 1.e-6) - ncldwtr(k)) * dtinv;
        // We need to port this outside of this function
        // this is a reduction
        // do kk = top_lev, pver
        //           ndropcol(icol)   = ndropcol(icol) +
        //           ncldwtr(icol,kk)*pdel(icol,kk)
        // enddo
        // ndropcol(icol) = ndropcol(icol)/gravit
        // sum up ndropcol_kk outside of kk loop
        // column-integrated droplet number [#/m2]
        ndropcol(k) = ncldwtr(k) * pdel(k) / gravit;
        // tendency of interstitial aerosol mass, number mixing ratios [#/kg/s
        // or kg/kg/s]
        Real raertend = zero;
        // tendency of cloudborne aerosol mass, number mixing ratios [#/kg/s or
        // kg/kg/s]
        Real qqcwtend = zero;
        // cloud-borne aerosol mass mixing ratios [kg/kg]
        Real qcldbrn[maxd_aspectype][ntot_amode] = {{zero}};
        // cloud-borne aerosol number mixing ratios [#/kg]
        Real qcldbrn_num[ntot_amode] = {zero};

        for (int imode = 0; imode < ntot_amode; ++imode) {
          // species index for given mode
          for (int lspec = 0; lspec < nspec_amode[imode] + 1; ++lspec) {
            // local array index for MAM number, species
            // Fortran indexing to C++ indexing
            const int mm = mam_idx[imode][lspec] - 1;
            //
            // Fortran indexing to C++ indexing
            const int lptr = mam_cnst_idx[imode][lspec] - 1;
            //
            qqcwtend = (raercol_cw[k][nnew](mm) - qqcw_fld[mm](k)) * dtinv;
            qqcw_fld[mm](k) = haero::max(
                raercol_cw[k][nnew](mm),
                zero); // ! update cloud-borne aerosol; HW: ensure non-negative

            if (lspec == 0) {
              // Fortran indexing to C++ indexing
              const int num_idx = numptr_amode[imode] - 1;
              raertend = (raercol[k][nnew](mm) - state_q[k](num_idx)) * dtinv;
              qcldbrn_num[imode] = qqcw_fld[mm](k);
            } else {
              // Fortran indexing to C++ indexing
              const int spc_idx = lmassptr_amode[lspec - 1][imode] - 1;
              raertend = (raercol[k][nnew](mm) - state_q[k](spc_idx)) * dtinv;
              //! Extract cloud borne MMRs from qqcw pointer
              qcldbrn[lspec][imode] = qqcw_fld[mm](k);
            } // end if
            // NOTE: perform sum after loop. Thus, we need to store coltend_kk
            // and coltend_cw_kk Port this code outside of this function
            // coltend(icol,mm)    = sum( pdel(icol,:)*raertend )/gravit
            // coltend_cw(icol,mm) = sum( pdel(icol,:)*qqcwtend )/gravit
            coltend[mm](k) = pdel(k) * raertend / gravit;
            coltend_cw[mm](k) = pdel(k) * qqcwtend / gravit;
            ptend_q[lptr](k) =
                raertend; //          ! set tendencies for interstitial aerosol

          } // lspec

        } // imode

        // !  Use interstitial and cloud-borne aerosol to compute output ccn
        // fields.

        const Real air_density =
            conversions::density_of_ideal_gas(temp(k), pmid(k));

        ccncalc(state_q[k].data(), temp(k), qcldbrn, qcldbrn_num, air_density,
                lspectype_amode, specdens_amode, spechygro, lmassptr_amode,
                voltonumbhi_amode, voltonumblo_amode, numptr_amode, nspec_amode,
                exp45logsig, alogsig, ccn[k].data());


      });

} // dropmixnuc

} // namespace ndrop

} // end namespace mam4

#endif