#ifndef MAM4XX_NDROP_OD_HPP
#define MAM4XX_NDROP_OD_HPP

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

namespace ndrop_od {

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

  ndrop::maxsat(zeta, eta, nmode, smc, smax);
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
    // printf("cldn_col_in < cldo_col_in \n");

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
    // printf("cldn_col_in - cldo_col_in > grow_cld_thresh \n");
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


// loops with 3 levels
const int pver = 72;
const int top_lev = 7;
KOKKOS_INLINE_FUNCTION
void dropmixnuc(
    const Real dtmicro, ColumnView temp, ColumnView pmid, ColumnView pint,
    ColumnView pdel, ColumnView rpdel, ColumnView zm, ColumnView state_q[nvars],
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
    ColumnView factnum[AeroConfig::num_modes()], ColumnView ndropcol,
    ColumnView ndropmix, ColumnView nsource, ColumnView wtke,
    ColumnView ccn[psat], ColumnView coltend[ncnst_tot],
    ColumnView coltend_cw[ncnst_tot],
    // work arrays
    ColumnView raercol_cw[2][ncnst_tot], ColumnView raercol[2][ncnst_tot],
    ColumnView nact[AeroConfig::num_modes()],
    ColumnView mact[AeroConfig::num_modes()], ColumnView ekd, ColumnView zn,
    ColumnView csbot, ColumnView zs, ColumnView overlapp, ColumnView overlapm,
    ColumnView ekk, ColumnView ekkp, ColumnView ekkm, ColumnView qncld,
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

  // Real nact[nlevels][ntot_amode] = {};
  // Real mact[nlevels][ntot_amode] = {};

  // const Real air_density = pmid/(rair*temp);//        ! air density (kg/m3)
  // geometric thickness of layers [m]
  // fractional aero. number  activation rate [/s]
  // Real raercol[nlevels][2][ncnst_tot] = {};
  // same as raercol but for cloud-borne phase [#/kg or kg/kg]
  // Real raercol_cw[nlevels][2][ncnst_tot] = {};

  // Initialize 1D (in space) versions of interstitial and cloud borne aerosol
  int nsav = 0;
  int print_this_k = 6;
  printf("pver %d \n", pver);
  printf("top_lev %d \n", top_lev);

  Kokkos::parallel_for(
      "update_from_newcld", pver - top_lev + 1, KOKKOS_LAMBDA(int kk) {
        // // ! g/pdel for layer [m^2/kg]
        // const Real zn = gravit * rpdel;
        // turbulent vertical velocity at base of layer k [m/s]
        const int k = kk + top_lev - 1;
        printf("update_from_newcld k %d \n", k);

        wtke(k) = haero::max(wsub(k), wmixmin);
        // qcld(:)  = ncldwtr(icol,:)
        // cloud droplet number mixing ratio [#/kg]
        qcld(k) = ncldwtr(k);
        // printf("k %d \n ", k);

        Real raercol_1_kk[ncnst_tot] = {};
        Real raercol_cw_1_kk[ncnst_tot] = {};

        for (int imode = 0; imode < ntot_amode; ++imode) {
          // Fortran indexing to C++ indexing

          const int mm = mam_idx[imode][0] - 1;
          raercol_cw_1_kk[mm] = qqcw_fld[mm](k);
          if (k == print_this_k) {
            printf("m %d \n", mm);
            printf("qqcw_fld[mm](k) %e \n", qqcw_fld[mm](k));
            printf("nspec_amode[imode] %d \n", nspec_amode[imode]);
          }
          // Fortran indexing to C++ indexing
          const int num_idx = numptr_amode[imode] - 1;
          raercol_1_kk[mm] = state_q[num_idx](k);
          for (int lspec = 1; lspec < nspec_amode[imode] + 1; ++lspec) {
            // Fortran indexing to C++ indexing
            const int mm = mam_idx[imode][lspec] - 1;

            raercol_cw_1_kk[mm] = qqcw_fld[mm](k);
            // Fortran indexing to C++ indexing
            const int spc_idx = lmassptr_amode[lspec - 1][imode] - 1;
            if (k == print_this_k) {
              printf("m %d lspec %d spc_idx %d \n", mm, lspec, spc_idx);
            }
            raercol_1_kk[mm] = state_q[spc_idx](k);
          } // lspec
        }   // imode

        // PART I:  changes of aerosol and cloud water from temporal changes in
        // cloud
        // fraction droplet nucleation/aerosol activation
        nsource(k) = zero;
        // for (int imode = 0; imode < ntot_amode; ++imode) {
        //   factnum[imode](k) = zero;
        // }

        const Real air_density =
            conversions::density_of_ideal_gas(temp(k), pmid(k));

        //
        Real state_q_kk[nvars];

        for (int i = 0; i < nvars; ++i) {
          // FIXME
          state_q_kk[i] = state_q[i](k);
        }
        Real factnum_kk[ntot_amode] = {};

        if (k == print_this_k) {
          printf("state_q_kk[i]");
          for (int i = 0; i < nvars; ++i) {
            printf(" %e", state_q_kk[i]);
          }
          printf("\n");

          printf(" cldn(k) %e \n", cldn(k));
          printf(" cldo(k) %e \n", cldo(k));
          printf(" dtinv %e \n", dtinv);
          printf(" wtke(k) %e \n", wtke(k));
          printf(" temp(k) %e \n", temp(k));
          printf(" air_density %e \n", air_density);
          printf(" spechygro %e \n", spechygro[0]);
          printf(" specdens_amode %e \n", specdens_amode[0]);
          // printf(" %e \n",);
          printf(" qcld %e \n", qcld(k));
          // printf(" raercol_cw_1_kk%e \n",raercol_cw_1_kk[0]);
          printf("raercol_cw_1_kk");
          for (int i = 0; i < ncnst_tot; ++i) {
            printf(" %e", raercol_cw_1_kk[i]);
          }
          printf("\n");

          printf("raercol_1_kk");
          for (int i = 0; i < ncnst_tot; ++i) {
            printf(" %e", raercol_1_kk[i]);
          }
          printf("\n");
        }

        update_from_newcld(cldn(k), cldo(k), dtinv, //& ! in
                           wtke(k), temp(k), air_density,
                           state_q_kk, //& ! in
                           lspectype_amode, specdens_amode, spechygro,
                           lmassptr_amode, voltonumbhi_amode, voltonumblo_amode,
                           numptr_amode, nspec_amode, exp45logsig, alogsig,
                           aten, mam_idx, qcld(k),
                           raercol_1_kk,            // inout
                           raercol_cw_1_kk,         //&      ! inout
                           nsource(k), factnum_kk); // inout
        // FIXME
        for (int i = 0; i < ntot_amode; ++i) {
          factnum[i](k) = factnum_kk[i];
          // printf("k:%d->factnum_kk[%d] %e \n",k,i, factnum_kk[i]);
        }

        for (int i = 0; i < ncnst_tot; ++i) {
          raercol[nsav][i](k) = raercol_1_kk[i];
          raercol_cw[nsav][i](k) = raercol_cw_1_kk[i];
        }

        if (k == print_this_k) {
          printf(" raercol_cw_1_kk \n");
          for (int i = 0; i < ncnst_tot; ++i) {
            printf(" %e", raercol_cw_1_kk[i]);
          }
          printf("\n");

          printf(" raercol_1_kk \n");
          for (int i = 0; i < ncnst_tot; ++i) {
            printf(" %e", raercol_1_kk[i]);
          }
          printf("\n");
        }
      }); // end k
  print_this_k = 6;
  // NOTE: update_from_cldn_profile loop from 7 to 71 in fortran code.
  Kokkos::parallel_for(
      "update_from_cldn_profile", pver - top_lev, KOKKOS_LAMBDA(int kk) {
        const int k = kk + top_lev - 1;
        const int kp1 = haero::min(k + 1, pver - 1);
        printf("update_from_cldn_profile k %d \n", k);

        const Real air_density =
            conversions::density_of_ideal_gas(temp(k), pmid(k));

        const Real air_density_kp1 =
            conversions::density_of_ideal_gas(temp(kp1), pmid(kp1));

        for (int imode = 0; imode < ntot_amode; ++imode) {
          // fractional aero. number  activation rate [/s]
          nact[k][imode] = zero;
          // fractional aero. mass    activation rate [/s]
          mact[k][imode] = zero;
        }

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

        Real state_q_kp1[nvars];

        for (int i = 0; i < nvars; ++i) {
          // FIXME
          state_q_kp1[i] = state_q[i](kp1);
        }

        Real raercol_1_kk[ncnst_tot] = {zero};
        Real raercol_1_kp1[ncnst_tot] = {zero};
        Real raercol_cw_1_kk[ncnst_tot] = {zero};

        for (int i = 0; i < ncnst_tot; ++i) {
          raercol_1_kk[i] = raercol[nsav][i](k);
          raercol_1_kp1[i] = raercol[nsav][i](kp1);
          raercol_cw_1_kk[i] = raercol_cw[nsav][i](k);
        }

        Real factnum_kk[ntot_amode] = {zero};
        for (int i = 0; i < ntot_amode; ++i) {
          factnum_kk[i] = factnum[i](k);
        }
        Real nact_kk[ntot_amode] = {zero};
        Real mact_kk[ntot_amode] = {zero};

        update_from_cldn_profile(
            cldn(k), cldn(kp1), dtinv, wtke(k), zs, dz, // ! in
            temp(k), air_density, air_density_kp1, csbot_cscen,
            state_q_kp1, // ! in
            lspectype_amode, specdens_amode, spechygro, lmassptr_amode,
            voltonumbhi_amode, voltonumblo_amode, numptr_amode, nspec_amode,
            exp45logsig, alogsig, aten, mam_idx, raercol_1_kk, raercol_1_kp1,
            raercol_cw_1_kk,
            nsource(k), // inout
            qcld(k), factnum_kk,
            ekd(k), // out
            nact_kk, mact_kk);

        //
        for (int i = 0; i < ncnst_tot; ++i) {
          raercol[nsav][i](k) = raercol_1_kk[i];
          raercol_cw[nsav][i](k) = raercol_cw_1_kk[i];
        }

        for (int i = 0; i < ntot_amode; ++i) {
          factnum[i](k) = factnum_kk[i];
          nact[i](k) = nact_kk[i];
          mact[i](k) = mact_kk[i];
        }

        if (k == print_this_k) {
          printf(" raercol_cw_1_kk \n");
          for (int i = 0; i < ncnst_tot; ++i) {
            printf(" %e", raercol_cw_1_kk[i]);
          }
          printf("\n");

          printf(" raercol_1_kk \n");
          for (int i = 0; i < ncnst_tot; ++i) {
            printf(" %e", raercol_1_kk[i]);
          }
          printf("\n");

          printf(" nsource(k) %e \n", nsource(k));
          printf(" qcld(k) %e \n", qcld(k));
          printf(" ekd %e \n", ekd(k));
          printf(" ekd(k-1) %e \n", ekd(k - 1));
          printf(" factnum_kk \n");
          for (int i = 0; i < ntot_amode; ++i) {
            printf(" %e", factnum_kk[i]);
          }
          printf("\n");
          printf("\n");
          printf(" nact_kk \n");
          for (int i = 0; i < ntot_amode; ++i) {
            printf(" %e", nact_kk[i]);
          }
          printf("\n");
          printf("\n");
          printf(" mact_kk \n");
          for (int i = 0; i < ntot_amode; ++i) {
            printf(" %e", mact_kk[i]);
          }
          printf("\n");
          printf("\n");
        }
      });

  // PART III:  perform explict integration of droplet/aerosol mixing using
  // substepping

  int nnew = 1;
  // NOTE: It will be ported by Jaelyn Litzinger
  // // single column of aerosol mass, number mixing ratios [#/kg or kg/kg]
  // Real raercol_2[ncnst_tot] = {};
  // // same as raercol but for cloud-borne phase [#/kg or kg/kg]
  // Real raercol_cw_2[ncnst_tot] = {};

  // printf("Before explmix raercol(top_lev:pver,mm,nnew) %e \n ",
  // raercol[6][nnew][8]);

  // printf( "dtmicro %e \n", dtmicro);
  // printf( "csbot %e \n", csbot[6]);
  // printf( "cldn(icol,:) %e \n", cldn[6]);
  // printf( "zn %e \n", gravit * rpdel[6]);
  // printf( "zs %e \n", zs[6]);
  // printf( "ekd %e \n", ekd[6]);

  // for (int i = 0; i < ntot_amode; ++i)
  // {
  //   printf( "nact %e ", nact[6][i]);
  // }
  // printf("\n");

  // for (int i = 0; i < ntot_amode; ++i)
  // {
  //   printf( "mact %e ", mact[6][i]);
  // }

  // printf("\n");
  // printf( "qcld %e \n",qcld[6]);

  // printf( "csbot %e \n", csbot[6]);

  print_this_k = 6;

  Kokkos::parallel_for(
      "pre_update_from_explmix", pver, KOKKOS_LAMBDA(int k) {
        printf("pre_update_from_explmix k %d \n", k);
        const int kp1 = haero::min(k + 1, pver - 1);
        const int km1 = haero::max(k - 1, top_lev - 1);

        zn(k) = gravit * rpdel[k];
        if (k == print_this_k) {
          printf("zn(%d) %e \n", k, zn(k));
        }

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
          // delta_zm_km1 = zm(k - 2) - zm(k - 1);
        }

        zs(k) = one / delta_zm;
        // const Real zs_km1 = one / delta_zm_km1;

        if (k == print_this_k) {

          printf("csbot %e\n", csbot(k));
          printf("csbot_km1 %e\n", csbot(km1));
          printf(" cldn(k) %e, cldn(km1) %e, cldn(kp1) %e\n", cldn(k),
                 cldn(km1), cldn(kp1));
          printf(" zn %e zs %e zs_km1 %e \n", zn(k), zs(k), zs(km1));
          printf(" ekd %e ekd_km1 %e \n", ekd(k), ekd(km1));
          printf(" qcld(k)%e  qcld(km1) %e qcld(kp1) %e \n", qcld(k), qcld(km1),
                 qcld(kp1));

          printf(" raercol_cw_kk 1 \n");
          for (int i = 0; i < ncnst_tot; ++i) {
            printf(" %e", raercol_cw[0][i](k));
          }
          printf("\n");

          printf(" raercol_cw_kk 2 \n");
          for (int i = 0; i < ncnst_tot; ++i) {
            printf(" %e", raercol_cw[1][i](k));
          }
          printf("\n");

          printf(" raercol_cw_km1 1 \n");
          for (int i = 0; i < ncnst_tot; ++i) {
            printf(" %e", raercol_cw[0][i](km1));
          }
          printf("\n");

          printf(" raercol_cw_km1 2 \n");
          for (int i = 0; i < ncnst_tot; ++i) {
            printf(" %e", raercol_cw[1][i](km1));
          }
          printf("\n");

          printf(" raercol_cw_kp1 1 \n");
          for (int i = 0; i < ncnst_tot; ++i) {
            printf(" %e", raercol_cw[0][i](kp1));
          }
          printf("\n");

          printf(" raercol_cw_kp1 2 \n");
          for (int i = 0; i < ncnst_tot; ++i) {
            printf(" %e", raercol_cw[1][i](kp1));
          }
          printf("\n");

          printf(" raercol_kk 1 \n");
          for (int i = 0; i < ncnst_tot; ++i) {
            printf(" %e", raercol[0][i](k));
          }
          printf("\n");

          printf(" raercol_kk 2 \n");
          for (int i = 0; i < ncnst_tot; ++i) {
            printf(" %e", raercol[1][i](k));
          }
          printf("\n");

          printf(" raercol_km1 1 \n");
          for (int i = 0; i < ncnst_tot; ++i) {
            printf(" %e", raercol[0][i](km1));
          }
          printf("\n");

          printf(" raercol_km1 2 \n");
          for (int i = 0; i < ncnst_tot; ++i) {
            printf(" %e", raercol[1][i](km1));
          }
          printf("\n");

          printf(" raercol_kp1 1 \n");
          for (int i = 0; i < ncnst_tot; ++i) {
            printf(" %e", raercol[0][i](kp1));
          }
          printf("\n");

          printf(" raercol_kp1 2 \n");
          for (int i = 0; i < ncnst_tot; ++i) {
            printf(" %e", raercol[1][i](kp1));
          }
          printf("\n");
        }
      }); // end k

  printf("B qcld %e \n", qcld(print_this_k));

  ndrop::update_from_explmix(dtmicro, top_lev, pver, csbot, cldn, zn, zs, ekd,
                             nact, mact, qcld, raercol, raercol_cw, nsav, nnew,
                             nspec_amode, mam_idx,
                             // work vars
                             overlapp, overlapm, ekk, ekkp, ekkm, qncld,
                             srcn, // droplet source rate [/s]
                             source);

  const int k = print_this_k;
  const int kp1 = haero::min(k + 1, pver - 1);
  const int km1 = haero::max(k - 1, top_lev - 1);

  printf("After ... \n");

  printf("nsav %d \n", nsav);
  printf("nnew %d \n", nnew);

  printf(" raercol_cw_kk 1 \n");
  for (int i = 0; i < ncnst_tot; ++i) {
    printf(" %e", raercol_cw[0][i](k));
  }
  printf("\n");

  printf(" raercol_cw_kk 2 \n");
  for (int i = 0; i < ncnst_tot; ++i) {
    printf(" %e", raercol_cw[1][i](k));
  }
  printf("\n");

  printf(" raercol_cw_km1 1 \n");
  for (int i = 0; i < ncnst_tot; ++i) {
    printf(" %e", raercol_cw[0][i](km1));
  }
  printf("\n");

  printf(" raercol_cw_km1 2 \n");
  for (int i = 0; i < ncnst_tot; ++i) {
    printf(" %e", raercol_cw[1][i](km1));
  }
  printf("\n");

  printf(" raercol_cw_kp1 1 \n");
  for (int i = 0; i < ncnst_tot; ++i) {
    printf(" %e", raercol_cw[0][i](kp1));
  }
  printf("\n");

  printf(" raercol_cw_kp1 2 \n");
  for (int i = 0; i < ncnst_tot; ++i) {
    printf(" %e", raercol_cw[1][i](kp1));
  }
  printf("\n");

  printf(" raercol_kk 1 \n");
  for (int i = 0; i < ncnst_tot; ++i) {
    printf(" %e", raercol[0][i](k));
  }
  printf("\n");

  printf(" raercol_kk 2 \n");
  for (int i = 0; i < ncnst_tot; ++i) {
    printf(" %e", raercol[1][i](k));
  }
  printf("\n");

  printf(" raercol_km1 1 \n");
  for (int i = 0; i < ncnst_tot; ++i) {
    printf(" %e", raercol[0][i](km1));
  }
  printf("\n");

  printf(" raercol_km1 2 \n");
  for (int i = 0; i < ncnst_tot; ++i) {
    printf(" %e", raercol[1][i](km1));
  }
  printf("\n");

  printf(" raercol_kp1 1 \n");
  for (int i = 0; i < ncnst_tot; ++i) {
    printf(" %e", raercol[0][i](kp1));
  }
  printf("\n");

  printf(" raercol_kp1 2 \n");
  for (int i = 0; i < ncnst_tot; ++i) {
    printf(" %e", raercol[1][i](kp1));
  }
  printf("\n");

  printf("\n");
  printf("\n");
  printf(" nact_kk \n");
  for (int i = 0; i < ntot_amode; ++i) {
    printf(" %e", nact[i](k));
  }
  printf("\n");
  printf("\n");
  printf(" mact_kk \n");
  for (int i = 0; i < ntot_amode; ++i) {
    printf(" %e", mact[i](k));
  }

  printf("\n");
  printf("B qcld %e \n", qcld(k));
  printf("B nsource %e \n", nsource(k));
  printf("B pdel %e \n", pdel(k));
  printf("B ncldwtr %e \n", ncldwtr(k));

  printf("\n");
  Kokkos::parallel_for(
      "ccncalc", top_lev - 1, KOKKOS_LAMBDA(int kk) {
        for (int i = 0; i < ncnst_tot; ++i) {
          qqcw_fld[i](kk) = zero;
        }
      });

  Kokkos::parallel_for(
      "ccncalc", pver - top_lev + 1, KOKKOS_LAMBDA(int kk) {
        const int k = kk + top_lev - 1;
        printf("ccncalc k %d \n", k);

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
            qqcwtend = (raercol_cw[nnew][mm](k) - qqcw_fld[mm](k)) * dtinv;
            qqcw_fld[mm](k) = haero::max(
                raercol_cw[nnew][mm](k),
                zero); // ! update cloud-borne aerosol; HW: ensure non-negative

            if (lspec == 0) {
              // Fortran indexing to C++ indexing
              const int num_idx = numptr_amode[imode] - 1;
              raertend = (raercol[nnew][mm](k) - state_q[num_idx](k)) * dtinv;
              if (k == print_this_k) {
                printf("lptr %d num_idx %d raercol[nnew][mm](k) %e \n", lptr,
                       num_idx, raercol[nnew][mm](k));
              }

              qcldbrn_num[imode] = qqcw_fld[mm](k);
            } else {
              // Fortran indexing to C++ indexing
              const int spc_idx = lmassptr_amode[lspec - 1][imode] - 1;
              raertend = (raercol[nnew][mm](k) - state_q[spc_idx](k)) * dtinv;
              //! Extract cloud borne MMRs from qqcw pointer
              qcldbrn[lspec][imode] = qqcw_fld[mm](k);
            } // end if
            if (k == print_this_k) {
              printf("lptr %d raertend %e \n", lptr, raertend);
            }
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

        Real state_q_kk[nvars];

        for (int i = 0; i < nvars; ++i) {
          // FIXME
          state_q_kk[i] = state_q[i](k);
        }
        Real ccn_kk[psat] = {};
        ccncalc(state_q_kk, temp(k), qcldbrn, qcldbrn_num, air_density,
                lspectype_amode, specdens_amode, spechygro, lmassptr_amode,
                voltonumbhi_amode, voltonumblo_amode, numptr_amode, nspec_amode,
                exp45logsig, alogsig, ccn_kk);

        for (int i = 0; i < psat; ++i) {
          ccn[i](k) = ccn_kk[i];
        }
      });

  printf("A tendnd %e \n", tendnd(print_this_k));

  printf(" A ptend_q \n");
  for (int i = nvar_ptend_q - 1; i < nvar_ptend_q; ++i) {
    printf(" %e ", ptend_q[i](print_this_k));
  }

  printf("\n");

} // dropmixnuc

} // namespace ndrop_od

} // end namespace mam4

#endif