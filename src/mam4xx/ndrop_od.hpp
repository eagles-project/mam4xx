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

// FIXME; surften is defined in ndrop_init
// BAD CONSTANT
const Real surften = 0.076;
const Real sq2 = haero::sqrt(2.);
const Real smcoefcoef = 2. / haero::sqrt(27.);

//// const Real percent_to_fraction = 0.01;
// super(:)=supersat(:)*percent_to_fraction
// supersaturation [fraction]
const Real super[psat] = {
    0.0002, 0.0005, 0.001, 0.002,
    0.005,  0.01}; //& ! supersaturation (%) to determine ccn concentration
//  [m-K]

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
    const int mam_idx[AeroConfig::num_modes()][10], Real raercol_nsav[40],
    Real raercol_nsav_kp1[40], Real raercol_cw_nsav[40],
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
  // [dimensionless] state_q_col_in(:,:)    ! aerosol mmrs [kg/kg]

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
        const int mm = mam_idx[imode][0];
        nact[imode] += fluxn[imode] * crdz * delz_cld;
        mact[imode] += fluxm[imode] * crdz * delz_cld;
        // ! note that kp1 is used here
        fluxntot +=
            fluxn[imode] * delz_cld * raercol_nsav_kp1[mm] * air_density;
      } // end imode

      nsource_col += fluxntot / (air_density * dz);
    } // end cldn_col_in(kk) - cldn_col_in(kp1) > cld_thresh
    else {

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
        int mm = mam_idx[imode][0];
        raercol_nsav[mm] += raercol_cw_nsav[mm]; // ! cloud-borne aerosol
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
    const int mam_idx[AeroConfig::num_modes()][10], Real &qcld,
    Real raercol_nsav[40],
    Real raercol_cw_nsav[40], //&      ! inout
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
      const int mm = mam_idx[imode][0];
      // cloud-borne aerosol tendency due to cloud frac tendency [#/kg or
      // kg/kg]
      const Real dact = raercol_cw_nsav[mm] * frac_delt_cld;
      raercol_cw_nsav[mm] += dact; //   ! cloud-borne aerosol
      raercol_nsav[mm] -= dact;
      for (int lspec = 0; lspec < nspec_amode[imode]; ++lspec) {
        const int mm = mam_idx[imode][lspec];
        const Real act = raercol_cw_nsav[mm] * frac_delt_cld;
        raercol_cw_nsav[mm] += dact; //  ! cloud-borne aerosol
        raercol_nsav[mm] -= act;
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
      const int mm = mam_idx[imode][0];
      const int num_idx = numptr_amode[imode];
      const Real dact = delt_cld * factnum_col_out[imode] *
                        state_q_col_in[num_idx]; // ! interstitial only
      qcld += dact;
      nsource_col_out += dact * dtinv;
      raercol_cw_nsav[mm] += dact; //  ! cloud-borne aerosol
      raercol_nsav[mm] -= dact;
      // fm change from fractional change in cloud fraction [fraction]
      const Real fm_delt_cld = delt_cld * fm[imode];

      for (int lspec = 0; lspec < nspec_amode[imode]; ++lspec) {
        const int mm = mam_idx[imode][lspec];
        const int spc_idx = lmassptr_amode[lspec][imode];
        const Real dact =
            fm_delt_cld * state_q_col_in[spc_idx]; // ! interstitial only
        raercol_cw_nsav[mm] += dact;               //  ! cloud-borne aerosol
        raercol_nsav[mm] -= dact;

      } // lspec
    }   // imode
  }     // cldn_col_in - cldo_col_in  > grow_cld_thresh

} // update_from_newcld

} // namespace ndrop_od

} // end namespace mam4

#endif