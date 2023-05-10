#ifndef MAM4XX_NDROP_MJS_HPP
#define MAM4XX_NDROP_MJS_HPP

#include <ekat/util/ekat_math_utils.hpp>

#include <haero/atmosphere.hpp>
#include <haero/math.hpp>

#include <mam4xx/aero_config.hpp>
#include <mam4xx/conversions.hpp>
#include <mam4xx/mam4_types.hpp>
#include <mam4xx/utils.hpp>

namespace mam4 {

namespace ndrop_mjs {

KOKKOS_INLINE_FUNCTION
void get_aer_mmr_sum(const int imode, const int nspec, const Real state_q[7],
                     const Real qcldbrn1d[7], const int lspectype_amode[7][7],
                     const Real specdens_amode[7], const Real spechygro[7],
                     const int lmassptr_amode[7][7], Real &vaerosolsum_icol,
                     Real &hygrosum_icol) {
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
    const int type_idx = lspectype_amode[lspec][imode];
    // density at species / mode indices [kg/m3]
    const Real density_sp = specdens_amode[type_idx]; //! species density
    // hygroscopicity at species / mode indices [dimensionless]
    const Real hygro_sp = spechygro[type_idx]; // !species hygroscopicity
    const int spc_idx =
        lmassptr_amode[lspec][imode]; //! index of species in state_q array
    // !aerosol volume mixing ratio [m3/kg]
    vaerosolsum_icol += haero::max(state_q[spc_idx] + qcldbrn1d[lspec], zero) /
                        density_sp;               // !volume = mmr/density
    hygrosum_icol += vaerosolsum_icol * hygro_sp; // !bulk hygroscopicity
  }                                               // end

} // end get_aer_mmr_sum

KOKKOS_INLINE_FUNCTION
void get_aer_num(const int voltonumbhi_amode, const int voltonumblo_amode,
                 const int num_idx, const Real state_q[7],
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
  // FIXME: num_idx for state_q: does this array contain only species?
  // or contains both species and modes concentrations ?
  naerosol = (state_q[num_idx] + qcldbrn1d_num) * air_density;
  //! adjust number so that dgnumlo < dgnum < dgnumhi
  naerosol = utils::min_max_bound(vaerosol * voltonumbhi_amode,
                                  vaerosol * voltonumblo_amode, naerosol);

} // end get_aer_num

KOKKOS_INLINE_FUNCTION
void loadaer(const Real state_q[7],
             const int nspec_amode[AeroConfig::num_modes()], Real air_density,
             const int phase, const int lspectype_amode[7][7],
             const Real specdens_amode[7], const Real spechygro[7],
             const int lmassptr_amode[7][7],
             const Real voltonumbhi_amode[AeroConfig::num_modes()],
             const Real voltonumblo_amode[AeroConfig::num_modes()],
             const int numptr_amode[AeroConfig::num_modes()],
             const Real qcldbrn1d[AeroConfig::num_modes()]
                                 [AeroConfig::num_aerosol_ids()],
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
      qcldbrn1d_imode[ispec] = qcldbrn1d[imode][ispec];
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
    const int num_idx = numptr_amode[imode];
    get_aer_num(voltonumbhi_amode[imode], voltonumblo_amode[imode], num_idx,
                state_q, air_density, vaerosol[imode], qcldbrn1d_num[imode],
                naerosol[imode]);

  } // end imode

} // loadaer

const int psat = 6; //  ! number of supersaturations to calc ccn concentration

KOKKOS_INLINE_FUNCTION
void ccncalc(
    const Real state_q[AeroConfig::num_aerosol_ids()], const Real tair,
    const Real qcldbrn[AeroConfig::num_modes()][AeroConfig::num_aerosol_ids()],
    const Real qcldbrn_num[AeroConfig::num_modes()], const Real air_density,
    const int lspectype_amode[7][7], const Real specdens_amode[7],
    const Real spechygro[7], const int lmassptr_amode[7][7],
    const Real voltonumbhi_amode[AeroConfig::num_modes()],
    const Real voltonumblo_amode[AeroConfig::num_modes()],
    const int numptr_amode[AeroConfig::num_modes()],
    const int nspec_amode[AeroConfig::num_modes()], Real ccn[psat]) {

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
  // qcldbrn(:,:,:,:) 1) icol 2) ispec 3) kk 4)imode
  // state_q(:,:,:) 1) icol 2) kk 3) imode or ispec ?
  const Real zero = 0;
  // BAD CONSTANT
  const Real nconc_thresh = 1.e-3;
  // phase of aerosol
  const int phase = 3; // ! interstitial+cloudborne
  const int nmodes = AeroConfig::num_modes();

  const Real mwh2o = haero::Constants::molec_weight_h2o * 1e3; // [kg/kmol]
  const Real r_universal = haero::Constants::r_gas * 1e3;      //[J/K/kmole]
  const Real rhoh2o = haero::Constants::density_h2o;
  const Real pi = haero::Constants::pi;

  // FIXME; surften is defined in ndrop_init
  // BAD CONSTANT
  const Real surften = 0.076;
  // FIME: drop_int
  Real exp45logsig[4] = {};
  Real alogsig[4] = {};
  for (int imode = 0; imode < 4; ++imode) {
    alogsig[imode] = haero::log(modes(imode).mean_std_dev);
    exp45logsig[imode] = haero::exp(4.5 * alogsig[imode] * alogsig[imode]);
  } // imode
  const Real sq2 = haero::sqrt(2.);
  //
  const Real twothird = 2. / 3.;

  // const Real percent_to_fraction = 0.01;
  const Real per_m3_to_per_cm3 = 1.e-6;
  const Real smcoefcoef = 2. / haero::sqrt(27.);

  // super(:)=supersat(:)*percent_to_fraction
  // supersaturation [fraction]
  const Real super[psat] = {
      0.0002, 0.0005, 0.001, 0.002,
      0.005,  0.001}; //& ! supersaturation (%) to determine ccn concentration
  //  [m-K]
  const Real surften_coef = 2. * mwh2o * surften / (r_universal * rhoh2o);

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

  ccn[psat] = {zero};
  for (int imode = 0; imode < nmodes; ++imode) {
    // here we assume that value shouldn't matter much since naerosol is small
    // endwhere
    Real sm = 1; // critical supersaturation at mode radius [fraction]
    if (naerosol[imode] > nconc_thresh) {
      // [dimensionless]
      const Real amcubecoef_imode = 3. / (4. * pi * exp45logsig[imode]);
      // [m3]
      const Real amcube = amcubecoef_imode * vaerosol[imode] / naerosol[imode];
      sm = smcoef /
           haero::sqrt(hygro[imode] * amcube); // ! critical supersaturation
    }

    const Real argfactor_imode = twothird / (sq2 * alogsig[imode]);
    for (int lsat = 0; lsat < psat; ++lsat) {
      // [dimensionless]
      const Real arg_erf_ccn = argfactor_imode * haero::log(sm / super[lsat]);
      ccn[lsat] += naerosol[imode] * 0.5 * (1. - haero::erf(arg_erf_ccn));
    }

  } // imode end

  for (int lsat = 0; lsat < psat; ++lsat) {
    ccn[lsat] *= per_m3_to_per_cm3; // ! convert from #/m3 to #/cm3
  }                                 // lsat

} /// ccncalc

KOKKOS_INLINE_FUNCTION
void qsat(const Real tair, const Real pres, Real es, Real qs) {
  // implementation
}

// FIXME;Jaelyn Litzinger is porting  maxsat
KOKKOS_INLINE_FUNCTION
void maxsat(const Real zeta, Real eta[4], const int nmode, const Real smc[4],
            Real smax) {}

KOKKOS_INLINE_FUNCTION
void activate_modal(const Real w_in, const Real wmaxf, const Real tair,
                    const Real rhoair, Real na[AeroConfig::num_modes()],
                    const Real volume[AeroConfig::num_modes()],
                    const Real hygro[AeroConfig::num_modes()],
                    Real fn[AeroConfig::num_modes()],
                    Real fm[AeroConfig::num_modes()],
                    Real fluxn[AeroConfig::num_modes()],
                    Real fluxm[AeroConfig::num_modes()], Real &flux_fullact)
//, const Real smax_prescribed=999
{
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
  // FIXME: hearo::Constants
  const Real zero = 0;
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
  const Real twothird = 2. / 3.;
  const Real r_universal = haero::Constants::r_gas * 1e3;      //[J/K/kmole]
  const Real mwh2o = haero::Constants::molec_weight_h2o * 1e3; // [kg/kmol]
  const Real t0 = 273; // reference temperature [K]
  // BAD CONSTANT
  const Real surften = 0.076;
  const Real aten = 2. * mwh2o * surften / (r_universal * t0 * rhoh2o);

  const Real p0 = 1013.25e2;              //  ! reference pressure [Pa]
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
      gravit * (latvap / (cpair * rh2o * tair * tair) - 1. / (rair * tair));
  // [m3/kg]
  const Real gamma = (1 + latvap / cpair * dqsdt) / (rhoair * qs);
  // [s^(3/2)]
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
      1. /
      (rhoh2o / (diff0 * rhoair * qs) +
       latvap * rhoh2o / (conduct0 * tair) *
           (latvap / (rh2o * tair) - 1.)); // gthermfac is same for all modes
  const Real beta = 2. * pi * rhoh2o * gthermfac * gamma; //[m2/s]
  // nucleation w, but = w_in if wdiab == 0 [m/s]
  const Real wnuc = w_in;
  const Real alw = alpha * wnuc;                  // [/s]
  const Real etafactor1 = alw * haero::sqrt(alw); // [/ s^(3/2)]
  // [unitless]
  const Real zeta = twothird * haero::sqrt(alw) * aten / haero::sqrt(gthermfac);

  Real amcube[nmode] = {}; // ! cube of dry mode radius [m3]

  // critical supersaturation for number mode radius [fraction]
  Real smc[nmode] = {};

  // FIXME
  // FIME: drop_int
  Real exp45logsig[nmode] = {};
  Real alogsig[nmode] = {};
  Real etafactor2[nmode] = {};
  Real lnsm[nmode] = {};
  for (int imode = 0; imode < nmode; ++imode) {
    alogsig[imode] = haero::log(modes(imode).mean_std_dev);
    exp45logsig[imode] = haero::exp(4.5 * alogsig[imode] * alogsig[imode]);
  } // imode

  Real eta[nmode] = {};
  // !Here compute smc, eta for all modes for maxsat calculation
  for (int imode = 0; imode < nmode; ++imode) {
    // BAD CONSTANT
    if (volume[imode] > 1.e-39 && na[imode] > 1.e-39) {
      // !number mode radius (m)
      amcube[imode] = (3. * volume[imode] /
                       (4. * pi * exp45logsig[imode] *
                        na[imode])); // ! only if variable size dist
      // !Growth coefficent Abdul-Razzak & Ghan 1998 eqn 16
      // !should depend on mean radius of mode to account for gas kinetic
      // effects !see Fountoukis and Nenes, JGR2005 and Meskhidze et al.,
      // JGR2006 !for approriate size to use for effective diffusivity.
      etafactor2[imode] = 1. / (na[imode] * beta * haero::sqrt(gthermfac));
      // BAD CONSTANT
      if (hygro[imode] > 1.e-10) {
        smc[imode] =
            2. * aten *
            haero::sqrt(aten / (27. * hygro[imode] *
                                amcube[imode])); // ! only if variable size dist
      } else {
        smc[imode] = 100.;
      } // hygro
    } else {
      smc[imode] = 1.;
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
  // FIXME;Jaelyn Litzinger is porting maxsat
  Real smax = zero;
  maxsat(zeta, eta, nmode, smc, smax);
  // endif
  // FIXME [unitless] ? lnsmax maybe has units of log(unit of smax ([fraction]))
  const Real lnsmax = haero::log(smax);
  const Real sq2 = haero::sqrt(2.);

  // !Use maximum supersaturation to calculate aerosol activation output
  for (int imode = 0; imode < 4; ++imode) {
    // ! [unitless]
    const Real arg_erf_n =
        twothird * (lnsm[imode] - lnsmax) / (sq2 * alogsig[imode]);

    fn[imode] = 0.5 * (1. - haero::erf(arg_erf_n)); //! activated number
    // ! [unitless]
    const Real arg_erf_m = arg_erf_n - 1.5 * sq2 * alogsig[imode];
    fm[imode] = 0.5 * (1. - haero::erf(arg_erf_m)); // !activated mass
    fluxn[imode] = fn[imode] * w_in; // !activated aerosol number flux
    fluxm[imode] = fm[imode] * w_in; // !activated aerosol mass flux
  }
  // FIXME: what is this??
  // is vertical velocity equal to flux of activated aerosol fraction assuming
  // 100% activation [m/s]?
  flux_fullact = w_in;

} // activate_modal
KOKKOS_INLINE_FUNCTION
void get_activate_frac(const Real state_q_kload[7],
                       const Real air_density_kload, const Real air_density_kk,
                       const Real wtke,
                       const Real tair, // in
                       const int lspectype_amode[7][7],
                       const Real specdens_amode[7], const Real spechygro[7],
                       const int lmassptr_amode[7][7],
                       const Real voltonumbhi_amode[AeroConfig::num_modes()],
                       const Real voltonumblo_amode[AeroConfig::num_modes()],
                       const int numptr_amode[AeroConfig::num_modes()],
                       const int nspec_amode[AeroConfig::num_modes()],
                       Real fn[AeroConfig::num_modes()],
                       Real fm[AeroConfig::num_modes()],
                       Real fluxn[AeroConfig::num_modes()],
                       Real fluxm[AeroConfig::num_modes()], Real flux_fullact) {

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
  const Real qcldbrn[AeroConfig::num_modes()][7] = {{zero}};
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
  activate_modal(wtke, wmax, tair, air_density_kk,    //   ! in
                 naermod, vaerosol, hygro,            //  ! in
                 fn, fm, fluxn, fluxm, flux_fullact); // out

} // get_activate_frac
KOKKOS_INLINE_FUNCTION
void update_from_newcld() {} // update_from_newcld
          // (cldn_col_in,cldo_col_in,dtinv, & // in
//        wtke_col_in,temp_col_in,cs_col_in,state_q_col_in, & // in
//        qcld,raercol_nsav,raercol_cw_nsav, &      // inout
//        nsource_col_out, factnum_col_out)              // inout

// // input arguments
// real(r8), intent(in) :: cldn_col_in(:)   // cloud fraction [fraction]
// real(r8), intent(in) :: cldo_col_in(:)   // cloud fraction on previous time step [fraction]
// real(r8), intent(in) :: dtinv     // inverse time step for microphysics [s^{-1}]
// real(r8), intent(in) :: wtke_col_in(:)   // subgrid vertical velocity [m/s]
// real(r8), intent(in) :: temp_col_in(:)   // temperature [K]
// real(r8), intent(in) :: cs_col_in(:)     // air density at actual level kk [kg/m^3]
// real(r8), intent(in) :: state_q_col_in(:,:) // aerosol mmrs [kg/kg]

// real(r8), intent(inout) :: qcld(:)  // cloud droplet number mixing ratio [#/kg]
// real(r8), intent(inout) :: nsource_col_out(:)   // droplet number mixing ratio source tendency [#/kg/s]
// real(r8), intent(inout) :: raercol_nsav(:,:)   // single column of saved aerosol mass, number mixing ratios [#/kg or kg/kg]
// real(r8), intent(inout) :: raercol_cw_nsav(:,:)  // same as raercol_nsav but for cloud-borne phase [#/kg or kg/kg]
// real(r8), intent(inout) :: factnum_col_out(:,:)  // activation fraction for aerosol number [fraction]


// //  local variables
// integer  :: imode           // mode counter variable
// integer  :: lspec           // species counter variable
// integer  :: mm              // local array index for MAM number, species
// integer  :: kk              // vertical level index
// integer  :: num_idx         // number index
// integer  :: spc_idx         // species index

// real(r8), parameter :: grow_cld_thresh = 0.01_r8   //  threshold cloud fraction growth [fraction]

// real(r8) :: delt_cld        // new - old cloud fraction [fraction]
// real(r8) :: frac_delt_cld   // fractional change in cloud fraction [fraction]
// real(r8) :: fm_delt_cld     // fm change from fractional change in cloud fraction [fraction]
// real(r8) :: dact             // cloud-borne aerosol tendency due to cloud frac tendency [#/kg or kg/kg]
// real(r8) :: fn(ntot_amode)              // activation fraction for aerosol number [fraction]
// real(r8) :: fm(ntot_amode)              // activation fraction for aerosol mass [fraction]
// real(r8) :: flux_fullact // flux of activated aerosol fraction assuming 100% activation [m/s]
// real(r8) :: fluxn(ntot_amode)     // flux of activated aerosol number fraction into cloud [m/s]
// real(r8) :: fluxm(ntot_amode)     // flux of activated aerosol mass fraction into cloud [m/s]


// // k-loop for growing/shrinking cloud calcs .............................
// do kk = top_lev, pver

//    delt_cld = (cldn_col_in(kk) - cldo_col_in(kk))

//    // shrinking cloud ......................................................
//    //    treat the reduction of cloud fraction from when cldn(i,k) < cldo(i,k)
//    //    and also dissipate the portion of the cloud that will be regenerated

//    if (cldn_col_in(kk) < cldo_col_in(kk)) then
//       //  droplet loss in decaying cloud
//       // ++ sungsup
//       nsource_col_out(kk) = nsource_col_out(kk) &
//            + qcld(kk)*(cldn_col_in(kk) - cldo_col_in(kk))/cldo_col_in(kk)*dtinv
//       qcld(kk) =  qcld(kk)*(1._r8 + (cldn_col_in(kk)-cldo_col_in(kk))/cldo_col_in(kk))
//       // -- sungsup

//       // convert activated aerosol to interstitial in decaying cloud

//       frac_delt_cld = (cldn_col_in(kk) - cldo_col_in(kk)) / cldo_col_in(kk)

//       do imode = 1, ntot_amode
//          mm = mam_idx(imode,0)
//          dact   = raercol_cw_nsav(kk,mm)*frac_delt_cld
//          raercol_cw_nsav(kk,mm) = raercol_cw_nsav(kk,mm) + dact   // cloud-borne aerosol
//          raercol_nsav(kk,mm)    = raercol_nsav(kk,mm) - dact
//          do lspec = 1, nspec_amode(imode)
//             mm = mam_idx(imode,lspec)
//             dact    = raercol_cw_nsav(kk,mm)*frac_delt_cld
//             raercol_cw_nsav(kk,mm) = raercol_cw_nsav(kk,mm) + dact  // cloud-borne aerosol
//             raercol_nsav(kk,mm)    = raercol_nsav(kk,mm) - dact
//          enddo
//       enddo
//    endif   // cldn(icol,kk) < cldo(icol,kk)

//    // growing cloud ......................................................
//    //    treat the increase of cloud fraction from when cldn(i,k) > cldo(i,k)
//    //    and also regenerate part of the cloud

//    if (cldn_col_in(kk)-cldo_col_in(kk) > grow_cld_thresh) then

//       call get_activate_frac(state_q_col_in(kk,:), cs_col_in(kk), cs_col_in(kk), & // in
//            wtke_col_in(kk), temp_col_in(kk), & // in
//            fn, fm, fluxn, fluxm, flux_fullact) // out

//       //  store for output activation fraction of aerosol
//       factnum_col_out(kk,:) = fn

//       do imode = 1, ntot_amode
//          mm = mam_idx(imode,0)
//          num_idx = numptr_amode(imode)
//          dact = delt_cld*fn(imode)*state_q_col_in(kk,num_idx) // interstitial only
//          qcld(kk) = qcld(kk) + dact
//          nsource_col_out(kk) = nsource_col_out(kk) + dact*dtinv
//          raercol_cw_nsav(kk,mm) = raercol_cw_nsav(kk,mm) + dact  // cloud-borne aerosol
//          raercol_nsav(kk,mm)    = raercol_nsav(kk,mm) - dact
//          fm_delt_cld = delt_cld * fm(imode)
//          do lspec = 1, nspec_amode(imode)
//             mm = mam_idx(imode,lspec)
//             spc_idx=lmassptr_amode(lspec,imode)
//             dact    = fm_delt_cld*state_q_col_in(kk,spc_idx) // interstitial only
//             raercol_cw_nsav(kk,mm) = raercol_cw_nsav(kk,mm) + dact  // cloud-borne aerosol
//             raercol_nsav(kk,mm)    = raercol_nsav(kk,mm) - dact
//          enddo
//       enddo
//    endif    //  cldn(icol,kk)-cldo(icol,kk) > 0.01_r8
// enddo  // end of k-loop for growing/shrinking cloud calcs ......................

KOKKOS_INLINE_FUNCTION
void update_from_cldn_profile() {}
//       cldn_col_in,dtinv,wtke_col_in,zs,dz,  &  // in
//      temp_col_in, cs_col_in,csbot_cscen,state_q_col_in,  &  // in
//      raercol_nsav,raercol_cw_nsav,nsource_col, &  // inout
//      qcld,factnum_col,ekd,nact,mact)  // inout

//   // input arguments
//   real(r8), intent(in) :: cldn_col_in(:)   // cloud fraction [fraction]
//   real(r8), intent(in) :: dtinv      // inverse time step for microphysics [s^{-1}]
//   real(r8), intent(in) :: wtke_col_in(:)   // subgrid vertical velocity [m/s]
//   real(r8), intent(in) :: zs(:)            // inverse of distance between levels [m^-1]
//   real(r8), intent(in) :: dz(:)         // geometric thickness of layers [m]
//   real(r8), intent(in) :: temp_col_in(:)   // temperature [K]
//   real(r8), intent(in) :: cs_col_in(:)     // air density [kg/m^3]
//   real(r8), intent(in) :: csbot_cscen(:)   // inverse normalized air density csbot(i)/cs(i,k) [dimensionless]
//   real(r8), intent(in) :: state_q_col_in(:,:)    // aerosol mmrs [kg/kg]

//   real(r8), intent(inout) :: raercol_nsav(:,:)    // single column of saved aerosol mass, number mixing ratios [#/kg or kg/kg]
//   real(r8), intent(inout) :: raercol_cw_nsav(:,:) // same as raercol but for cloud-borne phase [#/kg or kg/kg]
//   real(r8), intent(inout) :: nsource_col(:)   // droplet number mixing ratio source tendency [#/kg/s]
//   real(r8), intent(inout) :: qcld(:)  // cloud droplet number mixing ratio [#/kg]
//   real(r8), intent(inout) :: factnum_col(:,:)  // activation fraction for aerosol number [fraction]
//   real(r8), intent(inout) :: ekd(:)     // diffusivity for droplets [m^2/s]
//   real(r8), intent(inout) :: nact(:,:)  // fractional aero. number  activation rate [/s]
//   real(r8), intent(inout) :: mact(:,:)  // fractional aero. mass    activation rate [/s]

//   // local arguments

//   integer :: kk         // vertical level index
//   integer :: kp1        // bounded vertical level index + 1
//   integer :: imode      // mode counter variable
//   integer :: lspec      // species counter variable
//   integer :: mm         // local array index for MAM number, species

//   real(r8), parameter :: cld_thresh = 0.01_r8   //  threshold cloud fraction [fraction]

//   real(r8) :: delz_cld        // vertical change in cloud raction [fraction]
//   real(r8) :: crdz            // conversion factor from flux to rate [m^{-1}]
//   real(r8) :: fn(ntot_amode)              // activation fraction for aerosol number [fraction]
//   real(r8) :: fm(ntot_amode)              // activation fraction for aerosol mass [fraction]
//   real(r8) :: flux_fullact(pver) // flux of activated aerosol fraction assuming 100% activation [m/s]
//   real(r8) :: fluxn(ntot_amode)     // flux of activated aerosol number fraction into cloud [m/s]
//   real(r8) :: fluxm(ntot_amode)     // flux of activated aerosol mass fraction into cloud [m/s]
//   real(r8) :: fluxntot         // flux of activated aerosol number into cloud [#/m^2/s]


//   // ......................................................................
//   // start of k-loop for calc of old cloud activation tendencies ..........
//   //
//   // rce-comment
//   //    changed this part of code to use current cloud fraction (cldn) exclusively
//   //    consider case of cldo(:)=0, cldn(k)=1, cldn(k+1)=0
//   //    previous code (which used cldo below here) would have no cloud-base activation
//   //       into layer k.  however, activated particles in k mix out to k+1,
//   //       so they are incorrectly depleted with no replacement

//   do kk = top_lev, pver - 1

//      kp1 = min0(kk+1, pver)

//      if (cldn_col_in(kk) > cld_thresh) then

//         if (cldn_col_in(kk) - cldn_col_in(kp1) > cld_thresh ) then

//            // cloud base

//            // rce-comments
//            //   first, should probably have 1/zs(k) here rather than dz(i,k) because
//            //      the turbulent flux is proportional to ekd(k)*zs(k),
//            //      while the dz(i,k) is used to get flux divergences
//            //      and mixing ratio tendency/change
//            //   second and more importantly, using a single updraft velocity here
//            //      means having monodisperse turbulent updraft and downdrafts.
//            //      The sq2pi factor assumes a normal draft spectrum.
//            //      The fluxn/fluxm from activate must be consistent with the
//            //      fluxes calculated in explmix.
//            ekd(kk) = wtke_col_in(kk)/zs(kk)
//            // rce-comment - use kp1 here as old-cloud activation involves
//            //   aerosol from layer below

//            call get_activate_frac(state_q_col_in(kp1,:),cs_col_in(kp1),  &   // in
//                 cs_col_in(kk), wtke_col_in(kk), temp_col_in(kk),  &   // in
//                 fn, fm, fluxn, fluxm, flux_fullact(kk) )  // out

//            //  store for output activation fraction of aerosol
//            factnum_col(kk,:) = fn
//            delz_cld = cldn_col_in(kk) - cldn_col_in(kp1)
//            fluxntot = 0

//            // rce-comment 1
//            //    flux of activated mass into layer k (in kg/m2/s)
//            //       = "actmassflux" = dumc*fluxm*raercol(kp1,lmass)*csbot(k)
//            //    source of activated mass (in kg/kg/s) = flux divergence
//            //       = actmassflux/(cs(i,k)*dz(i,k))
//            //    so need factor of csbot_cscen = csbot(k)/cs(i,k)
//            //                   dum=1./(dz(i,k))
//            crdz=csbot_cscen(kk)/(dz(kk))
//            // rce-comment 2
//            //    code for k=pver was changed to use the following conceptual model
//            //    in k=pver, there can be no cloud-base activation unless one considers
//            //       a scenario such as the layer being partially cloudy,
//            //       with clear air at bottom and cloudy air at top
//            //    assume this scenario, and that the clear/cloudy portions mix with
//            //       a timescale taumix_internal = dz(i,pver)/wtke_cen(i,pver)
//            //    in the absence of other sources/sinks, qact (the activated particle
//            //       mixratio) attains a steady state value given by
//            //          qact_ss = fcloud*fact*qtot
//            //       where fcloud is cloud fraction, fact is activation fraction,
//            //       qtot=qact+qint, qint is interstitial particle mixratio
//            //    the activation rate (from mixing within the layer) can now be
//            //       written as
//            //          d(qact)/dt = (qact_ss - qact)/taumix_internal
//            //                     = qtot*(fcloud*fact*wtke/dz) - qact*(wtke/dz)
//            //    note that (fcloud*fact*wtke/dz) is equal to the nact/mact
//            //    also, d(qact)/dt can be negative.  in the code below
//            //       it is forced to be >= 0
//            //
//            // steve --
//            //    you will likely want to change this.  i did not really understand
//            //       what was previously being done in k=pver
//            //    in the cam3_5_3 code, wtke(i,pver) appears to be equal to the
//            //       droplet deposition velocity which is quite small
//            //    in the cam3_5_37 version, wtke is done differently and is much
//            //       larger in k=pver, so the activation is stronger there
//            //

//            do imode = 1, ntot_amode
//               mm = mam_idx(imode,0)
//               fluxn(imode) = fluxn(imode)*delz_cld
//               fluxm(imode) = fluxm(imode)*delz_cld
//               nact(kk,imode) = nact(kk,imode) + fluxn(imode)*crdz
//               mact(kk,imode) = mact(kk,imode) + fluxm(imode)*crdz
//               // note that kp1 is used here
//               fluxntot = fluxntot &
//                    + fluxn(imode)*raercol_nsav(kp1,mm)*cs_col_in(kk)
//            enddo
//            nsource_col(kk) = nsource_col(kk) + fluxntot/(cs_col_in(kk)*dz(kk))

//         endif  // (cldn(icol,kk) - cldn(icol,kp1) > 0.01)

//      else
//         //  if cldn < 0.01_r8 at any level except kk=pver, deplete qcld, turn all raercol_cw to raercol, put appropriate tendency
//         //  in nsource
//         //  Note that if cldn(kk) >= 0.01_r8 but cldn(kk) - cldn(kp1)  <= 0.01, nothing is done.

//         // no cloud

//         nsource_col(kk) = nsource_col(kk) - qcld(kk)*dtinv
//         qcld(kk)      = 0

//         // convert activated aerosol to interstitial in decaying cloud

//         do imode = 1, ntot_amode
//            mm = mam_idx(imode,0)
//            raercol_nsav(kk,mm)    = raercol_nsav(kk,mm) + raercol_cw_nsav(kk,mm)  // cloud-borne aerosol
//            raercol_cw_nsav(kk,mm) = 0._r8

//            do lspec = 1, nspec_amode(imode)
//               mm = mam_idx(imode,lspec)
//               raercol_nsav(kk,mm)    = raercol_nsav(kk,mm) + raercol_cw_nsav(kk,mm) // cloud-borne aerosol
//               raercol_cw_nsav(kk,mm) = 0._r8
//            enddo
//         enddo

//      endif  // (cldn(icol,kk) > 0.01_r8) if-else structure

//   enddo


// end subroutine update_from_cldn_profile
KOKKOS_INLINE_FUNCTION
void dropmixnuc() {}
//      lchnk,ncol,psetcols,dtmicro,temp,pmid,pint,pdel,rpdel,zm,   &  // in
//      state_q,ncldwtr,kvh,wsub,cldn,cldo, &  // in
//      qqcw, & // inout
//      ptend, tendnd, factnum)  // out

//   // vertical diffusion and nucleation of cloud droplets
//   // assume cloud presence controlled by cloud fraction
//   // doesn't distinguish between warm, cold clouds

//   use modal_aero_data,   only: qqcw_get_field, maxd_aspectype
//   use mam_support, only: min_max_bound

//   // input arguments
//   integer, intent(in)  :: lchnk               // chunk identifier
//   integer, intent(in)  :: ncol                // number of columns
//   integer, intent(in)  :: psetcols            // maximum number of columns
//   real(r8), intent(in) :: dtmicro     // time step for microphysics [s]
//   real(r8), intent(in) :: temp(:,:)    // temperature [K]
//   real(r8), intent(in) :: pmid(:,:)    // mid-level pressure [Pa]
//   real(r8), intent(in) :: pint(:,:)    // pressure at layer interfaces [Pa]
//   real(r8), intent(in) :: pdel(:,:)    // pressure thickess of layer [Pa]
//   real(r8), intent(in) :: rpdel(:,:)   // inverse of pressure thickess of layer [/Pa]
//   real(r8), intent(in) :: zm(:,:)      // geopotential height of level [m]
//   real(r8), intent(in) :: state_q(:,:,:) // aerosol mmrs [kg/kg]
//   real(r8), intent(in) :: ncldwtr(:,:) // initial droplet number mixing ratio [#/kg]
//   real(r8), intent(in) :: kvh(:,:)     // vertical diffusivity [m^2/s]
//   real(r8), intent(in) :: wsub(pcols,pver)    // subgrid vertical velocity [m/s]
//   real(r8), intent(in) :: cldn(pcols,pver)    // cloud fraction [fraction]
//   real(r8), intent(in) :: cldo(pcols,pver)    // cloud fraction on previous time step [fraction]

//   // inout arguments
//   type(ptr2d_t), intent(inout) :: qqcw(:)     // cloud-borne aerosol mass, number mixing ratios [#/kg or kg/kg]

//   // output arguments
//   type(physics_ptend), intent(out)   :: ptend
//   real(r8), intent(out) :: tendnd(pcols,pver) // tendency in droplet number mixing ratio [#/kg/s]
//   real(r8), intent(out) :: factnum(:,:,:)     // activation fraction for aerosol number [fraction]

//   // --------------------Local storage-------------------------------------

//   integer  :: mm                  // local array index for MAM number, species
//   integer  :: nnew, nsav          // indices for old, new time levels in substepping
//   integer  :: lptr
//   integer  :: ccn3d_idx   // index of ccn3d in pbuf
//   integer  :: icol        // column index
//   integer  :: imode       // mode index
//   integer  :: kk          // level index
//   integer  :: lspec      // species index for given mode
//   integer  :: lsat       //  level of supersaturation
//   integer  :: spc_idx, num_idx  // species, number indices

//   real(r8), parameter :: zkmin = 0.01_r8, zkmax = 100._r8  // min, max vertical diffusivity [m^2/s]
//   real(r8), parameter :: wmixmin = 0.1_r8        // minimum turbulence vertical velocity [m/s]

//   real(r8) :: dtinv      // inverse time step for microphysics [s^-1]
//   real(r8) :: raertend(pver)  // tendency of interstitial aerosol mass, number mixing ratios [#/kg/s or kg/kg/s]
//   real(r8) :: qqcwtend(pver)  // tendency of cloudborne aerosol mass, number mixing ratios [#/kg/s or kg/kg/s]
//   real(r8) :: zs(pver) // inverse of distance between levels [m^-1]
//   real(r8) :: qcld(pver) // cloud droplet number mixing ratio [#/kg]
//   real(r8) :: csbot(pver)       // air density at bottom (interface) of layer [kg/m^3]
//   real(r8) :: csbot_cscen(pver) // inverse normalized air density csbot(i)/cs(i,k) [dimensionless]
//   real(r8) :: zn(pver)   // g/pdel for layer [m^2/kg]
//   real(r8) :: ekd(pver)       // diffusivity for droplets [m^2/s]
//   real(r8) :: ndropcol(pcols)               // column-integrated droplet number [#/m2]
//   real(r8) :: cs(pcols,pver)      // air density [kg/m^3]
//   real(r8) :: dz(pver)      // geometric thickness of layers [m]
//   real(r8) :: wtke(pcols,pver)     // turbulent vertical velocity at base of layer k [m/s]
//   real(r8) :: nsource(pcols,pver)            // droplet number mixing ratio source tendency [#/kg/s]
//   real(r8) :: ndropmix(pcols,pver)           // droplet number mixing ratio tendency due to mixing [#/kg/s]
//   real(r8) :: ccn(pcols,pver,psat)    // number conc of aerosols activated at supersat [#/m^3]
//   real(r8) :: qcldbrn(pcols,pver,maxd_aspectype,ntot_amode) // // cloud-borne aerosol mass mixing ratios [kg/kg]
//   real(r8) :: qcldbrn_num(pcols,pver,ntot_amode) // // cloud-borne aerosol number mixing ratios [#/kg]


//   real(r8), pointer :: ccn3d(:, :)  //  CCN at 0.2% supersat [#/m^3]
//   real(r8), allocatable :: nact(:,:)  // fractional aero. number  activation rate [/s]
//   real(r8), allocatable :: mact(:,:)  // fractional aero. mass    activation rate [/s]
//   real(r8), allocatable :: raercol(:,:,:)    // single column of aerosol mass, number mixing ratios [#/kg or kg/kg]
//   real(r8), allocatable :: raercol_cw(:,:,:) // same as raercol but for cloud-borne phase [#/kg or kg/kg]

//   //     note:  activation fraction fluxes are defined as
//   //     fluxn = [flux of activated aero. number into cloud [#/m^2/s]]
//   //           / [aero. number conc. in updraft, just below cloudbase [#/m^3]]

//   real(r8), allocatable :: coltend(:,:)       // column tendency for diagnostic output
//   real(r8), allocatable :: coltend_cw(:,:)    // column tendency

// #include "../../chemistry/yaml/cam_ndrop/f90_yaml/dropmixnuc_beg_yml.f90"

//   // -------------------------------------------------------------------------------

//   // aerosol tendencies
//   call physics_ptend_init(ptend, psetcols, 'ndrop_aero', lq=lq)

//   //  Allocate / define local variables

//   allocate( &
//        nact(pver,ntot_amode),          &
//        mact(pver,ntot_amode),          &
//        raercol(pver,ncnst_tot,2),      &
//        raercol_cw(pver,ncnst_tot,2),   &
//        coltend(pcols,ncnst_tot),       &
//        coltend_cw(pcols,ncnst_tot)          )

//   dtinv = 1._r8/dtmicro

//   // initialize variables to zero
//   ndropmix(:,:) = 0._r8
//   nsource(:,:) = 0._r8
//   wtke(:,:)    = 0._r8
//   factnum(:,:,:) = 0._r8

//   // NOTE FOR C++ PORT: Get the cloud borne MMRs from AD in variable qcldbrn, do not port the code before END NOTE
//   qcldbrn(:,:,:,:) = huge(qcldbrn) // store invalid values
//   // END NOTE FOR C++ PORT

//   // overall_main_icol_loop
//   do icol = 1, ncol

//      nact(:,1:ntot_amode) = 0._r8
//      mact(:,1:ntot_amode) = 0._r8
//      cs(icol,:)  = pmid(icol,:)/(rair*temp(icol,:))        // air density (kg/m3)
//      dz(:)  = 1._r8/(cs(icol,:)*gravit*rpdel(icol,:)) // layer thickness in m
//      zn(:) = gravit*rpdel(icol,:)

//      wtke(icol,:)     = max(wsub(icol,:),wmixmin)

//      // load number nucleated into qcld on cloud boundaries

//      qcld(:)  = ncldwtr(icol,:)

//      do kk = top_lev, pver-1
//         zs(kk) = 1._r8/(zm(icol,kk) - zm(icol,kk+1))
//         ekd(kk)   = min_max_bound(zkmin,zkmax,kvh(icol,kk+1))
//         csbot(kk) = 2.0_r8*pint(icol,kk+1)/(rair*(temp(icol,kk) + temp(icol,kk+1)))
//         csbot_cscen(kk) = csbot(kk)/cs(icol,kk)
//      enddo
//      zs(pver) = zs(pver-1)
//      ekd(pver)   = 0._r8
//      csbot(pver) = cs(icol,pver)
//      csbot_cscen(pver) = 1.0_r8

//      //  Initialize 1D (in space) versions of interstitial and cloud borne aerosol

//      nsav = 1

//      do imode = 1, ntot_amode
//         mm = mam_idx(imode,0)
//         raercol_cw(:,mm,nsav) = 0.0_r8
//         raercol(:,mm,nsav)    = 0.0_r8
//         raercol_cw(top_lev:pver,mm,nsav) = qqcw(mm)%fld(icol,top_lev:pver)
//         num_idx = numptr_amode(imode)
//         raercol(top_lev:pver,mm,nsav) = state_q(icol,top_lev:pver,num_idx)
//         do lspec = 1, nspec_amode(imode)
//            mm = mam_idx(imode,lspec)
//            raercol_cw(top_lev:pver,mm,nsav) = qqcw(mm)%fld(icol,top_lev:pver)
//            spc_idx=lmassptr_amode(lspec,imode)
//            raercol(top_lev:pver,mm,nsav)    = state_q(icol,top_lev:pver,spc_idx)
//         enddo
//      enddo

//      //  PART I:  changes of aerosol and cloud water from temporal changes in cloud fraction
//      // droplet nucleation/aerosol activation

//      call update_from_newcld(cldn(icol,:),cldo(icol,:),dtinv,     &   // in
//           wtke(icol,:),temp(icol,:),cs(icol,:),state_q(icol,:,:),  &  // in
//           qcld(:),raercol(:,:,nsav),raercol_cw(:,:,nsav), &  // inout
//           nsource(icol,:), factnum(icol,:,:))  // inout

//      //  PART II: changes in aerosol and cloud water from vertical profile of new cloud fraction

//      call update_from_cldn_profile(cldn(icol,:),dtinv,wtke(icol,:),zs(:),dz(:),temp(icol,:), & // in
//           cs(icol,:),csbot_cscen(:),state_q(icol,:,:), & // in
//           raercol(:,:,nsav),raercol_cw(:,:,nsav), &  // inout
//           nsource(icol,:),qcld(:),factnum(icol,:,:),ekd(:),nact(:,:),mact(:,:))  // inout

//      //  PART III:  perform explict integration of droplet/aerosol mixing using substepping

//      nnew = 2

//      call update_from_explmix(dtmicro,csbot,cldn(icol,:),zn,zs,ekd,   &  // in
//           nact,mact,qcld,raercol,raercol_cw,nsav,nnew)       // inout
//      // droplet number

//      ndropcol(icol) = 0._r8
//      do kk = top_lev, pver
//         ndropmix(icol,kk) = (qcld(kk) - ncldwtr(icol,kk))*dtinv - nsource(icol,kk)
//         tendnd(icol,kk)   = (max(qcld(kk), 1.e-6_r8) - ncldwtr(icol,kk))*dtinv
//         ndropcol(icol)   = ndropcol(icol) + ncldwtr(icol,kk)*pdel(icol,kk)
//      enddo
//      ndropcol(icol) = ndropcol(icol)/gravit


//      raertend = 0._r8
//      qqcwtend = 0._r8

//      do imode = 1, ntot_amode
//         do lspec = 0, nspec_amode(imode)

//            mm   = mam_idx(imode,lspec)
//            lptr = mam_cnst_idx(imode,lspec)

//            qqcwtend(top_lev:pver) = (raercol_cw(top_lev:pver,mm,nnew) - qqcw(mm)%fld(icol,top_lev:pver))*dtinv
//            qqcw(mm)%fld(icol,:) = 0.0_r8
//            qqcw(mm)%fld(icol,top_lev:pver) = max(raercol_cw(top_lev:pver,mm,nnew),0.0_r8) // update cloud-borne aerosol; HW: ensure non-negative

//            if( lspec == 0 ) then
//               num_idx = numptr_amode(imode)
//               raertend(top_lev:pver) = (raercol(top_lev:pver,mm,nnew) - state_q(icol,top_lev:pver,num_idx))*dtinv
//               qcldbrn_num(icol,top_lev:pver,imode) = qqcw(mm)%fld(icol,top_lev:pver)
//            else
//               spc_idx=lmassptr_amode(lspec,imode)
//               raertend(top_lev:pver) = (raercol(top_lev:pver,mm,nnew) - state_q(icol,top_lev:pver,spc_idx))*dtinv
//               // Extract cloud borne MMRs from qqcw pointer
//               qcldbrn(icol,top_lev:pver,lspec,imode) = qqcw(mm)%fld(icol,top_lev:pver)
//            endif

//            coltend(icol,mm)    = sum( pdel(icol,:)*raertend )/gravit
//            coltend_cw(icol,mm) = sum( pdel(icol,:)*qqcwtend )/gravit

//            ptend%q(icol,:,lptr) = 0.0_r8
//            ptend%q(icol,top_lev:pver,lptr) = raertend(top_lev:pver)           // set tendencies for interstitial aerosol

//         enddo  // lspec loop
//      enddo   // imode loop

//   enddo  // overall_main_i_loop
//   // end of main loop over i/longitude ....................................

//   call outfld('NDROPCOL', ndropcol, pcols, lchnk)
//   call outfld('NDROPSRC', nsource,  pcols, lchnk)
//   call outfld('NDROPMIX', ndropmix, pcols, lchnk)
//   call outfld('WTKE    ', wtke,     pcols, lchnk)

//   //  Use interstitial and cloud-borne aerosol to compute output ccn fields.

//   call ccncalc(state_q, temp, qcldbrn, qcldbrn_num, ncol, cs, ccn)

//   do lsat = 1, psat
//      call outfld(ccn_name(lsat), ccn(1,1,lsat), pcols, lchnk)
//   enddo

//   // do column tendencies
//   do imode = 1, ntot_amode
//      do lspec = 0, nspec_amode(imode)
//         mm = mam_idx(imode,lspec)
//         call outfld(fieldname(mm),    coltend(:,mm),    pcols, lchnk)
//         call outfld(fieldname_cw(mm), coltend_cw(:,mm), pcols, lchnk)
//      enddo
//   enddo

//   deallocate( &
//        nact,       &
//        mact,       &
//        raercol,    &
//        raercol_cw, &
//        coltend,    &
//        coltend_cw       )

// #include "../../chemistry/yaml/cam_ndrop/f90_yaml/dropmixnuc_end_yml.f90"

// end subroutine dropmixnuc

} // namespace ndrop_mjs

} // end namespace mam4

#endif
