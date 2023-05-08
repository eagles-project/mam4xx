#ifndef MAM4XX_NDROP_OD_HPP
#define MAM4XX_NDROP_OD_HPP

#include <ekat/util/ekat_math_utils.hpp>

#include <haero/atmosphere.hpp>
#include <haero/math.hpp>

#include <mam4xx/aero_config.hpp>
#include <mam4xx/conversions.hpp>
#include <mam4xx/mam4_types.hpp>
#include <mam4xx/utils.hpp>

namespace mam4 {

namespace ndrop_od {


const int psat = 6; //  ! number of supersaturations to calc ccn concentration
// FIXME: ask about state_q. Is it a prognostic variable? 
const int nvars = 40;
const int maxd_aspectype = 14;

KOKKOS_INLINE_FUNCTION
void get_aer_mmr_sum(const int imode,
                     const int nspec,
                     const Real state_q[nvars],
                     const Real qcldbrn1d[maxd_aspectype],
                     const int lspectype_amode[maxd_aspectype][AeroConfig::num_modes()],
                     const Real specdens_amode[maxd_aspectype],
                     const Real spechygro[maxd_aspectype],
                     const int lmassptr_amode[maxd_aspectype][AeroConfig::num_modes()],
                     Real &vaerosolsum_icol,
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
void get_aer_num(const int voltonumbhi_amode,
                 const int voltonumblo_amode,
                 const int num_idx,
                 const Real state_q[nvars],
                 const Real air_density,
                 const Real vaerosol,
                 const Real qcldbrn1d_num,
                 Real &naerosol) {

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
void loadaer(const Real state_q[nvars],
             const int nspec_amode[AeroConfig::num_modes()],
             Real air_density,
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

KOKKOS_INLINE_FUNCTION
void ccncalc(
    const Real state_q[nvars], 
    const Real tair,
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
    Real ccn[psat]) {

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

  loadaer(state_q,
          nspec_amode,
          air_density,
          phase,
          lspectype_amode,
          specdens_amode,
          spechygro,
          lmassptr_amode,
          voltonumbhi_amode,
          voltonumblo_amode,
          numptr_amode,
          qcldbrn,
          qcldbrn_num,
          naerosol,
          vaerosol,
          hygro);

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
void get_activate_frac(const Real state_q_kload[nvars],
                       const Real air_density_kload,
                       const Real air_density_kk,
                       const Real wtke,
                       const Real tair, // in
                       const int lspectype_amode[maxd_aspectype][AeroConfig::num_modes()],
                       const Real specdens_amode[maxd_aspectype],
                       const Real spechygro[maxd_aspectype],
                       const int lmassptr_amode[maxd_aspectype][AeroConfig::num_modes()],
                       const Real voltonumbhi_amode[AeroConfig::num_modes()],
                       const Real voltonumblo_amode[AeroConfig::num_modes()],
                       const int numptr_amode[AeroConfig::num_modes()],
                       const int nspec_amode[maxd_aspectype],
                       Real fn[AeroConfig::num_modes()],
                       Real fm[AeroConfig::num_modes()],
                       Real fluxn[AeroConfig::num_modes()],
                       Real fluxm[AeroConfig::num_modes()],
                       Real flux_fullact) {

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
  loadaer(state_q_kload,
          nspec_amode,
          air_density_kload,
          phase,
          lspectype_amode,
          specdens_amode,
          spechygro,
          lmassptr_amode,
          voltonumbhi_amode,
          voltonumblo_amode,
          numptr_amode,
          qcldbrn,
          qcldbrn_num,
          naermod,
          vaerosol,
          hygro);
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

} // namespace ndrop_od

} // end namespace mam4

#endif