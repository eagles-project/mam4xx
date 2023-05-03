#ifndef MAM4XX_NDROP_OD_HPP
#define MAM4XX_NDROP_OD_HPP

namespace mam4 {

namespace ndrop_od {

KOKKOS_INLINE_FUNCTION
void get_aer_mmr_sum(const int imode, const int nspec,
                     const Real state_q_icol[7], const Real qcldbrn1d_icol[7],
                     const int lspectype_amode[7][7],
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
    vaerosolsum_icol +=
        haero::max(state_q_icol[spc_idx] + qcldbrn1d_icol[lspec], zero) /
        density_sp;                               // !volume = mmr/density
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
  naerosol = (state_q[num_idx] + qcldbrn1d_num) * air_density;
  //! adjust number so that dgnumlo < dgnum < dgnumhi
  naerosol = haero::max(naerosol, vaerosol * voltonumbhi_amode);
  naerosol = haero::min(naerosol, vaerosol * voltonumblo_amode);
} // end get_aer_num

KOKKOS_INLINE_FUNCTION
void loadaer(const Real state_q[7], const int imode, const int nspec,
             Real air_density, const int phase, const int lspectype_amode[7][7],
             const Real specdens_amode[7], const Real spechygro[7],
             const int lmassptr_amode[7][7], const Real voltonumbhi_amode,
             const Real voltonumblo_amode, const int num_idx, Real qcldbrn1d[7],
             Real &qcldbrn1d_num, Real &naerosol, Real &vaerosol, Real &hygro) {
  // return aerosol number, volume concentrations, and bulk hygroscopicity
  // aerosol mmrs [kg/kg]
  //
  // @param [in] istart      ! start column index (1 <= istart <= istop <=
  // pcols)
  // @param [in] istop       ! stop column index
  // @param [in] imode       ! mode index
  // @param [in] nspec       ! total # of species in mode imode
  // @param [in] k_in          ! level index
  // @param [in] air_density(:,:)     ! air density [kg/m3]
  // @param [in] phase       ! phase of aerosol: 1 for interstitial, 2 for
  // cloud-borne, 3 for sum

  // output arguments
  // @param [out] naerosol(:)  ! number conc [#/m3]
  // @param [out]  vaerosol(:)  ! volume conc [m3/m3]
  // @param [out]  hygro(:)     ! bulk hygroscopicity of mode [dimensionless]
  // optional input arguments
  // @param [in]  qcldbrn1d(:,:), qcldbrn1d_num(:) ! ! cloud-borne aerosol mass
  // / number  mixing ratios [kg/kg or #/kg]

  // vaerosolsum(pcols)  ! sum to find volume conc [m3/kg]
  // hygrosum(pcols)     ! sum to bulk hygroscopicity of mode [m3/kg]
  // qcldbrn_local(pcols,nspec)  ! local cloud-borne aerosol mass mixing ratios
  // [kg/kg] qcldbrn_num_local(pcols) ! local cloud-borne aerosol number mixing
  // ratios [#/kg]

  // !Currenly supports only phase 1 (interstitial) and 3 (interstitial+cldbrn)
  if (phase != 1 && phase != 3) {
    // write(iulog,*)'phase=',phase,' in loadaer'
    Kokkos::abort("phase error in loadaer");
  }

  // assume is present
  // qcldbrn_local(:,:nspec) = qcldbrn1d(:,:nspec)

  // Sum over all species within imode to get bulk hygroscopicity and volume
  // conc phase == 1 is interstitial only. phase == 3 is interstitial + cldborne
  // Assumes iphase =1 or 3, so interstitial is always summed, added with cldbrn
  // when present iphase = 2 would require alternate logic from following
  // subroutine
  const Real zero = 0;
  Real vaerosolsum_icol = zero;
  Real hygrosum_icol = zero;

  get_aer_mmr_sum(imode, nspec, state_q, qcldbrn1d, lspectype_amode,
                  specdens_amode, spechygro, lmassptr_amode, vaerosolsum_icol,
                  hygrosum_icol);
  //  Finalize computation of bulk hygrospopicity and volume conc
  // BAD CONSTANT
  // FIXME
  if (vaerosolsum_icol < 1.0e-30) {
    hygro = hygrosum_icol / vaerosolsum_icol;
    vaerosol = vaerosolsum_icol * air_density;
  } else {
    hygro = zero;
    vaerosol = zero;
  }

  // ! Compute aerosol number concentration
  get_aer_num(voltonumbhi_amode, voltonumblo_amode, num_idx, state_q,
              air_density, vaerosol, qcldbrn1d_num, naerosol);

} // loadaer

const int psat = 6; //  ! number of supersaturations to calc ccn concentration

KOKKOS_INLINE_FUNCTION
void ccncalc(const Real state_q[7], const Real tair, const Real qcldbrn[7][4],
             const Real qcldbrn_num[4], const Real air_density,
             const int lspectype_amode[7][7], const Real specdens_amode[7],
             const Real spechygro[7], const int lmassptr_amode[7][7],
             const Real voltonumbhi_amode, const Real voltonumblo_amode,
             const int numptr_amode[4], Real ccn[psat]) {
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

  // SHR_CONST_MWWV    = 18.016; //       ! molecular weight water vapor
  const Real mwh2o = haero::Constants::molec_weight_h2o * 1e3; // [kg/kmol]
  // SHR_CONST_BOLTZ   = 1.38065e-23_R8  ! Boltzmann's constant ~ J/K/molecule
  // SHR_CONST_AVOGAD  = 6.02214e26_R8   ! Avogadro's number ~ molecules/kmole
  // SHR_CONST_RGAS    = SHR_CONST_AVOGAD*SHR_CONST_BOLTZ       ! Universal gas
  // constant ~ J/K/kmole
  const Real r_universal = haero::Constants::r_gas * 1e3; //[J/K/kmole]
  // SHR_CONST_RHOFW   = 1.000e3_R8      ! density of fresh water     ~ kg/m^3
  const Real rhoh2o = haero::Constants::density_h2o;
  const Real pi = haero::Constants::pi;

  // FIXME; surften is defined in ndrop_init
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
  // phase of aerosol
  const int phase = 3; // ! interstitial+cloudborne
  // FIXME number of modes
  ccn[psat] = {zero};
  Real qcldbrn_imode[7] = {};
  Real qcldbrn_num_imode = zero;

  Real naerosol = zero; // ! interstit+activated aerosol number conc [#/m3]
  Real vaerosol = zero; // ! interstit+activated aerosol volume conc [m3/m3]

  Real sm = zero; // critical supersaturation at mode radius [fraction]
  Real hygro = zero;
  for (int imode = 0; imode < 4; ++imode) {
    // [dimensionless]
    const Real amcubecoef_imode = 3. / (4. * pi * exp45logsig[imode]);
    // [dimensionless]
    const Real argfactor_imode = twothird / (sq2 * alogsig[imode]);

    // FIXME
    const Real nspec = 7;
    const int num_idx = numptr_amode[imode];
    for (int ispec = 0; ispec < nspec; ++ispec) {
      qcldbrn_imode[ispec] = qcldbrn[ispec][imode];
    }
    qcldbrn_num_imode = qcldbrn_num[imode];

    loadaer(state_q, imode, nspec, air_density, phase, lspectype_amode,
            specdens_amode, spechygro, lmassptr_amode, voltonumbhi_amode,
            voltonumblo_amode, num_idx, qcldbrn_imode, qcldbrn_num_imode,
            naerosol, vaerosol, hygro);

    // BAD CONSTANT
    if (naerosol > 1e-3) {
      // [m3]
      const Real amcube = amcubecoef_imode * vaerosol / naerosol;
      sm = smcoef / haero::sqrt(hygro * amcube); // ! critical supersaturation
    } else {
      // value shouldn't matter much since naerosol is small endwhere
      sm = 1;
    }

    for (int lsat = 0; lsat < psat; ++lsat) {
      // [dimensionless]
      const Real arg_erf_ccn = argfactor_imode * haero::log(sm / super[lsat]);
      ccn[lsat] += naerosol * 0.5 * (1. - haero::erf(arg_erf_ccn));
    }

  } // imode end

  for (int lsat = 0; lsat < psat; ++lsat) {
    ccn[lsat] *= per_m3_to_per_cm3; // ! convert from #/m3 to #/cm3
  }                                 // lsat

} /// ccncalc

} // namespace ndrop_od

} // end namespace mam4

#endif