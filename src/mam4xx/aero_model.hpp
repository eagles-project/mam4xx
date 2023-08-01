// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MAM4XX_AERO_MODEL_HPP
#define MAM4XX_AERO_MODEL_HPP

#include <ekat/util/ekat_math_utils.hpp>

#include <haero/atmosphere.hpp>
#include <haero/math.hpp>

#include <mam4xx/aero_config.hpp>
#include <mam4xx/conversions.hpp>
#include <mam4xx/mam4_types.hpp>
#include <mam4xx/utils.hpp>

namespace mam4 {

namespace aero_model {

// BAD CONSTANT
constexpr int nimptblgrow_mind = -7, nimptblgrow_maxd = 12;
constexpr int nimptblgrow_total = -nimptblgrow_mind + nimptblgrow_maxd + 1;
const int nrainsvmax = 50; // maximum bin number for rain
const int naerosvmax = 51; //  maximum bin number for aerosol
const int maxd_aspectype = 14;

KOKKOS_INLINE_FUNCTION
void modal_aero_bcscavcoef_get(
    const int imode, const bool isprx_kk, const Real dgn_awet_imode_kk, //& ! in
    const Real dgnum_amode_imode,
    const Real scavimptblvol[nimptblgrow_total][AeroConfig::num_modes()],
    const Real scavimptblnum[nimptblgrow_total][AeroConfig::num_modes()],
    Real &scavcoefnum_kk, Real &scavcoefvol_kk) {

  // !-----------------------------------------------------------------------
  // ! compute impaction scavenging removal amount for aerosol volume and number
  // !-----------------------------------------------------------------------

  // @param [in]  imode mode index
  // @param [in]  isprx_kk if there is precip
  // @param [in]  dgn_awet_imode_kk ! wet aerosol diameter of mode imode at
  // elevation kk [m]
  // @param [out] scavcoefnum scavenging removal for aerosol number [1/h]
  // @param [out] scavcoefvol scavenging removal for aerosol volume [1/h]

  // NOTE: original FORTRAN function has two internal loops: kk and icol.
  // We removed these loops. Hence, the inputs/outputs of
  // modal_aero_bcscavcoef_get are reals at kk , icol locations

  const Real zero = 0;
  const Real one = 1;
  // BAD CONSTANT
  const Real dlndg_nimptblgrow = haero::log(1.25);
  if (isprx_kk) {
    // With precipitation
    // interpolate table values using log of
    // (actual-wet-size)/(base-dry-size) ratio of wet and dry aerosol diameter
    // [fraction]
    const Real wetdiaratio = dgn_awet_imode_kk / dgnum_amode_imode;
    // Note: indexing in scavimptblnum and scavimptblvol
    // Fortran : [-7,12]
    // C++ :  [0,19]
    // Therefore, -7 (Fortran) => 0 (C++), or jgrow => jgrow - nimptblgrow_mind;
    // Here, we are assuming that nimptblgrow_mind is negative
    Real scavimpvol, scavimpnum = zero;
    // BAD CONSTANT
    if (wetdiaratio >= 0.99 && wetdiaratio <= 1.01) {
      // 8th position: Fortran (0) C++(7 or -nimptblgrow_mind)
      scavimpvol = scavimptblvol[-nimptblgrow_mind][imode];
      scavimpnum = scavimptblnum[-nimptblgrow_mind][imode];
    } else {
      Real xgrow = haero::log(wetdiaratio) / dlndg_nimptblgrow;
      int jgrow = int(xgrow); // get index jgrow
      if (xgrow < zero) { // // adjust jgrow appropriately if xgrow is negative
        jgrow = jgrow - 1;
      }
      // bound jgrow within max and min values
      if (jgrow < nimptblgrow_mind) {
        jgrow = nimptblgrow_mind;
        xgrow = jgrow;
      } else {
        jgrow = haero::min(jgrow, nimptblgrow_maxd - 1);
      }
      // compute factors for interpolating impaction scavenging removal amounts
      const Real dumfhi = xgrow - jgrow;
      const Real dumflo = one - dumfhi;
      // Fortran to C++ index conversion
      // Note: nimptblgrow_mind is negative (-7)
      int jgrow_pp = jgrow - nimptblgrow_mind;
      scavimpvol = dumflo * scavimptblvol[jgrow_pp][imode] +
                   dumfhi * scavimptblvol[jgrow_pp + 1][imode];
      scavimpnum = dumflo * scavimptblnum[jgrow_pp][imode] +
                   dumfhi * scavimptblnum[jgrow_pp + 1][imode];

    } // wetdiaratio

    // impaction scavenging removal amount for volume
    scavcoefvol_kk = haero::exp(scavimpvol);
    // impaction scavenging removal amount to number
    scavcoefnum_kk = haero::exp(scavimpnum);
  } else {
    // Without precipitation
    scavcoefvol_kk = zero;
    scavcoefnum_kk = zero;
  } // isprx_kk

} // modal_aero_bcscavcoef_get

KOKKOS_INLINE_FUNCTION
Real air_dynamic_viscosity(const Real temp) {
  /*-----------------------------------------------------------------
  ! Calculate dynamic viscosity of air, unit [g/cm/s]
  !
  ! note that this calculation is different with that used in dry deposition
  ! see the same-name function in modal_aero_drydep.F90
  !-----------------------------------------------------------------*/
  // @param [in] temp   ! air temperature [K]
  // @return     dynamic viscosity of air, unit [g/cm/s]
  // Note: We do not have a reference for this correlation.
  // However, this equation is presented in page 3 of
  // http://pages.erau.edu/~snivelyj/ep711sp12/EP711_15.pdf.
  return 1.8325e-4 * (416.16 / (temp + 120.)) * haero::pow(temp / 296.16, 1.5);

} // end air_dynamic_viscosity

KOKKOS_INLINE_FUNCTION
Real air_kinematic_viscosity(const Real temp, const Real rhoair) {
  /*-----------------------------------------------------------------
  ! Calculate kinematic viscosity of air, unit [cm^2/s]
  !-----------------------------------------------------------------*/
  // @param [in] temp     ! air temperature [K]
  // @param [in] rhoair   ! air density [g/cm3]
  // @return     vsc_dyn_atm  ! dynamic viscosity of air [g/cm/s]
  return air_dynamic_viscosity(temp) / rhoair;

} // air_kinematic_viscosity

KOKKOS_INLINE_FUNCTION
void calc_rain_drop_conc(const int nr, const Real rlo, const Real dr,
                         const Real rhoair,
                         const Real precip, //! in
                         Real rrainsv[nrainsvmax], Real xnumrainsv[nrainsvmax],
                         Real vfallrainsv[nrainsvmax]) {
  /*-----------------------------------------------------------------
  !   compute rain drop number concentrations, radius and falling velocity
  !-----------------------------------------------------------------*/
  // @param [in] nr           number of rain bins
  // @param [in] rlo          lower limit of rain radius [cm]
  // @param [in] dr           rain radius bin width [cm]
  // @param [in] rhoair       air mass density [g/cm^3]
  // @param [in] precip       precipitation [cm/s]

  // @param [out] rrainsv(:)   rain radius in each bin [cm]
  // @param [out] xnumrainsv(:)  rain number concentration in each
  // @param [out] vfallrainsv(:) rain droplet falling bin [#/cm3]
  // @param [out] velocity [cm/s] bin [#/cm3]

  const Real zero = 0;
  Real precipsum = zero;
  const Real four_thirds = 4. / 3.;
  // loop over  cloud bins
  for (int ii = 0; ii < nr; ++ii) {
    // rain radius in the bin [cm]
    const Real rr = rlo + ii * dr;
    rrainsv[ii] = rr;
    xnumrainsv[ii] = haero::exp(-rr / 2.7e-2);
    // rain diameter in the bin [cm]
    const Real dd = 2. * rr;
    Real vfallstp = zero;
    // BAD CONSTANT
    if (dd <= 0.007) {
      vfallstp = 2.88e5 * dd * dd;
    } else if (dd <= 0.025) {
      vfallstp = 2.8008e4 * haero::pow(dd, 1.528);
    } else if (dd <= 0.1) {
      vfallstp = 4104.9 * haero::pow(dd, 1.008);
    } else if (dd <= 0.25) {
      vfallstp = 1812.1 * haero::pow(dd, 0.638);
    } else {
      vfallstp = 1069.8 * haero::pow(dd, 0.235);
    }
    // rain droplet falling speed [cm/s]
    vfallrainsv[ii] = vfallstp * haero::sqrt(1.204e-3 / rhoair);
    // sum of precipitation in all bins
    precipsum += vfallrainsv[ii] * rr * rr * rr * xnumrainsv[ii];

  } // ii

  precipsum *= haero::Constants::pi * four_thirds;
  for (int ii = 0; ii < nr; ++ii) {
    xnumrainsv[ii] *= (precip / precipsum);
  }

} // calc_rain_drop_conc

KOKKOS_INLINE_FUNCTION
void calc_aer_conc_frac(const int na, const Real xlo, const Real dx,
                        const Real xg0,
                        const Real sx, // ! in
                        Real raerosv[naerosvmax], Real fnumaerosv[naerosvmax],
                        Real fvolaerosv[naerosvmax]) // out
{

  /*-----------------------------------------------------------------
  !   compute aerosol concentration, radius and volume in each bin
  !-----------------------------------------------------------------*/

  // @param [in]  na           ! number of aerosol bins
  // @param [in]  xlo          ! lower limit of aerosol radius (log)
  // @param [in]  dx           ! aerosol radius bin width (log)
  // @param [in]  xg0          ! log(mean radius)
  // @param [in]  sx           ! standard deviation (log)

  // @param [out] raerosv(:)   ! aerosol radius [cm]
  // @param [out] fnumaerosv(:)! fraction of total number in the bin
  // @param [out] fvolaerosv(:)! fraction of total volume
  // in the bin [fraction]

  // ! calculate total aerosol number and volume
  const Real zero = 0;
  const Real four_thirds = 4. / 3.;
  // total aerosol number
  Real anumsum = zero;
  // total aerosol volume
  Real avolsum = zero;
  for (int ii = 0; ii < na; ++ii) {
    const Real xx = xlo + ii * dx;
    // aerosol radius in the bin [cm]
    const Real aa = haero::exp(xx);
    raerosv[ii] = aa;
    const Real dum = (xx - xg0) / sx;
    fnumaerosv[ii] = haero::exp(-0.5 * dum * dum);
    fvolaerosv[ii] =
        four_thirds * fnumaerosv[ii] * haero::Constants::pi * aa * aa * aa;
    anumsum += fnumaerosv[ii];
    avolsum += fvolaerosv[ii];
  } // end ii

  // ! calculate fraction in each aerosol bin
  for (int ii = 0; ii < na; ++ii) {
    fnumaerosv[ii] /= anumsum;
    fvolaerosv[ii] /= avolsum;
  } // end ii

} // calc_aer_conc_frac

KOKKOS_INLINE_FUNCTION
void calc_schmidt_number(const Real freepath, const Real r_aer,
                         const Real temp, //& ! in
                         const Real rhoaero, const Real rhoair,
                         const Real airkinvisc,         // & ! in
                         Real &schmidt, Real &taurelax) //! out
{
  /*-----------------------------------------------------------------
  ! calculate Schmidt number
  ! also output relaxation time for Stokes number
  !
  ! note that there is a similar calculation of Schmidt number in dry
  ! depositon (in modal_aero_drydep.F90) but the calculation of dumfuchs (or
  ! slip_correction_factor) looks differently
  !-----------------------------------------------------------------*/

  // @param [in]  freepath      ! molecular freepath [cm]
  // @param [in]  r_aer         ! aerosol radius [cm]
  // @param [in]  temp          ! temperature [K]
  // @param [in]  rhoaero       ! density of aerosol particles[g/cm^3]
  // @param [in]  rhoair        ! air mass density [g/cm^3]
  // @param [in]  airkinvisc    ! air kinematic viscosity [cm2/s]

  // @param [out]  schmidt       ! Schmidt number [unitless]
  // @param [out]  taurelax      ! relaxation time for Stokes number
  // [s]

  // Unit conversion from J/K/molecule to erg/K
  const Real one = 1.;
  const Real two = 2.;
  const Real four_thirds = 4. / 3.;

  const Real boltz_cgs = haero::Constants::boltzmann * 1.e7; // erg/K

  // working variables [unitless]
  const Real dum = freepath / r_aer;
  // ! slip correction factor [unitless]
  const Real dumfuchs =
      one + 1.246 * dum + 0.42 * dum * haero::exp(-0.87 / dum);
  taurelax =
      two * rhoaero * r_aer * r_aer * dumfuchs / (9. * rhoair * airkinvisc);

  // single-particle aerosol mass [g]
  const Real aeromass = four_thirds * haero::Constants::pi * r_aer * r_aer *
                        r_aer * rhoaero; // ![g]
  // aerosol diffusivity [cm^2/s]
  const Real aerodiffus = boltz_cgs * temp * taurelax / aeromass; //  ! [cm^2/s]
  schmidt = airkinvisc / aerodiffus;
}

KOKKOS_INLINE_FUNCTION
void calc_impact_efficiency(const Real r_aer, const Real r_rain,
                            const Real temp, //   & ! in
                            const Real freepath, const Real rhoaero,
                            const Real rhoair,                       // & ! in
                            const Real vfall, const Real airkinvisc, // & ! in
                            Real &etotal) {

  /*-----------------------------------------------------------------
  ! calculate aerosol-collection efficiency for a given radius of rain and
  aerosol particles
  !-----------------------------------------------------------------*/
  // @param [in]  r_aer         ! aerosol radius [cm]
  // @param [in]  r_rain        ! rain radius [cm]
  // @param [in]  temp          ! temperature [K]
  // @param [in]  freepath      ! molecular freepath [cm]
  // @param [in]  rhoaero       ! density of aerosol particles
  // @param [in]  rhoair        ! air mass density [g/cm^3]
  // @param [in]  airkinvisc    ! air kinematic viscosity [cm^2/s]
  // @param [in]  vfall         ! rain droplet falling speed [cm/s]
  // @param [out] etotal        ! efficiency of total effects
  // [fraction]

  // ! local variables
  const Real zero = 0.;
  const Real one = 1.;
  const Real two = 2.;
  const Real one_third = 1. / 3.;
  const Real two_thirds = 2. / 3.;
  const Real four = 4.;
  // BAD CONSTANT
  // FIXME move this constant to hearo
  // ! ratio of water viscosity to air viscosity (from Slinn)
  const Real xmuwaterair = 60.0; // ! [fraction]
  // ! ratio of aerosol and rain radius [fraction]
  const Real chi = r_aer / r_rain;
  // ---------- calcualte Brown effect ------------
  // Schmidt number [unitless]
  Real schmidt = zero;
  // Stokes number relaxation time [s]
  Real taurelax = zero;

  // ! calculate unitless numbers
  calc_schmidt_number(freepath, r_aer, temp,       //       & ! in
                      rhoaero, rhoair, airkinvisc, //   & ! in
                      schmidt, taurelax);          // ! out
  // Stokes number [unitless]
  const Real stokes = vfall * taurelax / r_rain;
  // Reynolds number [unitless]
  const Real reynolds = r_rain * vfall / airkinvisc;
  const Real sqrtreynolds = haero::sqrt(reynolds);
  // efficiency of aerosol-collection  in different processes
  const Real ebrown =
      four * (one + 0.4 * sqrtreynolds * haero::pow(schmidt, one_third)) /
      (reynolds * schmidt);

  //------------ calculate intercept effect ------------
  Real dum =
      (one + two * xmuwaterair * chi) / (one + xmuwaterair / sqrtreynolds);
  const Real eintercept = four * chi * (chi + dum);

  // ! ------------ calculate impact effect ------------
  dum = haero::log(one + reynolds);
  const Real sstar = (1.2 + dum / 12.) / (one + dum);
  Real eimpact = zero;
  if (stokes > sstar) {
    dum = stokes - sstar;
    eimpact = haero::pow(dum / (dum + two_thirds), 1.5);
  }
  // ! ------------ calculate total effects ------------
  etotal = ebrown + eintercept + eimpact;

  etotal = haero::min(etotal, one);
} // calc_impact_efficiency

/*=====================================================================*/
KOKKOS_INLINE_FUNCTION
void calc_1_impact_rate(const Real dg0,     //  in
                        const Real sigmag,  //  in
                        const Real rhoaero, //  in
                        const Real temp,    //  in
                        const Real press,   //  in
                        Real &scavratenum,  // out
                        Real &scavratevol)  // out

{
  // this subroutine computes a single impaction scavenging rate
  //  for precipitation rate of 1 mm/h

  //   function parameters
  // @param [in]  dg0         geometric mean diameter of aerosol
  // @param [in]  sigmag      geometric standard deviation of
  // size distribution [cm]
  // @param [in]  rhoaero     aerosol density [g/cm^3]
  // @param [in]  temp        temperature [K] real(r8),
  // @param [in]  press       pressure [dyne/cm^2]
  // @param [out] scavratenum scavenging rate for aerosol number [1/hour]
  // @param [out] scavratenum scavenging rate for aerosol volume [1/hour]

  // const Real SHR_CONST_BOLTZ =
  //     1.38065e-23; //  ! Boltzmann's constant ~ J/K/molecule
  // const Real SHR_CONST_AVOGAD =
  //     6.02214e26; //   ! Avogadro's number ~ molecules/kmole
  // const Real rgas_kmol =
  //     SHR_CONST_AVOGAD *
  //     SHR_CONST_BOLTZ; //       ! Universal gas constant ~ J/K/kmole => ! Gas
  //                      //       constant (J/K/mol)
  // const Real rgas = rgas_kmol * 1.e-3 * 1.e7; //
  // air molar density [dyne/cm^2/erg*mol = mol/cm^3]
  // const Real cair = press / (rgas * temp);
  // air molar density [mol/cm^3]
  // const Real rhoair = 28.966 * cair;

  const Real pi = haero::Constants::pi;
  const Real zero = 0;
  const Real one = 1;
  const Real two = 2;
  const Real ten = 10;
  const Real one_thousand = 1000;
  const Real three = 3.;
  const Real four = 4.;

  // local variables
  Real rrainsv[nrainsvmax] = {zero};    // rain radius for each bin [cm]
  Real xnumrainsv[nrainsvmax] = {zero}; // rain number for each bin [#/cm3]
  Real vfallrainsv[nrainsvmax] = {
      zero}; // rain falling velocity for each bin [cm/s]

  Real raerosv[naerosvmax] = {zero}; // aerosol particle radius in each bin [cm]
  Real fnumaerosv[naerosvmax] = {
      zero}; // fraction of total number in the bin [fraction]
  Real fvolaerosv[naerosvmax] = {
      zero}; // fraction of total volume in the bin [fraction]

  // this subroutine is calculated for a fix rainrate of 1 mm/hr
  // precipitation rate, fix as 1 mm/hr in this subroutine [cm/s]
  const Real precip = one / Real(36000.); //  1 mm/hr in cm/s

  // set the iteration radius for rain droplet
  // rain droplet bin information [cm]
  // BAD CONSTANT
  const Real rlo = .005;
  // const Real rhi = .250;
  const Real dr = 0.005;
  // // Nearest whole number: nint
  // // number of rain bins
  // const int nr = 1 + haero::round((rhi - rlo) / dr);
  // FIXME: values to compute nr are hard-coded.
  const int nr = 50;
  // We comment this line because nr is hard-coded.
  // if (nr > nrainsvmax) {
  //   Kokkos::abort("subr. calc_1_impact_rate -- nr > nrainsvmax \n ");
  //   return;
  // }

  // aerosol modal information
  // aerosol bin information
  const Real ag0 = dg0 / two; // mean radius of aerosol
  // standard deviation (log-normal distribution)
  const Real sx = haero::log(sigmag);
  // log(mean radius) (log-normal distribution)
  const Real xg0 = haero::log(ag0);
  const Real xg3 = xg0 + three * sx * sx; // mean + 3*std^2
  // BAD CONSTANT
  // set the iteration radius for aerosol particles
  const Real dx = haero::max(0.2 * sx, 0.01);
  const Real xlo = xg3 - haero::max(four * sx, two * dx);
  const Real xhi = xg3 + haero::max(four * sx, two * dx);
  // Nearest whole number: nint
  const int na = 1 + haero::round((xhi - xlo) / dx);

  if (na > naerosvmax) {
    Kokkos::abort("subr. calc_1_impact_rate -- na > naerosvmax \n ");
  }

  // Note: pressure units are: ! dynes/cm2
  // We need pressure in units of Pa in density_of_ideal_gas
  // 10 dynes/cm2 = Pa
  const Real pressure = press / ten;
  // air mass density [g/cm^3]
  // unit conversion from [kg/m^3]/1000 to [g/cm^3]
  const Real rhoair = conversions::density_of_ideal_gas(
                          temp, pressure, Constants::r_gas_dry_air) /
                      one_thousand;
  // unit conversion from [mol g /cm^3/kg]/1000 to [mol/cm^3]
  // air molar density [mol/cm^3]
  const Real cair =
      rhoair / haero::Constants::molec_weight_dry_air / one_thousand;
  // !   molecular freepath [cm]
  // BAD CONSTANT
  // FIXME move this constant to haero
  const Real freepath = 2.8052e-10 / cair;
  // ! air kinematic viscosity [cm^2/s]
  const Real airkinvisc = air_kinematic_viscosity(temp, rhoair);

  // compute rain drop number concentrations
  calc_rain_drop_conc(nr, rlo, dr, rhoair, precip,       // ! in
                      rrainsv, xnumrainsv, vfallrainsv); // ! out

  // compute aerosol concentrations
  calc_aer_conc_frac(na, xlo, dx, xg0, sx,             //  in
                     raerosv, fnumaerosv, fvolaerosv); // out

  // compute scavenging

  Real scavsumnum = zero; // ! scavenging rate of aerosol number, "*bb" is for
                          // each rain droplet radius bin [1/s]
  Real scavsumvol = zero; //! scavenging rate of aerosol volume, "*bb" is for
                          //! each rain droplet radius bin [1/s]

  // outer loop for rain drop radius
  for (int jr = 0; jr < nr; ++jr) {
    // rain droplet radius
    // rain droplet and aerosol particle radius [cm]
    Real r_rain = rrainsv[jr];
    // rain droplet fall speed [cm/s]
    Real vfall = vfallrainsv[jr];
    // inner loop for aerosol particle radius
    Real scavsumnumbb = zero;
    Real scavsumvolbb = zero;
    for (int ja = 0; ja < na; ++ja) {
      // aerosol particle radius
      // rain droplet and aerosol particle radius [cm]
      const Real r_aer = raerosv[ja];
      Real etotal = zero; // efficiency of total scavenging effects [fraction]
      calc_impact_efficiency(r_aer, r_rain, temp,       // & ! in
                             freepath, rhoaero, rhoair, // & ! in
                             vfall, airkinvisc,         // & ! in
                             etotal);                   // out

      // rain droplet sweep out volume [cm3/cm3/s]
      const Real rainsweepout =
          xnumrainsv[jr] * four * pi * r_rain * r_rain * vfall;
      scavsumnumbb += rainsweepout * etotal * fnumaerosv[ja];
      scavsumvolbb += rainsweepout * etotal * fvolaerosv[ja];
    } // ja_loop

    scavsumnum += scavsumnumbb;
    scavsumvol += scavsumvolbb;

  } // jr_loop

  scavratenum = scavsumnum * 3600;
  scavratevol = scavsumvol * 3600;

} // end calc_1_impact_rate

KOKKOS_INLINE_FUNCTION
void modal_aero_bcscavcoef_init(
    const Real dgnum_amode[AeroConfig::num_modes()],
    const Real sigmag_amode[AeroConfig::num_modes()],
    const Real aerosol_dry_density[AeroConfig::num_modes()],
    Real scavimptblnum[nimptblgrow_total][AeroConfig::num_modes()],
    Real scavimptblvol[nimptblgrow_total][AeroConfig::num_modes()]) {
  // -----------------------------------------------------------------------
  //
  //  Purpose:
  //  Computes lookup table for aerosol impaction/interception scavenging rates
  //
  //  Authors: R. Easter
  //
  // -----------------------------------------------------------------------
  // @param [in]   dgnum_amode aerosol diameters [m]
  // @param [in]   sigmag_amode standard deviation of aerosol size distribution
  // @param [out]  scavimptblnum scavenging rate of aerosol number [1/s]
  // @param [out]  scavimptblnum scavenging rate of aerosol volume [1/s]
  // @aerosol_dry_density [in] aerosol dry density [k/m3]
  // FIXME : create an 4 elements array that contains the aerosol densities to
  // replace specdens_amode and lspectype_amode

  const Real zero = 0;
  const Real one = 1;
  const Real three = 3;
  // BAD CONSTANT
  const Real dlndg_nimptblgrow = haero::log(1.25);
  // ! set up temperature-pressure pair to compute impaction scavenging rates
  const Real temp_0C = haero::Constants::melting_pt_h2o; //     ! K
  const Real press_750hPa = 0.75e6;                      //  ! dynes/cm2
  for (int imode = 0; imode < AeroConfig::num_modes(); ++imode) {
    const Real sigmag = sigmag_amode[imode];
    // clang-format off
    // Note: we replaced lspectype_amode and lspectype_amode for
    // dry_aero_density 
    // const int ll = lspectype_amode[0][imode]; 
    // const Real rhodryaero = specdens_amode[ll];
    // clang-format on
    const Real rhodryaero = aerosol_dry_density[imode];
    for (int jgrow = nimptblgrow_mind; jgrow <= nimptblgrow_maxd; ++jgrow) {
      // ratio of diameter for wet/dry aerosols [fraction]
      const Real wetdiaratio = haero::exp(Real(jgrow) * dlndg_nimptblgrow);
      // aerosol diameter [m]
      const Real dg0 = dgnum_amode[imode] * wetdiaratio;
      // ratio of volume for wet/dry aerosols [fraction]
      const Real wetvolratio =
          haero::exp(Real(jgrow) * dlndg_nimptblgrow * three);
      // dry and wet aerosol density [kg/m3]
      Real rhowetaero = one + (rhodryaero - one) / wetvolratio;
      rhowetaero = haero::min(rhowetaero, rhodryaero);
      /* FIXME: not sure why wet aerosol density is set as dry aerosol density
         here ! but the above calculation of rhowetaero is incorrect. ! I think
         the number 1.0_r8 should be 1000._r8 as the unit is kg/m3 ! the above
         calculation gives wet aerosol density very small number (a few kg/m3) !
         this may cause some problem. I guess this is the reason of using dry
         density. ! should be better if fix the wet density bug and use it. Keep
         it for now for BFB testing ! -- (commented by Shuaiqi Tang when
         refactoring for MAM4xx) */

      rhowetaero = rhodryaero;
      /*compute impaction scavenging rates at 1 temp-press pair and save
              ! note that the subroutine calc_1_impact_rate uses CGS units */
      // aerosol diameter in CGS unit [cm]
      const Real dg0_cgs = dg0 * 1.0e2; //  ! m to cm
      // wet aerosol density in CGS unit [g/cm3]
      const Real rhowetaero_cgs = rhowetaero * 1.0e-3; //   ! kg/m3 to g/cm3
      // scavenging rate of aerosol number [1/s]
      Real scavratenum = zero;
      // scavenging rate of aerosol volume [1/s]
      Real scavratevol = zero;

      calc_1_impact_rate(dg0_cgs, sigmag, rhowetaero_cgs, temp_0C, press_750hPa,
                         scavratenum, scavratevol);

      scavimptblnum[jgrow - nimptblgrow_mind][imode] = haero::log(scavratenum);
      scavimptblvol[jgrow - nimptblgrow_mind][imode] = haero::log(scavratevol);

    } // jgrow

  } // end imode

} // modal_aero_bcscavcoef_init

// =============================================================================
KOKKOS_INLINE_FUNCTION
void define_act_frac(const int lphase, const int imode, Real &sol_facti,
                     Real &sol_factic, Real &sol_factb, Real &f_act_conv) {
  // clang-format off
  // -----------------------------------------------------------------------
  //  define sol_factb and sol_facti values, and f_act_conv
  // sol_factb - currently this is basically a tuning factor
  // sol_facti & sol_factic - currently has a physical basis, and
  // reflects activation fraction
  // f_act_conv is the activation fraction
  //
  // 2008-mar-07 rce - sol_factb (interstitial) changed from 0.3 to 0.1
  // - sol_factic (interstitial, dust modes) changed from 1.0 to 0.5
  // - sol_factic (cloud-borne, pcarb modes) no need to set it to 0.0
  // because the cloud-borne pcarbon == 0 (no activation)
  //
  // rce 2010/05/02
  // prior to this date, sol_factic was used for convective in-cloud wet removal,
  // and its value reflected a combination of an activation fraction
  // (which varied between modes) and a tuning factor
  // from this date forward, two parameters are used for convective
  // in-cloud wet removal
  //
  // note that "non-activation" of aerosol in air entrained into updrafts should
  // be included here
  // eventually we might use the activate routine (with w ~= 1 m/s) to calculate
  // this, but there is still the entrainment issue
  //
  // sol_factic is strictly a tuning factor
  //-----------------------------------------------------------------------
  /*

  in :: lphase ! index for interstitial / cloudborne aerosol
  in :: imode  ! index for aerosol mode

  out :: sol_facti  ! in-cloud scavenging fraction
  out :: sol_factb  ! below-cloud scavenging fraction
  out :: sol_factic ! in-cloud convective scavenging fraction
  out :: f_act_conv ! convection activation fraction
  */
  // clang-format on
  const int modeptr_pcarbon = static_cast<int>(mam4::ModeIndex::PrimaryCarbon);
  const Real sol_facti_cloud_borne = 1.0;
  if (lphase == 1) { // interstial aerosol
    sol_facti = 0.0; // strat in-cloud scav totally OFF for institial
    // if modal aero convproc is turned on for aerosols, then
    // turn off the convective in-cloud removal for interstitial aerosols
    // (but leave the below-cloud on, as convproc only does in-cloud)
    // and turn off the outfld SFWET, SFSIC, SFSID, SFSEC, and SFSED calls
    // for (stratiform)-cloudborne aerosols, convective wet removal
    // (all forms) is zero, so no action is needed
    sol_factic = 0.0;
    // all below-cloud scav ON (0.1 "tuning factor")
    sol_factb = 0.03;
    if (imode == modeptr_pcarbon)
      f_act_conv = 0.0;
    else
      f_act_conv = 0.4;

  } else {
    // cloud-borne aerosol (borne by stratiform cloud drops)
    // all below-cloud scav OFF (anything cloud-borne is located "in-cloud")
    sol_factb = 0.0;
    // strat  in-cloud scav totally ON for cloud-borne
    sol_facti = haero::min(0.6, sol_facti_cloud_borne);
    // conv   in-cloud scav OFF (having this on would mean
    // that conv precip collects strat droplets)
    sol_factic = 0.0;
    // conv   in-cloud scav OFF (having this on would mean
    f_act_conv = 0.0;
  }
}

// lmassptr_amode(l,m) = gchm r-array index for the mixing ratio
// (moles-x/mole-air) for chemical species l in aerosol mode m
// that is in clear air or interstitial air (but not in cloud water).
// If negative then number is not being simulated.
KOKKOS_INLINE_FUNCTION
int lmassptr_amode(const int i, const int j) {
  const int num_modes = AeroConfig::num_modes();
  const int lmassptr_amode[maxd_aspectype][num_modes] = {
      {15, 23, 28, 36}, {16, 24, 29, 37}, {17, 25, 30, 38}, {18, 26, 31, -1},
      {19, -1, 32, -1}, {20, -1, 33, -1}, {21, -1, 34, -1}, {-1, -1, -1, -1},
      {-1, -1, -1, -1}, {-1, -1, -1, -1}, {-1, -1, -1, -1}, {-1, -1, -1, -1},
      {-1, -1, -1, -1}, {-1, -1, -1, -1}};
  return lmassptr_amode[i][j];
}
// lmassptrcw_amode(l,m) = gchm r-array index for the mixing ratio
// (moles-x/mole-air) for chemical species l in aerosol mode m
// that is currently bound/dissolved in cloud water
KOKKOS_INLINE_FUNCTION
int lmassptrcw_amode(const int i, const int j) { return lmassptr_amode(i, j); }
// numptr_amode(m) = gchm r-array index for the number mixing ratio
// (particles/mole-air) for aerosol mode m that is in clear air or
// interstitial are (but not in cloud water).  If zero or negative,
// then number is not being simulated.
KOKKOS_INLINE_FUNCTION
int numptr_amode(const int i) {
  const int num_modes = AeroConfig::num_modes();
  const int numptr_amode[num_modes] = {22, 27, 35, 39};
  return numptr_amode[i];
}
KOKKOS_INLINE_FUNCTION
int numptrcw_amode(const int i) { return numptr_amode(i); }
// =============================================================================
KOKKOS_INLINE_FUNCTION
void index_ordering(const int lspec, const int imode, const int lphase, int &mm,
                    int &jnv, int &jnummaswtr) {
  // clang-format off
  //-----------------------------------------------------------------------
  // changed ordering (mass then number) for prevap resuspend to coarse
  //-----------------------------------------------------------------------
  /*
  in :: lspec  ! index for aerosol number / chem-mass / water-mass
  in :: imode  ! index for aerosol mode
  in :: lphase ! index for 1 == interstitial / 2 == cloudborne aerosol
  out :: mm         ! index of the tracers
  out :: jnv        ! index for scavcoefnv 3rd dimension
  out :: jnummaswtr ! indicates current aerosol species type (0 = number, 1 = dry mass, 2 = water)
  */
  // clang-format on
  const int jaeronumb = 0;
  const int jaeromass = 1;
  const int jaerowater = 2;
  const int nspec_amode = mam4::num_species_mode(imode);
  if (lspec < nspec_amode) { // non-water mass
    jnummaswtr = jaeromass;
    if (lphase == 1) {
      mm = lmassptr_amode(lspec, imode);
      jnv = 2;
    } else {
      mm = lmassptrcw_amode(lspec, imode);
      jnv = 0;
    }
  } else if (lspec == nspec_amode) { // number
    jnummaswtr = jaeronumb;
    if (lphase == 1) {
      mm = numptr_amode(imode);
      jnv = 1;
    } else {
      mm = numptrcw_amode(imode);
      jnv = 0;
    }
  } else { // water mass
    jnummaswtr = jaerowater;
  }
}

// =============================================================================
KOKKOS_INLINE_FUNCTION
bool examine_prec_exist(const int level_for_precipitation, const Real pdel[],
                        const Real prain[], const Real cmfdqr[],
                        const Real evapr[]) {
  // clang-format off
  // ----------------------------------------------------------------------
  // examine if level level_for_precipitation has precipitation.
  // ----------------------------------------------------------------------
  /*
  in :: pdel     ! pressure difference between two layers [Pa]
  in :: prain    ! rain production rate from stratiform clouds [kg/kg/s]
  in :: cmfdqr   ! dq/dt due to convective rainout [kg/kg/s]
  in :: evapr    ! rain evaporation rate [kg/kg/s]
  out :: examine_prec_exist   ! if there is precipitation falling into the level
  */
  // clang-format on
  // BAD CONSTANT
  const Real small_value_7 = 1.0e-7;
  const Real gravit = Constants::gravity;

  // initiate precipitation at the top level
  // precipitation falling from layers above [kg/m2/s]
  Real prec = 0;
  // check from the top level downward
  for (int k = 0; k < level_for_precipitation; ++k) {
    // update precipitation to the level below k
    prec += (prain[k] + cmfdqr[k] - evapr[k]) * pdel[k] / gravit;
  }
  const bool isprx = prec >= small_value_7;
  return isprx;
}

} // end namespace aero_model

} // end namespace mam4

#endif
