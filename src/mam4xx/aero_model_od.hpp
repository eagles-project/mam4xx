// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MAM4XX_AERO_MODEL_OD_HPP
#define MAM4XX_AERO_MODEL_OD_HPP

#include <ekat/util/ekat_math_utils.hpp>

#include <haero/atmosphere.hpp>
#include <haero/math.hpp>

#include <mam4xx/aero_config.hpp>
#include <mam4xx/conversions.hpp>
#include <mam4xx/mam4_types.hpp>
#include <mam4xx/utils.hpp>

namespace mam4 {

namespace aero_model_od {

// BAD CONSTANT
const Real dlndg_nimptblgrow = haero::log(1.25);
constexpr int nimptblgrow_mind = -7, nimptblgrow_maxd = 12;
constexpr int nimptblgrow_total = -nimptblgrow_mind + nimptblgrow_maxd + 1;
const int nrainsvmax = 50; // maximum bin number for rain
const int naerosvmax = 51; //  maximum bin number for aerosol
const int maxd_aspectype = 14;

// NOTE: original FORTRAN function two internal loop over kk and icol.
// We removed these loop,so the inputs/outputs of  modal_aero_bcscavcoef_get are
// reals at the kk icol location
KOKKOS_INLINE_FUNCTION
void modal_aero_bcscavcoef_get(
    const int imode, const bool isprx_kk, const Real dgn_awet_imode_kk, //& ! in
    const Real dgnum_amode_imode,
    const Real scavimptblvol[nimptblgrow_total][AeroConfig::num_modes()],
    const Real scavimptblnum[nimptblgrow_total][AeroConfig::num_modes()],
    Real &scavcoefnum_kk, Real &scavcoefvol_kk) {
  // integer,  intent(in) :: imode, ncol
  // logical,  intent(in) :: isprx(pcols,pver)           ! if there is precip
  // real(r8), intent(in) :: dgn_awet(pcols,pver,ntot_amode)  ! wet aerosol
  // diameter [m] real(r8), intent(out):: scavcoefnum(pcols,pver)     !
  // scavenging removal for aerosol number [1/h] real(r8), intent(out)::
  // scavcoefvol(pcols,pver)     ! scavenging removal for aerosol volume [1/h]
  // !-----------------------------------------------------------------------
  // ! compute impaction scavenging removal amount for aerosol volume and number
  // !-----------------------------------------------------------------------
  // Note: We will do the loop over kk annd icol outside of these function
  // ! do only if there is precip
  // isprx_kk values of isprx at kk and icol
  const Real zero = 0;
  const Real one = 1;
  if (isprx_kk) {
    // ! interpolate table values using log of
    // (actual-wet-size)/(base-dry-size) ratio of wet and dry aerosol diameter
    // [fraction]
    const Real wetdiaratio = dgn_awet_imode_kk / dgnum_amode_imode;
    // BAD CONSTANT
    Real scavimpvol, scavimpnum = zero;
    if (wetdiaratio >= 0.99 && wetdiaratio <= 1.01) {
      scavimpvol = scavimptblvol[0][imode];
      scavimpnum = scavimptblnum[0][imode];
    } else {
      Real xgrow = haero::log(wetdiaratio) / dlndg_nimptblgrow;
      // FIXME check Fortran to C++ indexing conversion
      int jgrow = int(xgrow);
      if (xgrow < zero) {
        jgrow = jgrow - 1;
      }

      if (jgrow < nimptblgrow_mind) {
        jgrow = nimptblgrow_mind;
        xgrow = jgrow;
      } else {
        jgrow = haero::min(jgrow, nimptblgrow_maxd - 1);
      }
      const Real dumfhi = xgrow - jgrow;
      const Real dumflo = one - dumfhi;
      scavimpvol = dumflo * scavimptblvol[jgrow][imode] +
                   dumfhi * scavimptblvol[jgrow + 1][imode];
      scavimpnum = dumflo * scavimptblnum[jgrow][imode] +
                   dumfhi * scavimptblnum[jgrow + 1][imode];

    } /// wetdiaratio
      // ! impaction scavenging removal amount for volume
    scavcoefvol_kk = haero::exp(scavimpvol);
    // ! impaction scavenging removal amount to number
    scavcoefnum_kk = exp(scavimpnum);
  } else {
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
  //  temp   ! air temperature [K]
  return 1.8325e-4 * (416.16 / (temp + 120.)) * haero::pow(temp / 296.16, 1.5);

} // end air_dynamic_viscosity

KOKKOS_INLINE_FUNCTION
Real air_kinematic_viscosity(const Real temp, const Real rhoair) {
  /*-----------------------------------------------------------------
  ! Calculate kinematic viscosity of air, unit [cm^2/s]
  !-----------------------------------------------------------------*/
  // temp     ! air temperature [K]
  // rhoair   ! air density [g/cm3]
  // vsc_dyn_atm  ! dynamic viscosity of air [g/cm/s]
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
  // integer,  intent(in) :: nr           ! number of rain bins
  // real(r8), intent(in) :: rlo          ! lower limit of rain radius [cm]
  // real(r8), intent(in) :: dr           ! rain radius bin width [cm]
  // real(r8), intent(in) :: rhoair       ! air mass density [g/cm^3]
  // real(r8), intent(in) :: precip       ! precipitation [cm/s]

  // real(r8), intent(out) :: rrainsv(:)  ! rain radius in each bin [cm]
  // real(r8), intent(out) :: xnumrainsv(:)  ! rain number concentration in each
  // bin [#/cm3] real(r8), intent(out) :: vfallrainsv(:) ! rain droplet falling
  // velocity [cm/s]

  const Real zero = 0;
  Real precipsum = zero;
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
    // 1.333333 is simplified 4/3 for sphere volume calculation
  precipsum *= haero::Constants::pi * 4 / 3;
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

  //   integer,  intent(in) :: na           ! number of aerosol bins
  // real(r8), intent(in) :: xlo          ! lower limit of aerosol radius (log)
  // real(r8), intent(in) :: dx           ! aerosol radius bin width (log)
  // real(r8), intent(in) :: xg0          ! log(mean radius)
  // real(r8), intent(in) :: sx           ! standard deviation (log)

  // real(r8), intent(out):: raerosv(:)   ! aerosol radius [cm]
  // real(r8), intent(out):: fnumaerosv(:)! fraction of total number in the bin
  // [fraction] real(r8), intent(out):: fvolaerosv(:)! fraction of total volume
  // in the bin [fraction]
  //   ! local variables
  // integer  :: ii                       ! index of aerosol bins
  // real(r8) :: xx                       ! aerosol radius in the bin (log)
  // real(r8) :: aa                       ! aerosol radius in the bin [cm]
  // real(r8) :: dum                      ! working variable
  // real(r8) :: anumsum, avolsum         ! total aerosol number and volume

  // ! calculate total aerosol number and volume
  const Real zero = 0;
  // total aerosol number
  Real anumsum = zero;
  // total aerosol volume
  Real avolsum = zero;
  for (int ii = 0; ii < na; ++ii) {
    const Real xx = xlo + ii * dx;
    const Real aa = haero::exp(xx);
    raerosv[ii] = aa;
    const Real dum = (xx - xg0) / sx;
    fnumaerosv[ii] = haero::exp(-0.5 * dum * dum);
    // 1.3333 is simplified 4/3 for sphere volume calculation
    fvolaerosv[ii] =
        fnumaerosv[ii] * 4 * haero::Constants::pi * aa * aa * aa / 3;
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
  // BAD CONSTANT
  // FIXME get values of boltz_cgs

  // real(r8), intent(in)  :: freepath      ! molecular freepath [cm]
  // real(r8), intent(in)  :: r_aer         ! aerosol radius [cm]
  // real(r8), intent(in)  :: temp          ! temperature [K]
  // real(r8), intent(in)  :: rhoaero       ! density of aerosol particles
  // [g/cm^3] real(r8), intent(in)  :: rhoair        ! air mass density [g/cm^3]
  // real(r8), intent(in)  :: airkinvisc    ! air kinematic viscosity [cm2/s]

  // real(r8), intent(out) :: schmidt       ! Schmidt number [unitless]
  // real(r8), intent(out) :: taurelax      ! relaxation time for Stokes number
  // [s]

  // BAD CONSTANT
  //  GET this constant from haero.
  const Real boltz = 1.38065e-23;      //  ! Boltzmann's constant ~ J/K/molecule
  const Real boltz_cgs = boltz * 1.e7; // erg/K

  // working variables [unitless]
  const Real dum = freepath / r_aer;
  // ! slip correction factor [unitless]
  const Real dumfuchs = 1. + 1.246 * dum + 0.42 * dum * haero::exp(-0.87 / dum);
  taurelax =
      2. * rhoaero * r_aer * r_aer * dumfuchs / (9. * rhoair * airkinvisc);

  // single-particle aerosol mass [g]
  const Real aeromass =
      4. * haero::Constants::pi * r_aer * r_aer * r_aer * rhoaero / 3.; // ![g]
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
  // real(r8), intent(in)  :: r_aer         ! aerosol radius [cm]
  // real(r8), intent(in)  :: r_rain        ! rain radius [cm]
  // real(r8), intent(in)  :: temp          ! temperature [K]
  // real(r8), intent(in)  :: freepath      ! molecular freepath [cm]
  // real(r8), intent(in)  :: rhoaero       ! density of aerosol particles
  // [g/cm^3] real(r8), intent(in)  :: rhoair        ! air mass density [g/cm^3]
  // real(r8), intent(in)  :: airkinvisc    ! air kinematic viscosity [cm^2/s]
  // real(r8), intent(in)  :: vfall         ! rain droplet falling speed [cm/s]
  // real(r8), intent(out) :: etotal        ! efficiency of total effects
  // [fraction]

  // ! local variables
  // real(r8)  :: chi
  // real(r8)  :: dum, sstar   ! working variables [unitless]
  // real(r8)  :: taurelax     ! Stokes number relaxation time [s]
  // real(r8)  :: schmidt      ! Schmidt number [unitless]
  // real(r8)  :: stokes       ! Stokes number [unitless]
  // real(r8)  :: reynolds     ! Raynold number [unitless]
  // real(r8)  :: sqrtreynolds ! sqrt of Raynold number [unitless]
  // real(r8)  :: ebrown, eintercept, eimpact ! efficiency of aerosol-collection
  // in different processes
  const Real zero = 0;
  const Real one = 1;
  // BAD CONSTANT
  // ! ratio of water viscosity to air viscosity (from Slinn)
  const Real xmuwaterair = 60.0; // ! [fraction]
  // ! ratio of aerosol and rain radius [fraction]
  const Real chi = r_aer / r_rain;
  // ---------- calcualte Brown effect ------------

  Real schmidt = zero;
  Real taurelax = zero;

  // ! calculate unitless numbers
  calc_schmidt_number(freepath, r_aer, temp,       //       & ! in
                      rhoaero, rhoair, airkinvisc, //   & ! in
                      schmidt, taurelax);          // ! out

  const Real stokes = vfall * taurelax / r_rain;
  const Real reynolds = r_rain * vfall / airkinvisc;
  const Real sqrtreynolds = haero::sqrt(reynolds);
  const Real ebrown = 4. *
                      (one + 0.4 * sqrtreynolds * haero::pow(schmidt, 1 / 3)) /
                      (reynolds * schmidt);

  //------------ calculate intercept effect ------------
  Real dum =
      (one + 2. * xmuwaterair * chi) / (one + xmuwaterair / sqrtreynolds);
  const Real eintercept = 4. * chi * (chi + dum);

  // ! ------------ calculate impact effect ------------
  dum = haero::log(one + reynolds);
  const Real sstar = (1.2 + dum / 12.) / (one + dum);
  Real eimpact = zero;
  if (stokes > sstar) {
    dum = stokes - sstar;
    eimpact = haero::pow(dum / (dum + 0.6666667), 1.5);
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
                        Real &scavratevol,  // out
                        int &lunerr)        // out

{
  // this subroutine computes a single impaction scavenging rate
  //  for precipitation rate of 1 mm/h
  // FIXME: look for this constant at haero::Constants
  const Real SHR_CONST_BOLTZ =
      1.38065e-23; //  ! Boltzmann's constant ~ J/K/molecule
  const Real SHR_CONST_AVOGAD =
      6.02214e26; //   ! Avogadro's number ~ molecules/kmole
  const Real rgas_kmol =
      SHR_CONST_AVOGAD *
      SHR_CONST_BOLTZ; //       ! Universal gas constant ~ J/K/kmole
  const Real rgas = rgas_kmol * 1.e-3; //        ! Gas constant (J/K/mol)
  const Real pi = haero::Constants::pi;
  const Real zero = 0;
  const Real two = 2;

  //   function parameters
  // real(r8), intent(in)  :: dg0         ! geometric mean diameter of aerosol
  // [cm] real(r8), intent(in)  :: sigmag      ! geometric standard deviation of
  // size distribution real(r8), intent(in)  :: rhoaero     ! aerosol density
  // [g/cm^3] real(r8), intent(in)  :: temp        ! temperature [K] real(r8),
  // intent(in)  :: press       ! pressure [dyne/cm^2] real(r8), intent(out) ::
  // scavratenum, scavratevol  ! scavenging rate for aerosol number and volume
  // [1/hour] integer,  intent(out) :: lunerr      ! logical unit for error
  // message

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
  const Real precip = Real(1.0) / Real(36000.); //  1 mm/hr in cm/s

  // set the iteration radius for rain droplet
  // rain droplet bin information [cm]
  const Real rlo = .005;
  const Real rhi = .250;
  const Real dr = 0.005;
  // Nearest whole number: nint
  // FIXME: find this function in c++
  // number of rain bins
  const int nr = Real(1) + int((rhi - rlo) / dr);

  if (nr > nrainsvmax) {
    printf("subr. calc_1_impact_rate -- nr > nrainsvmax \n ");
    return;
  }

  // aerosol modal information
  // aerosol bin information
  const Real ag0 = dg0 / two; // mean radius of aerosol
  // standard deviation (log-normal distribution)
  const Real sx = haero::log(sigmag);
  // log(mean radius) (log-normal distribution)
  const Real xg0 = haero::log(ag0);
  const Real xg3 = xg0 + Real(3.) * sx * sx; // mean + 3*std^2

  // set the iteration radius for aerosol particles
  const Real dx = haero::max(0.2 * sx, 0.01);
  const Real xlo = xg3 - haero::max(4. * sx, 2. * dx);
  const Real xhi = xg3 + haero::max(4. * sx, 2. * dx);
  // Nearest whole number: nint
  // FIXME: find this function in c++
  const int na = 1 + int((xhi - xlo) / dx);

  if (na > naerosvmax) {
    printf("subr. calc_1_impact_rate -- na > naerosvmax \n ");
    return;
  }

  // air molar density [dyne/cm^2/erg*mol = mol/cm^3]
  // FIXME: look for a function call to compute density in mam4xx
  const Real cair = press / (rgas * temp);
  // air mass density [g/cm^3]
  // BAD CONSTANT
  const Real rhoair = 28.966 * cair;
  // !   molecular freepath [cm]
  // BAD CONSTANT
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
          xnumrainsv[jr] * 4 * pi * r_rain * r_rain * vfall;
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
    const Real specdens_amode[AeroConfig::num_modes()],
    const int lspectype_amode[maxd_aspectype][AeroConfig::num_modes()],
    Real scavimptblnum[nimptblgrow_total][AeroConfig::num_modes()],
    Real scavimptblvol[nimptblgrow_total][AeroConfig::num_modes()]) {
  //   !-----------------------------------------------------------------------
  // !
  // ! Purpose:
  // ! Computes lookup table for aerosol impaction/interception scavenging rates
  // !
  // ! Authors: R. Easter
  // !
  // !-----------------------------------------------------------------------
  const Real zero = 0;
  const Real one = 1;
  // Real sigmag_amode[4] = {zero};
  // int lspectype_amode[10][4] = {{0}};
  // Real specdens_amode[7] = {zero};

  int lunerr = 6; //           ! logical unit for error message

  // Real scavimptblnum[nimptblgrow_total][AeroConfig::num_modes()] = {{zero}};
  // Real scavimptblvol[nimptblgrow_total][AeroConfig::num_modes()] = {{zero}};

  // ! set up temperature-pressure pair to compute impaction scavenging rates
  // BAD CONSTANT
  const Real temp_0C = 273.16;      //     ! K
  const Real press_750hPa = 0.75e6; //  ! dynes/cm2
  for (int imode = 0; imode < AeroConfig::num_modes(); ++imode) {
    const Real sigmag = sigmag_amode[imode];
    // FIXME: can we get this aero density from mam4xx?
    const int ll = lspectype_amode[0][imode];
    printf("ll %d specdens_amode[ll] %e \n ", ll, specdens_amode[ll]);
    const Real rhodryaero = specdens_amode[ll];
    printf("nimptblgrow_mind %d \n ", nimptblgrow_mind);
    for (int jgrow = nimptblgrow_mind; jgrow < nimptblgrow_maxd; ++jgrow) {
      // printf("jgrow %d \n ", jgrow);
      // ratio of diameter for wet/dry aerosols [fraction]
      const Real wetdiaratio = haero::exp(jgrow * dlndg_nimptblgrow);
      // aerosol diameter [m]
      const Real dg0 = dgnum_amode[imode] * wetdiaratio;
      // ratio of volume for wet/dry aerosols [fraction]
      const Real wetvolratio = haero::exp(jgrow * dlndg_nimptblgrow * 3);
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
                         scavratenum, scavratevol, lunerr);

      scavimptblnum[jgrow - nimptblgrow_mind][imode] = haero::log(scavratenum);
      scavimptblvol[jgrow - nimptblgrow_mind][imode] = haero::log(scavratevol);

    } // jgrow

  } // end imode

} // modal_aero_bcscavcoef_init

} // end namespace aero_model_od

/// @class aero_model_od
/// This class implements MAM4's aero_model_od parameterization.
class AeroModelOD {
public:
  // aero_model_od-specific configuration
  struct Config {
    Config() {}
    Config(const Config &) = default;
    ~Config() = default;
    Config &operator=(const Config &) = default;
  };

private:
  Config config_;

public:
  // name--unique name of the process implemented by this class
  const char *name() const { return "MAM4 aero_model_od"; }
}; // end class aero_model_od

} // end namespace mam4

#endif
