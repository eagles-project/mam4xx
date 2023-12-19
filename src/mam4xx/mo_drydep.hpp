#ifndef MAM4XX_MO_DRYDEP_HPP
#define MAM4XX_MO_DRYDEP_HPP

#include <haero/math.hpp>
#include <mam4xx/aero_config.hpp>
#include <mam4xx/mam4_types.hpp>
#include <mam4xx/utils.hpp>

namespace mam4::mo_drydep {

constexpr int gas_pcnst = mam4::gas_chemistry::gas_pcnst;
constexpr int n_land_type = 11; // from eam/src/chemistry/mozart/mo_drydep.F90

// FIXME: Where is seq_drydep_mod.F90??
constexpr int nddvels = 0; // FIXME

KOKKOS_INLINE_FUNCTION
void drydep_xactive(const int ncdate,               // date [YYMMDD]
                    const int col_index_season[12], // column-specific mapping of month indices to seasonal land-type indices [-]
                    const Real sfc_temp,            // surface temperature [K]
                    const Real air_temp,            // surface air temperature [K]
                    const Real tv,                  // potential temperature [K]
                    const Real pressure_sfc,        // surface pressure [Pa]
                    const Real pressure_10m,        // 10-meter pressure [Pa]
                    const Real spec_hum,            // specific humidity [kg/kg]
                    const Real wind_speed,          // 10-meter wind spped [m/s]
                    const Real rain,                // rain content [??]
                    const Real snow,                // snow height [m]
                    const Real solar_flux,          // direct shortwave surface radiation [W/m^2]
                    const Real mmr[gas_pcnst],      // constituent MMRs [kg/kg]
                    Real dvel[gas_pcnst],           // deposition velocity [1/cm/s]
                    Real dflx[gas_pcnst]) {         // deposition flux [1/cm^2/s]
  // BAD_CONSTANTS
  constexpr Real rain_threshold = 1e-7;   // of the order of 1cm/day [m/s]
  constexpr Real temp_highbound = 313.15; // [K]
  constexpr Real ric            = 0.2;    // ???

  // NOTE: our set of species is fixed, so in principle we know which ones can
  // NOTE: be deposited
  // FIXME: populate this array
  bool has_dvel[gas_pcnst] = {};

  for (int ispec = 0; ispec < gas_pcnst; ++ispec) {
    dvel[ispec] = 0.0;
  }

  //-------------------------------------------------------------------------------------
  // define species-dependent parameters (temperature dependent)
  //-------------------------------------------------------------------------------------
  Real heff[nddvels];
  seq_drydep_setHCoeff(sfc_temp, heff);

  Real dep_ra[n_land_type] = {}, dep_rb[n_land_type] = {};

  //-------------------------------------------------------------------------------------
  // 	... set month
  //-------------------------------------------------------------------------------------
  int month = (ncdate % 10000) / 100;

  //-------------------------------------------------------------------------------------
  // define which season (relative to Northern hemisphere climate)
  //-------------------------------------------------------------------------------------

  //-------------------------------------------------------------------------------------
  // define season index based on fixed LAI
  //-------------------------------------------------------------------------------------

  int index_season[n_land_type];
  for (int lt = 0; lt < n_land_type; ++lt) {
    index_season[lt] = index_season[month];
  }

  //-------------------------------------------------------------------------------------
  // special case for snow covered terrain
  //-------------------------------------------------------------------------------------
  if( snow > 0.01 ) { // BAD_CONSTANT
    for (int lt = 0; lt < n_land_type; ++lt) {
      index_season[lt] = 4;
    }
  }

  //-------------------------------------------------------------------------------------
  // scale rain and define logical arrays
  //-------------------------------------------------------------------------------------
  bool has_rain = (rain > rain_threshold);

  //-------------------------------------------------------------------------------------
  // potential temperature
  //-------------------------------------------------------------------------------------
  Real tha = get_potential_temperature(air_temp, pressure_10m, spec_hum);l
  Real thg = get_potential_temperature(sfc_temp, pressure_sfc, spec_hum);

  //-------------------------------------------------------------------------------------
  // height of 1st level
  //-------------------------------------------------------------------------------------
  // BAD_CONSTANTS
  Real zl = -rair/grav * air_temp * (1._r8 + .61_r8*spec_hum) * log(pressure_10m/pressure_sfc);

  //-------------------------------------------------------------------------------------
  // wind speed
  //-------------------------------------------------------------------------------------
  // BAD_CONSTANT
  Real va = haero::max(0.01, wind_speed);

  //-------------------------------------------------------------------------------------
  // Richardson number
  //-------------------------------------------------------------------------------------
  Real ribn = zl * grav * (tha - thg)/thg / (va*va);
  ribn = haero::min(ribn,ric);

  bool unstable = (ribn < 0.0);

  //-------------------------------------------------------------------------------------
  // saturation specific humidity
  //-------------------------------------------------------------------------------------
  Real qs = get_saturation_specific_humidity(sfc_temp, pressure_sfc);

  bool has_dew = false;
  if (qs <= spec_hum) {
    has_dew = true;
  }
  if (sfc_temp < tmelt) {
    has_dew = false;
  }

  //-------------------------------------------------------------------------------------
  // constant in determining rs
  //-------------------------------------------------------------------------------------
  Real tc = sfc_temp - tmelt;
  Real crs;
  if ((sfc_temp > tmelt) && (sfc_temp < temp_highbound)) { // BAD_CONSTANTS
    crs = (1.0 + haero::square(200.0/(solar_flux + 0.1))) * (400.0/(tc*(40.0 - tc)));
  } else {
    crs = large_value;
  }

  //-------------------------------------------------------------------------------------
  // rdc (lower canopy res)
  //-------------------------------------------------------------------------------------
  Real rdc = 100._r8*(1._r8 + 1000._r8/(solar_flux(icol) + 10._r8))
    enddo col_loop

    !-------------------------------------------------------------------------------------
    ! 	... form working arrays
    !-------------------------------------------------------------------------------------
    do lt = 1,n_land_type
       do icol=1,ncol
          lcl_frc_landuse(icol,lt) = fraction_landuse(icol,lt,lchnk)
       enddo
    enddo
    do lt = 1,n_land_type
       do icol=1,ncol
          fr_lnduse(icol,lt) = lcl_frc_landuse(icol,lt) > 0._r8
       enddo
    enddo


    call calculate_uustar(ncol, index_season, fr_lnduse, & ! in
                          unstable, lcl_frc_landuse, va, zl, ribn, &  ! in
                          uustar)                                     ! out

    call calculate_ustar(ncol, beglt, endlt, index_season, fr_lnduse, unstable, zl, uustar, ribn, &  ! in
                         ustar, cvar, bycp)                                                          ! out
  
    call calculate_ustar_over_water(ncol, beglt, endlt, index_season, fr_lnduse, unstable, zl, uustar, ribn, &  ! in
                                    ustar, cvar, bycp)                                                          ! inout

    call calculate_obukhov_length(ncol, beglt, endlt, fr_lnduse, unstable, tha, thg, ustar, cvar, va, bycp, ribn, & ! in
                                  obklen)                                                                           ! out 

    call calculate_aerodynamic_and_quasilaminar_resistance(ncol, beglt, endlt, fr_lnduse, zl, obklen, ustar, cvar, &  ! in
                                                           dep_ra(:,:,lchnk), dep_rb(:,:,lchnk))                      ! out

    !-------------------------------------------------------------------------------------
    ! surface resistance : depends on both land type and species
    ! land types are computed seperately, then resistance is computed as average of values
    ! following wesely rc=(1/(rs+rm) + 1/rlu +1/(rdc+rcl) + 1/(rac+rgs))**-1
    !
    ! compute rsmx = 1/(rs+rm) : multiply by 3 if surface is wet
    !-------------------------------------------------------------------------------------
    call calculate_resistance_rgsx_and_rsmx(ncol, beglt, endlt, index_season, fr_lnduse, has_rain, has_dew, &  ! in
                                            tc, heff, crs, &                                               ! in
                                            cts, rgsx, rsmx)                                               ! out
   
    call calculate_resistance_rclx(ncol, beglt, endlt, index_season, fr_lnduse, heff, cts, & ! in
                                   rclx)                                                     ! out

    call calculate_resistance_rlux(ncol, beglt, endlt, index_season, fr_lnduse, has_rain, has_dew, & ! in
                                   sfc_temp, qs, spec_hum, heff, cts, &                              ! in
                                   rlux)                                                             ! out


    term(:ncol) = 1.e-2_r8 * pressure_10m(:ncol) / (rair*tv(:ncol))
    call  calculate_gas_drydep_vlc_and_flux( ncol, beglt, endlt, index_season, fr_lnduse, lcl_frc_landuse, & ! in
                                             mmr, dep_ra(:,:,lchnk), dep_rb(:,:,lchnk), term, &              ! in
                                             rsmx, rlux, rclx, rgsx, rdc, &                                  ! in
                                             dvel, dflx)                                                     ! out
  
}

} // namespace mam4::mo_drydep

#endif

