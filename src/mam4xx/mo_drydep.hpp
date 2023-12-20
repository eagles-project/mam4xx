#ifndef MAM4XX_MO_DRYDEP_HPP
#define MAM4XX_MO_DRYDEP_HPP

#include <haero/math.hpp>
#include <mam4xx/aero_config.hpp>
#include <mam4xx/mam4_types.hpp>
#include <mam4xx/utils.hpp>

namespace mam4::mo_drydep {

constexpr int gas_pcnst = mam4::gas_chemistry::gas_pcnst;
constexpr int n_land_type = 11; // from eam/src/chemistry/mozart/mo_drydep.F90
constexpr int lt_for_water = 7; // from eam/src/chemistry/mozart/mo_drydep.F90
constexpr int so2_ndx = -1;     // FIXME

// BAD_CONSTANTS
constexpr Real large_value = 1e36;
constexpr Real rair = 287.04;
constexpr Real grav = 9.81;
constexpr Real karman = 0.4;   // from shr_const_mod.F90
constexpr Real tmelt = 273.15; // from shr_const_mod.F90 via physconst.F90

// FIXME: Where is seq_drydep_mod.F90??
constexpr int nddvels = 0; // FIXME

KOKKOS_INLINE_FUNCTION
void calculate_uustar(
    const int index_season[n_land_type], const bool fr_lnduse[n_land_type],
    const bool unstable, const Real lcl_frc_landuse[n_land_type],
    const Real va,   // magnitude of v on cross points
    const Real zl,   // height of lowest level
    const Real ribn, // richardson number [-]
    Real &uustar) {  // u * ustar (assumed constant over grid) [m^2/s^2]
  // FIXME: use seq_drydep_mod, only: z0

  //-------------------------------------------------------------------------------------
  // find grid averaged z0: z0bar (the roughness length)
  // z_o=exp[S(f_i*ln(z_oi))]
  // this is calculated so as to find u_i, assuming u*u=u_i*u_i
  //-------------------------------------------------------------------------------------
  Real z0b = 0.0; // average roughness length over grid
  for (int lt = 0; lt < n_land_type; ++lt) {
    if (fr_lnduse[lt]) {
      // FIXME: z0 may need its indices switched?
      z0b += lcl_frc_landuse[lt] * log(z0[index_season[lt]][lt]);
    }
  }

  //-------------------------------------------------------------------------------------
  // find the constant velocity uu*=(u_i)(u*_i)
  //-------------------------------------------------------------------------------------
  z0b = haero::exp(z0b);
  Real cvarb = karman / log(zl / z0b);

  //-------------------------------------------------------------------------------------
  // unstable and stable cases
  //-------------------------------------------------------------------------------------
  Real ustarb;
  if (unstable) {
    Real bb = 9.4 * (cvarb * cvarb) * haero::sqrt(haero::abs(ribn) * zl / z0b);
    ustarb =
        cvarb * va *
        haero::sqrt(1.0 - (9.4 * ribn / (1.0 + 7.4 * bb))); // BAD_CONSTANTS
  } else {
    ustarb = cvarb * va / (1. + 4.7 * ribn);
  }
  uustar = va * ustarb;
}

KOKKOS_INLINE_FUNCTION
void calculate_ustar(
    const int beglt, const int endlt, const int index_season[n_land_type],
    const bool fr_lnduse[n_land_type], const bool unstable,
    const Real zl,            // height of lowest level [m]
    const Real uustar,        // u*ustar (assumed constant over grid) [m^2/s^2]
    const Real ribn,          // richardson number [unitless]
    Real ustar[n_land_type],  // friction velocity [m/s]
    Real cvar[n_land_type],   // height parameter
    Real bycp[n_land_type]) { // buoyancy parameter for unstable conditions
  // FIXME: use seq_drydep_mod, only: z0

  //-------------------------------------------------------------------------------------
  // calculate the friction velocity for each land type u_i=uustar/u*_i
  //-------------------------------------------------------------------------------------
  for (int lt = beglt; lt < endlt; ++lt) {
    if (fr_lnduse[lt]) { // BAD_CONSTANTS
      // FIXME: z0 may need its indices switched?
      if (unstable) {
        cvar[lt] = karman / haero::log(zl / z0[index_season[lt]][lt]);
        bycp[lt]  = 9.4 * haero::square(cvar[lt] * haero::sqrt(haero::abs(ribn)*zl/z0[index_season[lt]][lt]]);
        ustar[lt] = haero::sqrt(cvar[lt]*uustar*haero::sqrt(1.0 - (9.4*ribn/(1.0 + 7.4*bycp[lt]))));
      } else {
        cvar[lt] = karman / haero::log(zl / z0[index_season[lt]][lt]);
        ustar[lt] = haero::sqrt(cvar[lt] * uustar / (1.0 + 4.7 * ribn));
      }
    }
  }
}

KOKKOS_INLINE_FUNCTION
void calculate_ustar_over_water(
    const int beglt, const int endlt, const int index_season[n_land_type],
    const bool fr_lnduse[n_land_type], const bool unstable,
    const Real zl,            // height of lowest level [m]
    const Real uustar,        // u*ustar (assumed constant over grid) [m^2/s^2]
    const Real ribn,          // richardson number [-]
    Real ustar[n_land_type],  // friction velocity [m/s]
    Real cvar[n_land_type],   // height parameter
    Real bycp[n_land_type]) { // buoyancy parameter for unstable conditions
  constexpr Real diffk = 1.461e-5; // BAD_CONSTANT

  //-------------------------------------------------------------------------------------
  // revise calculation of friction velocity and z0 over water
  //-------------------------------------------------------------------------------------
  int lt = lt_for_water;
  if (fr_lnduse[lt]) {
    // BAD_CONSTANTS
    Real z0water =
        0.016 * haero::square(ustar[lt]) / grav + diffk / (9.1 * ustar[lt]);
    cvar[lt] = karman / haero::log(zl / z0water);
    if (unstable) {
      bycp[lt] = 9.4 * haero::square(cvar[lt]) *
                 haero::sqrt(haero::abs(ribn) * zl / z0water);
      ustar[lt] =
          haero::sqrt(cvar[lt] * uustar *
                      haero::sqrt(1.0 - (9.4 * ribn / (1.0 + 7.4 * bycp[lt]))));
    } else {
      ustar[lt] = haero::sqrt(cvar[lt] * uustar / (1.0 + 4.7 * ribn));
    }
  }
}

//-------------------------------------------------------------------------------------
// compute monin-obukhov length for unstable and stable conditions/ sublayer
// resistance
//-------------------------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION
void calculate_obukhov_length(
    const int beglt, const int endlt, const bool fr_lnduse[n_land_type],
    const bool unstable,
    const Real tha, // atmospheric virtual potential temperature [K]
    const Real thg, // ground virtual potential temperature [K]
    const Real ustar[n_land_type], // friction velocity [m/s]
    const Real cvar[n_land_type],  // height parameter
    const Real va,                 // magnitude of v on cross points [m/s]
    const Real bycp[n_land_type],  // buoyancy parameter for unstable conditions
    const Real ribn,               // richardson number [unitless]
    Real obklen[n_land_type]) {    // monin-obukhov length [m]
  for (int lt = beglt; lt < endlt; ++lt) {
    if (fr_lnduse[lt]) {
      // BAD_CONSTANTS
      Real hvar = (va / 0.74) * (tha - thg) * haero::square(cvar[lt]);
      Real htmp;
      if (unstable) {
        htmp = hvar * (1.0 - (9.4 * ribn / (1.0 + 5.3 * bycp[lt])));
      } else {
        htmp = hvar / haero::square((1.0 + 4.7 * ribn));
      }
      obklen[lt] = thg * ustar[lt] * ustar[lt] / (karman * grav * htmp);
    }
  }
}

KOKKOS_INLINE_FUNCTION
void calculate_aerodynamic_and_quasilaminar_resistance(
    const int beglt, const int endlt, const bool fr_lnduse[n_land_type],
    const Real zl,                  // height of lowest level [m]
    const Real obklen[n_land_type], // monin-obukhov length [m]
    const Real ustar[n_land_type],  // friction velocity [m/s]
    const Real cvar[n_land_type],   // height parameter
    Real dep_ra[n_land_type],       // aerodynamic resistance [s/m]
    Real dep_rb[n_land_type]) {     // sublayer resistance [s/m]
  for (int lt = beglt; lt < endlt; ++lt) {
    if (fr_lnduse[lt]) {
      Real psih; // stability correction factor [-]
      // BAD_CONSTANTS
      if (obklen[lt] < 0.0) {
        Real zeta = zl / obklen[lt];
        zeta = haero::max(-1.0, zeta);
        psih = haero::exp(0.598 + 0.39 * haero::log(-zeta) -
                          0.09 * haero::square(haero::log(-zeta)));
      } else {
        Real zeta = zl / obklen[lt];
        zeta = haero::min(1.0, zeta);
        psih = -5.0 * zeta;
      }
      // FIXME: crb is a mo_drpdep module variable initialized from diffm, difft
      // params
      // FIXME: it's only used here, so we hardwire it here
      constexpr Real diffm = 1.789e-5;
      constexpr Real difft = 2.060e-5;
      constexpr Real crb = haero::pow(difft / diffm, 2.0 / 3.0);
      dep_ra[lt] = (karman - psih * cvar[lt]) / (ustar[lt] * karman * cvar[lt]);
      dep_rb[lt] = (2.0 / (karman * ustar[lt])) * crb;
    }
  }
}

KOKKOS_INLINE_FUNCTION
void calculate_resistance_rgsx_and_rsmx(
    const int beglt, const int endlt, const int index_season[n_land_type],
    const bool fr_lnduse[n_land_type], const bool has_rain, const bool has_dew,
    const Real tc,            // temperature [C]
    const Real heff[nddvels], // Henry's Law coefficients
    const Real crs,           // multiplier to calculate rs
    Real &cts,                // correction to rlu rcl and rgs for frost
    Real rgsx[gas_pcnst][n_land_type], // ground resistance [s/m]
    Real rsmx[gas_pcnst]
             [n_land_type]) // vegetative resistance (plant mesophyll) [s/m]
{
  // FIXME: use seq_drydep_mod, only: ri, rgso, rgss, foxd, drat
  for (int ispec = 0; ispec < gas_pcnst; ++ispec) {
    if (has_dvel[ispec]) {
      int idx_drydep = map_dvel[ispec];
      for (int lt = beglt; lt < endlt; ++lt) {
        if (fr_lnduse[lt]) {
          Real rmx;
          int sndx = index_season[lt];
          if (ispec == so2_ndx) {
            rmx = 0.0;
          } else { // BAD_CONSTANTS
            rmx = 1.0 / (heff[idx_drydep] / 3000.0 + 100.0 * foxd[idx_drydep]);
          }
          cts = 1000.0 * haero::exp(-tc - 4.0); // correction for frost
          // FIXME: rgss, rgso indices probably need to be swapped
          rgsx[ispec][lt] =
              cts + 1.0 / ((heff[idx_drydep] / (1e5 * rgss[sndx][lt])) +
                           (foxd[idx_drydep] / rgso[sndx][lt]));

          if (lt == lt_for_water) {
            rsmx[ispec][lt] = large_value;
          } else {
            // FIXME: ri indices probably need to be swapped
            Real rs = ri[sndx][lt] * crs;
            if (has_dew || has_rain) {
              dewm = 3.0;
            } else {
              dewm = 1.0;
            }
            rsmx[ispec][lt] = (dewm * rs * drat[idx_drydep] + rmx);
          }
        }
      }
    }
  }
}

KOKKOS_INLINE_FUNCTION
void calculate_resistance_rclx(
    const int beglt, const int endlt, const int index_season[n_land_type],
    const bool fr_lnduse[n_land_type],
    const Real heff[nddvels], // Henry's law coefficients
    const Real cts,           // correction to rlu rcl and rgs for frost
    Real rclx[gas_pcnst][n_land_type]) { // lower canopy resistance [s/m]
  // FIXME: use seq_drydep_mod, only: rclo, rcls, foxd
  for (int ispec = 0; ispec < gas_pcnst; ++ispec) {
    if (has_dvel[ispec]) {
      idx_drydep = map_dvel[ispec];
      for (int lt = beglt; lt < endlt; ++lt) {
        if (fr_lnduse[lt]) {
          if (lt == lt_for_water) {
            rclx[ispec][lt] = large_value;
          } else {
            int sndx = index_season[lt];
            // BAD_CONSTANT
            // FIXME: rcls and rclo probably need indices swapped
            rclx[ispec][lt] =
                cts + 1.0 / ((heff[idx_drydep] / (1e5 * rcls[sndx][lt])) +
                             (foxd[idx_drydep] / rclo[sndx][lt]));
          }
        }
      }
    }
  }

  for (int ispec = 0; ispec < gas_pcnst; ++ispec) {
    if (has_dvel[ispec] && (ispec == so2_ndx)) {
      for (int lt = beglt; lt < endlt; ++lt) {
        if (lt != lt_for_water) {
          if (fr_lnduse[lt]) {
            // FIXME: rcls probably needs its indices swapped
            rclx[ispec][lt] = cts + rcls[index_season[lt]][lt];
          }
        }
      }
    }
  }
}

KOKKOS_INLINE_FUNCTION
void calculate_resistance_rlux(
    const int beglt, const int endlt, const int index_season[n_land_type],
    const bool fr_lnduse[n_land_type], const bool has_rain, const bool has_dew,
    const Real sfc_temp,      // surface temperature [K]
    const Real qs,            // saturation specific humidity [kg/kg]
    const Real spec_hum,      // specific humidity [kg/kg]
    const Real heff[nddvels], // Henry's Law coefficients
    const Real cts,           // correction to rlu rcl and rgs for frost
    Real rlux[gas_pcnst][n_land_type]) { // lower canopy resistance [s/m] ! out
  // FIXME: use seq_drydep_mod, only: rclo, rcls, rlu, foxd

  integer ::icol, lt, ispec, idx_drydep,
      sndx Real rlux_o3[n_land_type] =
          {}; // vegetative resistance (upper canopy) [s/m]
  for (int ispec = 0; ispec < gas_pcnst; ++ispec) {
    if (has_dvel[ispec]) {
      int idx_drydep = map_dvel[ispec];
      for (int lt = beglt; lt < endlt; ++lt) {
        if (fr_lnduse[lt]) {
          if (lt == lt_for_water) {
            rlux[ispec][lt] = large_value;
          } else { // BAD_CONSTANT
            int sndx = index_season[lt];
            // FIXME: rlu probably needs its indices swapped
            rlux[ispec][lt] = cts + rlu[sndx][lt] / (1e-5 * heff[idx_drydep] +
                                                     foxd[idx_drydep]);
          }
        }
      }
    }
  }

  for (int lt = beglt; lt < endlt; ++lt) {
    if (lt != lt_for_water) {
      if (fr_lnduse[lt]) {
        int sndx = index_season[lt];
        //-------------------------------------------------------------------------------------
        //       ... no effect if sfc_temp < O C
        //-------------------------------------------------------------------------------------
        if (sfc_temp > tmelt) {
          // BAD_CONSTANTS
          // FIXME: rlu probably needs its indices swapped
          if (has_dew) {
            rlux_o3[lt] = 3000.0 * rlu[sndx][lt] / (1000.0 + rlu[sndx][lt]);
          }
          if (has_rain) {
            rlux_o3[lt] =
                3000.0 * rlu[sndx][lt] / (1000.0 + 3.0 * rlu[sndx][lt]);
          }
        }
      }
    }
  }

  for (int ispec = 0; ispec < gas_pcnst; ++ispec) {
    int idx_drydep = map_dvel[ispec];
    if (has_dvel[ispec] && (ispec != so2_ndx)) {
      for (int lt = beglt; lt < endlt; ++lt) {
        if (lt != lt_for_water) {
          if (fr_lnduse[lt] && (sfc_temp > tmelt) && has_dew) {
            //-------------------------------------------------------------------------------------
            // no effect if sfc_temp < O C
            //-------------------------------------------------------------------------------------
            // BAD_CONSTANTS
            rlux[ispec][lt] = 1.0 / ((1.0 / (3. * rlux[ispec][lt])) +
                                     1e-7 * heff[idx_drydep] +
                                     foxd[idx_drydep] / rlux_o3[lt]);
          }
        }
      }
    } else if (ispec == so2_ndx) {
      for (int lt = beglt; lt < endlt; ++lt) {
        if (lt != lt_for_water) {
          if (fr_lnduse[lt]) {
            //-------------------------------------------------------------------------------------
            // no effect if sfc_temp < O C
            //-------------------------------------------------------------------------------------
            if (sfc_temp > tmelt) {
              // BAD_CONSTANTS vvvvv
              if (qs <= spec_hum) {
                rlux[ispec][lt] = 100.0;
              }
              if (has_rain) {
                // FIXME: rlu probably needs its indices swapped
                rlux[ispec][lt] = 15.0 * rlu[index_season[lt]][lt] /
                                  (5.0 + 3e-3 * rlu[index_season[lt]][lt]);
              }
            }
            rlux[ispec][lt] += cts;
          }
        }
      }
      if (fr_lnduse[0] && (has_dew || has_rain)) {
        rlux[ispec][0] = 50.0; // BAD_CONSTANT
      }
    }
  }
}

KOKKOS_INLINE_FUNCTION
void calculate_gas_drydep_vlc_and_flux(
    const int beglt, const int endlt, // land type index endpoints
    const int index_season[n_land_type], const bool fr_lnduse[n_land_type],
    const Real lcl_frc_landuse[n_land_type],
    const Real mmr[gas_pcnst],      // constituent mmrs at surface [kg/kg]
    const Real dep_ra[n_land_type], // aerodynamic resistance [s/m]
    const Real dep_rb[n_land_type], // sublayer resistance [s/m]
    const Real term,
    const Real rsmx[gas_pcnst][n_land_type], // vegetative resistance (plant
                                             // mesophyll) [s/m]
    const Real rlux[gas_pcnst][n_land_type], // vegetative resistance (upper
                                             // canopy) [s/m]
    const Real rclx[gas_pcnst][n_land_type], // lower canopy resistance [s/m]
    const Real rgsx[gas_pcnst][n_land_type], // ground resistance [s/m]
    const Real rdc,         // part of lower canopy resistance [s/m]
    Real dvel[gas_pcnst],   // deposition velocity [cm/s]
    Real dflx[gas_pcnst]) { // deposition flux [1/cm^2/s]
  constexpr Real m_to_cm_per_s = 100.0;

  // FIXME: use seq_drydep_mod, only: rac
  const Real rac[n_land_type][???];

  for (int ispec = 0; ispec < gas_pcnst; ++ispec) {
    if (has_dvel[ispec]) {
      Real wrk = 0.0;
      Real resc, lnd_frc;
      for (int lt = beglt; lt < endlt; ++lt) {
        if (fr_lnduse[lt]) {
          resc = 1.0 / (1.0 / rsmx[ispec][lt] + 1.0 / rlux[ispec][lt]) +
                 1.0 / (rdc + rclx[ispec][lt]) +
                 1.0 / (rac[lt][index_season[lt]] + rgsx[ispec][lt]);
          resc = haero::max(10.0, resc);
          lnd_frc = lcl_frc_landuse[lt];
        }
      }

      //-------------------------------------------------------------------------------------
      //  ... compute average deposition velocity
      //-------------------------------------------------------------------------------------
      if (ispec == so2_ndx) {
        if (lt == lt_for_water) {
          if (fr_lnduse[lt]) {
            // assume no surface resistance for SO2 over water
            wrk += lnd_frc / (dep_ra[lt] + dep_rb[lt]);
          }
        } else {
          if (fr_lnduse[lt]) {
            wrk += lnd_frc / (dep_ra[lt] + dep_rb[lt] + resc);
          }
        }
      } else {
        if (fr_lnduse[lt]) {
          wrk += lnd_frc / (dep_ra[lt] + dep_rb[lt] + resc);
        }
      }

      dvel[ispec] = wrk * m_to_cm_per_s;
      dflx[ispec] = term * dvel[ispec] * mmr[ispec];
    }
  }
}

KOKKOS_INLINE_FUNCTION
Real get_potential_temperature(const Real temperature,         // [K]
                               const Real pressure,            // [Pa]
                               const Real specific_humidity) { // [kg/kg]
  constexpr Real p00 = 100000.0;
  constexpr Real cp = 1004.0;
  constexpr Real rovcp = rair / cp;

  // BAD_CONSTANT
  return temperature * haero::pow(p00 / pressure, rovcp) *
         (1.0 + 0.61 * specific_humidity);
}

KOKKOS_INLINE_FUNCTION
Real get_saturation_specific_humidity(const Real temperature, // [K]
                                      const Real pressure) {  // [Pa]
  // saturation vapor pressure [Pa] (BAD_CONSTANTS)
  Real es = 611.0 *
            haero::exp(5414.77 * (temperature - tmelt) / (tmelt * temperature));
  // saturation specific humidity [kg/kg] (BAD_CONSTANTS)
  Real ws = 0.622 * es / (pressure - es);

  return ws / (1. + ws);
}

KOKKOS_INLINE_FUNCTION
void drydep_xactive(
    const int ncdate,               // date [YYMMDD]
    const int col_index_season[12], // column-specific mapping of month indices
                                    // to seasonal land-type indices [-]
    const Real sfc_temp,            // surface temperature [K]
    const Real air_temp,            // surface air temperature [K]
    const Real tv,                  // potential temperature [K]
    const Real pressure_sfc,        // surface pressure [Pa]
    const Real pressure_10m,        // 10-meter pressure [Pa]
    const Real spec_hum,            // specific humidity [kg/kg]
    const Real wind_speed,          // 10-meter wind spped [m/s]
    const Real rain,                // rain content [??]
    const Real snow,                // snow height [m]
    const Real solar_flux,     // direct shortwave surface radiation [W/m^2]
    const Real mmr[gas_pcnst], // constituent MMRs [kg/kg]
    Real dvel[gas_pcnst],      // deposition velocity [1/cm/s]
    Real dflx[gas_pcnst]) {    // deposition flux [1/cm^2/s]
  // BAD_CONSTANTS
  constexpr Real rain_threshold = 1e-7;   // of the order of 1cm/day [m/s]
  constexpr Real temp_highbound = 313.15; // [K]
  constexpr Real ric = 0.2;               // ???

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
  if (snow > 0.01) { // BAD_CONSTANT
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
  Real tha = get_potential_temperature(air_temp, pressure_10m, spec_hum);
  l Real thg = get_potential_temperature(sfc_temp, pressure_sfc, spec_hum);

  //-------------------------------------------------------------------------------------
  // height of 1st level
  //-------------------------------------------------------------------------------------
  // BAD_CONSTANTS
  Real zl = -rair / grav * air_temp * (1. + 0.61 * spec_hum) *
            log(pressure_10m / pressure_sfc);

  //-------------------------------------------------------------------------------------
  // wind speed
  //-------------------------------------------------------------------------------------
  // BAD_CONSTANT
  Real va = haero::max(0.01, wind_speed);

  //-------------------------------------------------------------------------------------
  // Richardson number
  //-------------------------------------------------------------------------------------
  Real ribn = zl * grav * (tha - thg) / thg / (va * va);
  ribn = haero::min(ribn, ric);

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
    crs = (1.0 + haero::square(200.0 / (solar_flux + 0.1))) *
          (400.0 / (tc * (40.0 - tc)));
  } else {
    crs = large_value;
  }

  //-------------------------------------------------------------------------------------
  // rdc (lower canopy res)
  //-------------------------------------------------------------------------------------
  Real rdc = 100.0 * (1.0 + 1000.0 / (solar_flux + 10.0)); // BAD_CONSTANTS

  //-------------------------------------------------------------------------------------
  // 	... form working arrays
  //-------------------------------------------------------------------------------------
  for (int lt = 0; lt < n_land_type; ++lt) {
    lcl_frc_landuse[lt] = fraction_landuse(icol, lt, lchnk); // FIXME: ???
    fr_lnduse[lt] = (lcl_frc_landuse[lt] > 0.0);
  }

  calculate_uustar(index_season, fr_lnduse, unstable, lcl_frc_landuse, va, zl,
                   ribn, uustar);

  calculate_ustar(0, n_land_type, index_season, fr_lnduse, unstable, zl, uustar,
                  ribn, ustar, cvar, bycp);

  calculate_ustar_over_water(0, n_land_type, index_season, fr_lnduse, unstable,
                             zl, uustar, ribn, ustar, cvar, bycp);

  calculate_obukhov_length(0, n_land_type, fr_lnduse, unstable, tha, thg, ustar,
                           cvar, va, bycp, ribn, obklen);

  calculate_aerodynamic_and_quasilaminar_resistance(
      0, n_land_type, fr_lnduse, zl, obklen, ustar, cvar, dep_ra, dep_rb);

  //-------------------------------------------------------------------------------------
  // surface resistance : depends on both land type and species
  // land types are computed seperately, then resistance is computed as
  // average of values following wesely rc=(1/(rs+rm) + 1/rlu +1/(rdc+rcl) +
  // 1/(rac+rgs))**-1
  //
  // compute rsmx = 1/(rs+rm) : multiply by 3 if surface is wet
  //-------------------------------------------------------------------------------------
  calculate_resistance_rgsx_and_rsmx(0, n_land_type, index_season, fr_lnduse,
                                     has_rain, has_dew, tc, heff, crs, cts,
                                     rgsx, rsmx);

  calculate_resistance_rclx(0, n_land_type, index_season, fr_lnduse, heff, cts,
                            rclx);

  calculate_resistance_rlux(0, n_land_type, index_season, fr_lnduse, has_rain,
                            has_dew, sfc_temp, qs, spec_hum, heff, cts, rlux);

  Real term = 1e-2 * pressure_10m / (rair * tv); // BAD_CONSTANT
  calculate_gas_drydep_vlc_and_flux(0, n_land_type, index_season, fr_lnduse,
                                    lcl_frc_landuse, mmr, dep_ra, dep_rb, term,
                                    rsmx, rlux, rclx, rgsx, rdc, dvel, dflx);
}

} // namespace mam4::mo_drydep

#endif
