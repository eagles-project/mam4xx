// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MAM4XX_MO_DRYDEP_HPP
#define MAM4XX_MO_DRYDEP_HPP

#include <haero/math.hpp>
#include <mam4xx/aero_config.hpp>
#include <mam4xx/gas_chem_mechanism.hpp>
#include <mam4xx/seq_drydep.hpp>
#include <mam4xx/utils.hpp>

namespace mam4::mo_drydep {

constexpr int gas_pcnst = mam4::gas_chemistry::gas_pcnst;
constexpr int n_land_type = mam4::seq_drydep::NLUse;
constexpr int lt_for_water = 6; // from eam/src/chemistry/mozart/mo_drydep.F90
constexpr int NSeas = mam4::seq_drydep::NSeas;

// BAD_CONSTANTS
constexpr Real large_value = 1e36;
constexpr Real rair = 287.04;
constexpr Real grav = 9.81;
constexpr Real karman = 0.4;   // from shr_const_mod.F90
constexpr Real tmelt = 273.15; // from shr_const_mod.F90 via physconst.F90

// nddvels is used only to determine array sizes, and is not involved in
// logic below. Because we rely on its constancy for C++ array sizes, we
// must use its maximum value, which is maxspc above
constexpr int nddvels = mam4::seq_drydep::maxspc;

KOKKOS_INLINE_FUNCTION
void calculate_uustar(
    const seq_drydep::Data &drydep_data, const int index_season[n_land_type],
    const bool fr_lnduse[n_land_type], const bool unstable,
    const Real lcl_frc_landuse[n_land_type],
    const Real va,   // magnitude of v on cross points
    const Real zl,   // height of lowest level
    const Real ribn, // richardson number [-]
    Real &uustar) {  // u * ustar (assumed constant over grid) [m^2/s^2]
  const auto z0 = drydep_data.z0;

  //-------------------------------------------------------------------------------------
  // find grid averaged z0: z0bar (the roughness length)
  // z_o=exp[S(f_i*ln(z_oi))]
  // this is calculated so as to find u_i, assuming u*u=u_i*u_i
  //-------------------------------------------------------------------------------------
  Real z0b = 0.0; // average roughness length over grid
  for (int lt = 0; lt < n_land_type; ++lt) {
    if (fr_lnduse[lt]) {
      z0b += lcl_frc_landuse[lt] * log(z0(index_season[lt], lt));
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
    Real bb =
        9.4 * haero::square(cvarb) * haero::sqrt(haero::abs(ribn) * zl / z0b);
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
    const seq_drydep::Data &drydep_data, const int beglt, const int endlt,
    const int index_season[n_land_type], const bool fr_lnduse[n_land_type],
    const bool unstable,
    const Real zl,            // height of lowest level [m]
    const Real uustar,        // u*ustar (assumed constant over grid) [m^2/s^2]
    const Real ribn,          // richardson number [unitless]
    Real ustar[n_land_type],  // friction velocity [m/s]
    Real cvar[n_land_type],   // height parameter
    Real bycp[n_land_type]) { // buoyancy parameter for unstable conditions
  const auto z0 = drydep_data.z0;

  //-------------------------------------------------------------------------------------
  // calculate the friction velocity for each land type u_i=uustar/u*_i
  //-------------------------------------------------------------------------------------
  for (int lt = beglt; lt <= endlt; ++lt) {
    if (fr_lnduse[lt]) { // BAD_CONSTANTS
      cvar[lt] = karman / haero::log(zl / z0(index_season[lt], lt));
      if (unstable) {
        bycp[lt] =
            9.4 * haero::square(cvar[lt]) *
            haero::sqrt(haero::abs(ribn) * zl / z0(index_season[lt], lt));
        ustar[lt] = haero::sqrt(
            cvar[lt] * uustar *
            haero::sqrt(1.0 - (9.4 * ribn / (1.0 + 7.4 * bycp[lt]))));
      } else {
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
  for (int lt = beglt; lt <= endlt; ++lt) {
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
  for (int lt = beglt; lt <= endlt; ++lt) {
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
      // NOTE: crb is a mo_drydep module variable initialized from the
      // NOTE: diffm, difft params. It's only used here, so we hardwire it
      constexpr Real diffm = 1.789e-5;
      constexpr Real difft = 2.060e-5;
      const Real crb = haero::pow(difft / diffm, 2.0 / 3.0);
      dep_ra[lt] = (karman - psih * cvar[lt]) / (ustar[lt] * karman * cvar[lt]);
      dep_rb[lt] = (2.0 / (karman * ustar[lt])) * crb;
    }
  }
}

KOKKOS_INLINE_FUNCTION
void calculate_resistance_rgsx_and_rsmx(
    const seq_drydep::Data &drydep_data, const int beglt, const int endlt,
    const int index_season[n_land_type], const bool fr_lnduse[n_land_type],
    const bool has_rain, const bool has_dew,
    const Real tc,            // temperature [C]
    const Real heff[nddvels], // Henry's Law coefficients
    const Real crs,           // multiplier to calculate rs
    Real &cts,                // correction to rlu rcl and rgs for frost
    Real rgsx[gas_pcnst][n_land_type], // ground resistance [s/m]
    Real rsmx[gas_pcnst]
             [n_land_type]) // vegetative resistance (plant mesophyll) [s/m]
{
  const auto ri = drydep_data.ri;
  const auto rgso = drydep_data.rgso;
  const auto rgss = drydep_data.rgss;
  const auto foxd = drydep_data.foxd;
  const auto drat = drydep_data.drat;

  for (int ispec = 0; ispec < gas_pcnst; ++ispec) {
    if (drydep_data.has_dvel[ispec]) {
      int idx_drydep = drydep_data.map_dvel(ispec);
      for (int lt = beglt; lt <= endlt; ++lt) {
        if (fr_lnduse[lt]) {
          Real rmx;
          int sndx = index_season[lt];
          if (ispec == drydep_data.so2_ndx) {
            rmx = 0.0;
          } else { // BAD_CONSTANTS
            rmx = 1.0 / (heff[idx_drydep] / 3000.0 + 100.0 * foxd(idx_drydep));
          }
          cts = 1000.0 * haero::exp(-tc - 4.0); // correction for frost
          rgsx[ispec][lt] =
              cts + 1.0 / ((heff[idx_drydep] / (1e5 * rgss(sndx, lt))) +
                           (foxd[idx_drydep] / rgso(sndx, lt)));

          if (lt == lt_for_water) {
            rsmx[ispec][lt] = large_value;
          } else {
            Real rs = ri(sndx, lt) * crs;
            Real dewm;
            if (has_dew || has_rain) {
              dewm = 3.0;
            } else {
              dewm = 1.0;
            }
            rsmx[ispec][lt] = (dewm * rs * drat(idx_drydep) + rmx);
          }
        }
      }
    }
  }
}

KOKKOS_INLINE_FUNCTION
void calculate_resistance_rclx(
    const seq_drydep::Data &drydep_data, const int beglt, const int endlt,
    const int index_season[n_land_type], const bool fr_lnduse[n_land_type],
    const Real heff[nddvels], // Henry's law coefficients
    const Real cts,           // correction to rlu rcl and rgs for frost
    Real rclx[gas_pcnst][n_land_type]) { // lower canopy resistance [s/m]

  const auto rclo = drydep_data.rclo;
  const auto rcls = drydep_data.rcls;
  const auto foxd = drydep_data.foxd;

  for (int ispec = 0; ispec < gas_pcnst; ++ispec) {
    if (drydep_data.has_dvel(ispec)) {
      int idx_drydep = drydep_data.map_dvel(ispec);
      for (int lt = beglt; lt <= endlt; ++lt) {
        if (fr_lnduse[lt]) {
          if (lt == lt_for_water) {
            rclx[ispec][lt] = large_value;
          } else {
            int sndx = index_season[lt];
            // BAD_CONSTANT
            rclx[ispec][lt] =
                cts + 1.0 / ((heff[idx_drydep] / (1e5 * rcls(sndx, lt))) +
                             foxd(idx_drydep) / rclo(sndx, lt));
          }
        }
      }
    }
  }

  for (int ispec = 0; ispec < gas_pcnst; ++ispec) {
    if (drydep_data.has_dvel(ispec) && (ispec == drydep_data.so2_ndx)) {
      for (int lt = beglt; lt <= endlt; ++lt) {
        if (lt != lt_for_water) {
          if (fr_lnduse[lt]) {
            rclx[ispec][lt] = cts + rcls(index_season[lt], lt);
          }
        }
      }
    }
  }
}

KOKKOS_INLINE_FUNCTION
void calculate_resistance_rlux(
    const seq_drydep::Data &drydep_data, const int beglt, const int endlt,
    const int index_season[n_land_type], const bool fr_lnduse[n_land_type],
    const bool has_rain, const bool has_dew,
    const Real sfc_temp,      // surface temperature [K]
    const Real qs,            // saturation specific humidity [kg/kg]
    const Real spec_hum,      // specific humidity [kg/kg]
    const Real heff[nddvels], // Henry's Law coefficients
    const Real cts,           // correction to rlu rcl and rgs for frost
    Real rlux[gas_pcnst][n_land_type]) // lower canopy resistance [s/m] ! out
{
  const auto rlu = drydep_data.rlu;
  const auto foxd = drydep_data.foxd;

  Real rlux_o3[n_land_type] = {}; // vegetative resistance (upper canopy) [s/m]
  for (int ispec = 0; ispec < gas_pcnst; ++ispec) {
    if (drydep_data.has_dvel(ispec)) {
      int idx_drydep = drydep_data.map_dvel(ispec);
      for (int lt = beglt; lt <= endlt; ++lt) {
        if (fr_lnduse[lt]) {
          if (lt == lt_for_water) {
            rlux[ispec][lt] = large_value;
          } else { // BAD_CONSTANT
            int sndx = index_season[lt];
            rlux[ispec][lt] = cts + rlu(sndx, lt) / (1e-5 * heff[idx_drydep] +
                                                     foxd(idx_drydep));
          }
        }
      }
    }
  }

  for (int lt = beglt; lt <= endlt; ++lt) {
    if (lt != lt_for_water) {
      if (fr_lnduse[lt]) {
        int sndx = index_season[lt];
        //-------------------------------------------------------------------------------------
        //       ... no effect if sfc_temp < O C
        //-------------------------------------------------------------------------------------
        if (sfc_temp > tmelt) {
          // BAD_CONSTANTS
          if (has_dew) {
            rlux_o3[lt] = 3000.0 * rlu(sndx, lt) / (1000.0 + rlu(sndx, lt));
          }
          if (has_rain) {
            rlux_o3[lt] =
                3000.0 * rlu(sndx, lt) / (1000.0 + 3.0 * rlu(sndx, lt));
          }
        }
      }
    }
  }

  for (int ispec = 0; ispec < gas_pcnst; ++ispec) {
    int idx_drydep = drydep_data.map_dvel(ispec);
    if (drydep_data.has_dvel(ispec) && (ispec != drydep_data.so2_ndx)) {
      for (int lt = beglt; lt <= endlt; ++lt) {
        if (lt != lt_for_water) {
          if (fr_lnduse[lt] && (sfc_temp > tmelt) && has_dew) {
            //-------------------------------------------------------------------------------------
            // no effect if sfc_temp < O C
            //-------------------------------------------------------------------------------------
            // BAD_CONSTANTS
            rlux[ispec][lt] = 1.0 / ((1.0 / (3. * rlux[ispec][lt])) +
                                     1e-7 * heff[idx_drydep] +
                                     foxd(idx_drydep) / rlux_o3[lt]);
          }
        }
      }
    } else if (ispec == drydep_data.so2_ndx) {
      for (int lt = beglt; lt <= endlt; ++lt) {
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
                rlux[ispec][lt] = 15.0 * rlu(index_season[lt], lt) /
                                  (5.0 + 3e-3 * rlu(index_season[lt], lt));
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
    const seq_drydep::Data &drydep_data, const int beglt, const int endlt,
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
  const auto rac = drydep_data.rac;

  for (int ispec = 0; ispec < gas_pcnst; ++ispec) {
    if (drydep_data.has_dvel(ispec)) {
      Real wrk = 0.0;
      Real resc, lnd_frc;
      for (int lt = beglt; lt <= endlt; ++lt) {
        if (fr_lnduse[lt]) {
          resc = 1.0 / (1.0 / rsmx[ispec][lt] + 1.0 / rlux[ispec][lt] +
                        1.0 / (rdc + rclx[ispec][lt]) +
                        1.0 / (rac(index_season[lt], lt) + rgsx[ispec][lt]));
          resc = haero::max(10.0, resc);
          lnd_frc = lcl_frc_landuse[lt];
        }

        //-------------------------------------------------------------------------------------
        //  ... compute average deposition velocity
        //-------------------------------------------------------------------------------------
        if (ispec == drydep_data.so2_ndx) {
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
    const seq_drydep::Data &drydep_data,
    const Real fraction_landuse[n_land_type], // fraction of land use for column
                                              // by land type
    const int ncdate,                         // date [YYMMDD]
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

  for (int ispec = 0; ispec < gas_pcnst; ++ispec) {
    dvel[ispec] = 0.0;
    dflx[ispec] = 0.0;
  }

  //-------------------------------------------------------------------------------------
  // define species-dependent parameters (temperature dependent)
  //-------------------------------------------------------------------------------------
  Real heff[nddvels];
  mam4::seq_drydep::setHCoeff(sfc_temp, heff);

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
  Real thg = get_potential_temperature(sfc_temp, pressure_sfc, spec_hum);

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
  bool fr_lnduse[n_land_type] = {};
  Real lcl_frc_landuse[n_land_type];
  for (int lt = 0; lt < n_land_type; ++lt) {
    lcl_frc_landuse[lt] = fraction_landuse[lt];
    fr_lnduse[lt] = (lcl_frc_landuse[lt] > 0.0);
  }

  Real uustar;
  calculate_uustar(drydep_data, index_season, fr_lnduse, unstable,
                   lcl_frc_landuse, va, zl, ribn, uustar);

  Real ustar[n_land_type], cvar[n_land_type], bycp[n_land_type];
  calculate_ustar(drydep_data, 0, n_land_type - 1, index_season, fr_lnduse,
                  unstable, zl, uustar, ribn, ustar, cvar, bycp);

  calculate_ustar_over_water(0, n_land_type - 1, index_season, fr_lnduse,
                             unstable, zl, uustar, ribn, ustar, cvar, bycp);

  Real obklen[n_land_type];
  calculate_obukhov_length(0, n_land_type - 1, fr_lnduse, unstable, tha, thg,
                           ustar, cvar, va, bycp, ribn, obklen);

  Real dep_ra[n_land_type], dep_rb[n_land_type];
  calculate_aerodynamic_and_quasilaminar_resistance(
      0, n_land_type - 1, fr_lnduse, zl, obklen, ustar, cvar, dep_ra, dep_rb);

  //-------------------------------------------------------------------------------------
  // surface resistance : depends on both land type and species
  // land types are computed seperately, then resistance is computed as
  // average of values following wesely rc=(1/(rs+rm) + 1/rlu +1/(rdc+rcl) +
  // 1/(rac+rgs))**-1
  //
  // compute rsmx = 1/(rs+rm) : multiply by 3 if surface is wet
  //-------------------------------------------------------------------------------------
  Real cts, rgsx[gas_pcnst][n_land_type], rsmx[gas_pcnst][n_land_type];
  calculate_resistance_rgsx_and_rsmx(drydep_data, 0, n_land_type - 1,
                                     index_season, fr_lnduse, has_rain, has_dew,
                                     tc, heff, crs, cts, rgsx, rsmx);

  Real rclx[gas_pcnst][n_land_type];
  calculate_resistance_rclx(drydep_data, 0, n_land_type - 1, index_season,
                            fr_lnduse, heff, cts, rclx);

  Real rlux[gas_pcnst][n_land_type];
  calculate_resistance_rlux(drydep_data, 0, n_land_type - 1, index_season,
                            fr_lnduse, has_rain, has_dew, sfc_temp, qs,
                            spec_hum, heff, cts, rlux);

  Real term = 1e-2 * pressure_10m / (rair * tv); // BAD_CONSTANT
  calculate_gas_drydep_vlc_and_flux(
      drydep_data, 0, n_land_type - 1, index_season, fr_lnduse, lcl_frc_landuse,
      mmr, dep_ra, dep_rb, term, rsmx, rlux, rclx, rgsx, rdc, dvel, dflx);
}

} // namespace mam4::mo_drydep

#endif
