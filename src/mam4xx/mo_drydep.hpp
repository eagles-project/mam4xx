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

#include <ekat_subview_utils.hpp>

namespace mam4::mo_drydep {
using View1D = DeviceType::view_1d<Real>;
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
constexpr Real r2d = 180.0 / haero::Constants::pi; // radians to degrees
// nddvels is equal to number of species in dry deposition list for gases.
constexpr int nddvels = mam4::seq_drydep::n_drydep;

/**
 * Finds the season index for each longitude point based on the most frequent
 * season in the 11 vegetation classes to mitigate banding issues in dvel.
 * @brief Finds the season index based on the given latitude.
 *
 * @param[in] clat_j Latitude in degrees from the host model.
 * @param[in] lat_lai Latitude values from the NC file.
 * @param[in] nlat_lai Size of lat_lai.
 * @param[in] wk_lai Season_wes from the NC file.
 * @param[out] index_season_lai Outputs the season indices.
 */

// find_season_index_at_clat only needs to be executed one time and is small.
// Thus, execute it only on the host.
using View1DHost = DeviceType::view_1d<Real>::HostMirror;
using ConstView1DHost = DeviceType::view_1d<const Real>::HostMirror;
using View1DIntHost = DeviceType::view_1d<int>::HostMirror;
using View2DIntHost = DeviceType::view_2d<int>::HostMirror;
using View3DIntHost = DeviceType::view_3d<int>::HostMirror;

using KTH = ekat::KokkosTypes<ekat::HostDevice>;

inline void find_season_index_at_lat(const Real clat_j,
                                     const View1DHost &lat_lai,
                                     const int nlat_lai,
                                     const View3DIntHost &wk_lai,
                                     const View1DIntHost &index_season_lai) {

  // Comment from Fortran code.
  /*For unstructured grids plon is the 1d horizontal grid size and plat=1
  ! So this code averages at the latitude of each grid point - not an ideal
  solution*/
  // BAD CONSTANT
  Real diff_min = 10.0;
  int pos_min = -99;

  // NOTE: EAM uses degrees for clat_j, but EAMxx uses radians.
  // Because we will use this function in EAMxx, we will not perform this unit
  // conversion. const Real target_lat = clat_j * r2d;
  const Real target_lat = clat_j;

  for (int i = 0; i < nlat_lai; ++i) {
    Real current_diff = haero::abs(lat_lai(i) - target_lat);
    if (current_diff < diff_min) {
      diff_min = current_diff;
      pos_min = i;
    }
  } // i
  EKAT_KERNEL_ASSERT_MSG(pos_min > -1,
                         "Error in mo_drydep: dvel_inti: cannot find index.\n");
  /* specify the season as the most frequent in the 11 vegetation classes
 ! this was done to remove a banding problem in dvel (JFL Oct 04)*/
  // BAD CONSTANT
  for (int m = 0; m < 12; ++m) {
    int num_seas[5] = {0, 0, 0, 0, 0};
    for (int l = 0; l < 11; ++l) {
      for (int k = 0; k < 5; ++k) {
        if (wk_lai(pos_min, l, m) == k + 1) {
          num_seas[k]++;
          break; // Exit the innermost loop
        }
      }
    }

    int num_max = -1;
    int k_max = 0;
    for (int k = 0; k < 5; ++k) {
      if (num_seas[k] > num_max) {
        num_max = num_seas[k];
        k_max = k; //
      }
    }

    index_season_lai(m) = k_max; //
  }                              // m
} // findSeasonIndex

/**
 * Finds the season index for each longitude point based on the most frequent
 * season in the 11 vegetation classes to mitigate banding issues in dvel.
 * @brief Finds the season index based on all latitudes.
 *
 * @param[in] clat Latitude in degrees from the host model.
 * @param[in] lat_lai Latitude in degrees values from the NC file.
 * @param[in] nlat_lai Size of lat_lai.
 * @param[in] wk_lai Season_wes from the NC file.
 * @param[out] index_season_lai Outputs the season indices.
 */
inline void find_season_index(const ConstView1DHost clat,
                              const View1DHost &lat_lai, const int nlat_lai,
                              const View3DIntHost &wk_lai,
                              const View2DIntHost &index_season_lai) {
  const int plon = clat.extent(0);
  auto policy = KTH::RangePolicy(0, plon);
  Kokkos::parallel_for(
      "mam4::mo_drydep::find_season_index", policy, [&](const int &j) {
        const auto index_season_lai_at_j = ekat::subview(index_season_lai, j);
        find_season_index_at_lat(clat(j), lat_lai, nlat_lai, wk_lai,
                                 index_season_lai_at_j);
      });
}

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
  EKAT_KERNEL_ASSERT_MSG(
      zl > 0, "Error in mo_drydep: cvarb: zl must be a positive number\n");
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
  for (int lt = 0; lt < n_land_type; ++lt)
    ustar[lt] = 0;
  for (int lt = 0; lt < n_land_type; ++lt)
    cvar[lt] = 0;
  for (int lt = 0; lt < n_land_type; ++lt)
    bycp[lt] = 0;

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

  for (int lt = 0; lt < n_land_type; ++lt)
    obklen[lt] = 0;
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

  for (int lt = 0; lt < n_land_type; ++lt)
    dep_ra[lt] = 0;
  for (int lt = 0; lt < n_land_type; ++lt)
    dep_rb[lt] = 0;
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
    // vegetative resistance (plant mesophyll) [s/m]
    Real rsmx[gas_pcnst][n_land_type]) {
  const auto ri = drydep_data.ri;
  const auto rgso = drydep_data.rgso;
  const auto rgss = drydep_data.rgss;
  const auto foxd = drydep_data.foxd;
  const auto drat = drydep_data.drat;

  for (int ispec = 0; ispec < gas_pcnst; ++ispec)
    for (int lt = 0; lt < n_land_type; ++lt)
      rgsx[ispec][lt] = 0;
  for (int ispec = 0; ispec < gas_pcnst; ++ispec)
    for (int lt = 0; lt < n_land_type; ++lt)
      rsmx[ispec][lt] = 0;

  for (int ispec = 0; ispec < gas_pcnst; ++ispec) {
    if (drydep_data.has_dvel[ispec]) {
      const int idx_drydep = drydep_data.map_dvel(ispec);
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

          EKAT_KERNEL_ASSERT_MSG(
              rgss(sndx, lt) != 0,
              "Error in mo_drydep: rgss should be non-zero\n");
          EKAT_KERNEL_ASSERT_MSG(
              rgso(sndx, lt) != 0,
              "Error in mo_drydep: rgso should be non-zero\n");
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

  for (int ispec = 0; ispec < gas_pcnst; ++ispec)
    for (int lt = 0; lt < n_land_type; ++lt)
      rclx[ispec][lt] = 0;

  for (int ispec = 0; ispec < gas_pcnst; ++ispec) {
    if (drydep_data.has_dvel(ispec)) {
      const int idx_drydep = drydep_data.map_dvel(ispec);
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

  // NOTE: as it stands, since rlux gets passed in, and we can't guarantee all
  // entries are initialized to 0, we do it here
  for (size_t ispec = 0; ispec < gas_pcnst; ++ispec) {
    for (size_t lt = 0; lt < n_land_type; ++lt) {
      rlux[ispec][lt] = 0.0;
    }
  }

  Real rlux_o3[n_land_type] = {}; // vegetative resistance (upper canopy) [s/m]
  for (int ispec = 0; ispec < gas_pcnst; ++ispec) {
    if (drydep_data.has_dvel(ispec)) {
      const int idx_drydep = drydep_data.map_dvel(ispec);
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
    if (drydep_data.has_dvel(ispec) && (ispec != drydep_data.so2_ndx)) {
      const int idx_drydep = drydep_data.map_dvel(ispec);
      for (int lt = beglt; lt <= endlt; ++lt) {
        if (lt != lt_for_water) {
          if (fr_lnduse[lt] && (sfc_temp > tmelt) && has_dew) {
            //-------------------------------------------------------------------------------------
            // no effect if sfc_temp < O C
            //-------------------------------------------------------------------------------------
            // BAD_CONSTANTS
            // NOTE: this is currently not called
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
      Real wrk = 0, resc = 0, lnd_frc = 0;
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
    const int index_season[n_land_type], // column-specific mapping of month
                                         // indices to seasonal land-type
                                         // indices [-]
    const Real sfc_temp,                 // surface temperature [K]
    const Real air_temp,                 // surface air temperature [K]
    const Real tv,                       // potential temperature [K]
    const Real pressure_sfc,             // surface pressure [Pa]
    const Real pressure_10m,             // 10-meter pressure [Pa]
    const Real spec_hum,                 // specific humidity [kg/kg]
    const Real wind_speed,               // 10-meter wind spped [m/s]
    const Real rain,                     // rain content [??]
    const Real solar_flux,     // direct shortwave surface radiation [W/m^2]
    const Real mmr[gas_pcnst], // constituent MMRs [kg/kg]
    Real dvel[gas_pcnst],      // deposition velocity [cm/s]
    Real dflx[gas_pcnst]) {    // deposition flux [1/cm^2/s]

  // FIXME: BAD_CONSTANTS
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
  mam4::seq_drydep::set_hcoeff_scalar(sfc_temp, heff);

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

  EKAT_KERNEL_ASSERT_MSG(pressure_sfc > pressure_10m,
                         "Error in mo_drydep: Surface pressure (pressure_sfc) "
                         "should be > pressure_10m\n");
  // BAD_CONSTANTS
  Real zl = -rair / grav * air_temp * (1. + 0.61 * spec_hum) *
            log(pressure_10m / pressure_sfc);
  EKAT_KERNEL_ASSERT_MSG(zl > 0,
                         "Error in mo_drydep: zl must be a positive number\n");

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
  if ((sfc_temp > tmelt) && (sfc_temp < temp_highbound)) {
    // FIXME: BAD_CONSTANTS
    crs = (1.0 + haero::square(200.0 / (solar_flux + 0.1))) *
          (400.0 / (tc * (40.0 - tc)));
  } else {
    crs = large_value;
  }

  //-------------------------------------------------------------------------------------
  // rdc (lower canopy res)
  //-------------------------------------------------------------------------------------
  // FIXME: BAD_CONSTANTS
  Real rdc = 100.0 * (1.0 + 1000.0 / (solar_flux + 10.0));

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
  Real cts = 0, rgsx[gas_pcnst][n_land_type] = {},
       rsmx[gas_pcnst][n_land_type] = {};
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
