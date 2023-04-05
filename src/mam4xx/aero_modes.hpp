// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MAM4XX_AERO_MODES_HPP
#define MAM4XX_AERO_MODES_HPP

#include <haero/aero_species.hpp>
#include <haero/constants.hpp>
#include <haero/gas_species.hpp>
#include <haero/math.hpp>

#include "mam4_types.hpp"

#include <iostream>
#include <string>
#include <vector>

namespace mam4 {

/// @struct Mode
/// This struct represents an aerosol particle mode and contains all associated
/// metadata. By definition, these metadata are immutable (constant in time).
/// The struct is not polymorphic, so don't derive any subclass from it.
///
/// This class represents the log-normal distribution that defines the mode via
/// the mean_std_dev member variable.  The other parameter necessary to define
/// the log-normal function is a variable (a function of mass- and number-
/// mixing ratios) and is not included in this class.
///
/// The member variables min_diameter and max_diameter do not define the bounds
/// of the log-normal distribution (which, matematically, are 0 and positive
/// infinity).  Rather, these min/max values are used to trigger a mass and
/// number redistribution elsewhere in the code; they signify the bounds beyond
/// which particles are considered to better belong in a different mode.
///
/// Variable nom_diameter is the nominal geometeric mean diameter [m]
/// of particles in a mode
///
/// Crystalization and deliquesence refer to the non-cloud water uptake process,
/// by which liquid water condenses into aerosol droplets.  They are relative
/// humidity values.  When the environmental relative humidity lies below the
/// cyrstalization point, water uptake does not occur.  When it lies between the
/// crystallization and deliquesence point, water uptake does occur, but not at
/// its maximum rate.   When the environmental relative humidty exceeds the
/// deliquescence_pt, particles achieve their maximum amount of liquid water.
///
struct Mode final {
  using Real = haero::Real;
  /// The minimum diameter for particles that belong to this mode.
  const Real min_diameter;
  /// The nominal diameter for particles that belong to this mode.
  const Real nom_diameter;
  /// The maximum diameter for particles that belong to this mode.
  const Real max_diameter;
  /// The geometric mean standard deviation for this mode.
  const Real mean_std_dev;
  /// The crystallization point [rel. humidity] for this mode.
  const Real crystallization_pt;
  /// The deliquescence point [rel. humidity] for this mode.
  const Real deliquescence_pt;
};

/// Mode indices in MAM4
enum class ModeIndex {
  Accumulation = 0,
  Aitken = 1,
  Coarse = 2,
  PrimaryCarbon = 3,
};

/// Map ModeIndex to string (for logging, e.g.)
/// This function cannot be called inside a GPU kernel,
/// but it's helpful to use with ekat::Logger statements
/// (which also cannot be called from inside a kernel)
std::string mode_str(const ModeIndex m);

static constexpr Real mam4_crystallization_rel_hum = 0.35;
static constexpr Real mam4_delequesence_rel_hum = 0.8;
static constexpr Real mam4_accum_min_diameter_m = 5.35e-8;
static constexpr Real mam4_accum_nom_diameter_m = 1.1e-7;
static constexpr Real mam4_accum_max_diameter_m = 4.4e-7;
static constexpr Real mam4_accum_mead_std_dev = 1.8;
static constexpr Real mam4_aitken_min_diameter_m = 8.7e-9;
static constexpr Real mam4_aitken_nom_diameter_m = 2.6e-8;
static constexpr Real mam4_aitken_max_diameter_m = 5.2e-8;
static constexpr Real mam4_aitken_mead_std_dev = 1.6;
static constexpr Real mam4_coarse_min_diameter_m = 1e-6;
static constexpr Real mam4_coarse_nom_diameter_m = 2e-6;
static constexpr Real mam4_coarse_max_diameter_m = 4e-6;
static constexpr Real mam4_coarse_mead_std_dev = 1.8;
static constexpr Real mam4_primary_carbon_min_diameter_m = 1e-8;
static constexpr Real mam4_primary_carbon_nom_diameter_m = 5e-8;
static constexpr Real mam4_primary_carbon_max_diameter_m = 1e-7;
static constexpr Real mam4_primary_carbon_mead_std_dev = 1.6;

/// A list of all modes within MAM4.
/// NOTE: MAM4 uses the same constant crystallization and deliquescence
/// NOTE: values for all modes & species.  See links for additional discussion:
/// NOTE:
/// https://eagles-project.atlassian.net/wiki/spaces/Computation/pages/1125515265/Aerosol+species+and+mode+data
/// NOTE:
/// https://eagles-project.atlassian.net/wiki/spaces/Computation/pages/354877515/Module+verifications
/// NOTE: These data are found on Anvil in
/// NOTE: /lcrc/group/acme/ccsm-data/inputdata/atm/cam/physprops/
KOKKOS_INLINE_FUNCTION const mam4::Mode &modes(const int i) {
  static const mam4::Mode M[4] = {
      // accumulation
      {mam4_accum_min_diameter_m, mam4_accum_nom_diameter_m,
       mam4_accum_max_diameter_m, mam4_accum_mead_std_dev,
       mam4_crystallization_rel_hum, mam4_delequesence_rel_hum},
      // aitken
      {mam4_aitken_min_diameter_m, mam4_aitken_nom_diameter_m,
       mam4_aitken_max_diameter_m, mam4_aitken_mead_std_dev,
       mam4_crystallization_rel_hum, mam4_delequesence_rel_hum},
      // coarse
      {mam4_coarse_min_diameter_m, mam4_coarse_nom_diameter_m,
       mam4_coarse_max_diameter_m, mam4_coarse_mead_std_dev,
       mam4_crystallization_rel_hum, mam4_delequesence_rel_hum},
      // primary carbon
      {mam4_primary_carbon_min_diameter_m, mam4_primary_carbon_nom_diameter_m,
       mam4_primary_carbon_max_diameter_m, mam4_primary_carbon_mead_std_dev,
       mam4_crystallization_rel_hum, mam4_delequesence_rel_hum}};
  return M[i];
};

/// Identifiers for aerosol species that inhabit MAM4 modes.
enum class AeroId {
  SOA = 0,  // secondary organic aerosol
  SO4 = 1,  // sulphate
  POM = 2,  // primary organic matter
  BC = 3,   // black carbon
  NaCl = 4, // sodium chloride
  DST = 5,  // dust
  MOM = 6,  // marine organic matter,
  None = 7  // invalid aerosol species
};

/// Map ModeIndex to string (for logging, e.g.)
/// This function cannot be called inside a GPU kernel,
/// but it's helpful to use with ekat::Logger statements
/// (which also cannot be called inside a kernel)
std::string aero_id_str(const AeroId aid);

/// Molecular weight of mam4 dust aerosol [kg/mol]
static constexpr Real mam4_molec_weight_dst = 0.135065;

/// Molecular weight of mam4 marine organic matter [kg/mol]
static constexpr Real mam4_molec_weight_mom = 250.093;

/// mam4 aerosol densities [kg/m3]
static constexpr Real mam4_density_soa = 1000.0;
static constexpr Real mam4_density_so4 = 1770.0;
static constexpr Real mam4_density_pom = 1000.0;
static constexpr Real mam4_density_bc = 1700.0;
static constexpr Real mam4_density_nacl = 1900.0;
static constexpr Real mam4_density_dst = 2600.0;
static constexpr Real mam4_density_mom = 1601.0;

/// mam4 aerosol hygroscopicities
static constexpr Real mam4_hyg_soa = 0.1;
static constexpr Real mam4_hyg_so4 = 0.507;
static constexpr Real mam4_hyg_pom = 1e-10;
static constexpr Real mam4_hyg_bc = 1e-10;
static constexpr Real mam4_hyg_nacl = 1.16;
static constexpr Real mam4_hyg_dst = 0.14;
static constexpr Real mam4_hyg_mom = 0.1;

/// A list of aerosol species in MAM4.
/**
  Note that in MAM4 fortran, molecular weights are given as g/mol, rather than
  kg/mol.

  Here and in Haero we use SI units for everything, so molecular weights
  are given as [kg/mol].

  When the variable is "universal" in the sense that it will be the same
  whether MAM4 is using or some other software package is using it, we
  use the external haero::Constants value, which is sourced to the latest
  NIST data available.  Additionally, this prepares Mam4xx to ultimately
  use an external source of constants with EAM.  Examples are the
  molecular weights of Carbon, Sulphate, and Sodium Chloride.

  Some of these constants are unique to mam4 -- these are listed here, with
  the prefix mam4_*. For example, its definition
  of primary carbon, dust, and marine organic matter are defined by choices
  of what those modes represent.  Other examples, such as the density of some
  substances, differ from the values provided by NIST; these, too, are listed
  here as mam4_* constants.
*/
KOKKOS_INLINE_FUNCTION AeroSpecies aero_species(const int i) {
  static const AeroSpecies species[7] = {
      AeroSpecies{Constants::molec_weight_c, mam4_density_soa,
                  mam4_hyg_soa}, // secondary organic aerosol
      AeroSpecies{Constants::molec_weight_so4, mam4_density_so4, mam4_hyg_so4},
      AeroSpecies{Constants::molec_weight_c, mam4_density_pom,
                  mam4_hyg_pom}, // primary organic matter
      AeroSpecies{Constants::molec_weight_c, mam4_density_bc,
                  mam4_hyg_bc}, // black carbon
      AeroSpecies{Constants::molec_weight_nacl, mam4_density_nacl,
                  mam4_hyg_nacl}, // sodium chloride
      AeroSpecies{mam4_molec_weight_dst, mam4_density_dst,
                  mam4_hyg_dst}, // dust
      AeroSpecies{mam4_molec_weight_mom, mam4_density_mom,
                  mam4_hyg_mom} // marine organic matter
  };
  return species[i];
}

// A list of species within each mode for MAM4.
KOKKOS_INLINE_FUNCTION AeroId mode_aero_species(const int modeNo,
                                                const int speciesNo) {
  // A list of species within each mode for MAM4.
  static constexpr AeroId mode_aero_species[4][7] = {
      {// accumulation mode
       AeroId::SOA, AeroId::SO4, AeroId::POM, AeroId::BC, AeroId::NaCl,
       AeroId::DST, AeroId::MOM},
      {
          // aitken mode
          AeroId::SOA,
          AeroId::SO4,
          AeroId::NaCl,
          AeroId::MOM,
          AeroId::None,
          AeroId::None,
          AeroId::None,
      },
      {// coarse mode
       AeroId::SOA, AeroId::SO4, AeroId::POM, AeroId::BC, AeroId::NaCl,
       AeroId::DST, AeroId::MOM},
      {// primary carbon mode
       AeroId::POM, AeroId::BC, AeroId::MOM, AeroId::None, AeroId::None,
       AeroId::None, AeroId::None}};

  return mode_aero_species[modeNo][speciesNo];
}

/// Returns number of species per mode
KOKKOS_INLINE_FUNCTION int num_species_mode(const int i) {
  static constexpr int _num_species_mode[4] = {7, 4, 7, 3};
  return _num_species_mode[i];
}

/// Returns the index of the given aerosol species within the given mode, or
/// -1 if the species is not found within the mode.
KOKKOS_INLINE_FUNCTION
int aerosol_index_for_mode(ModeIndex mode, AeroId aero_id) {
  int mode_index = static_cast<int>(mode);
  for (int s = 0; s < 7; ++s) {
    if (aero_id == mode_aero_species(mode_index, s)) {
      return s;
    }
  }
  return -1;
}
/// Convenient function that returns bool indicating if species is
/// within mode.
KOKKOS_INLINE_FUNCTION
bool mode_contains_species(ModeIndex mode, AeroId aero_id) {
  return -1 != aerosol_index_for_mode(mode, aero_id);
}

// Identifiers for gas species in MAM4, specified in the same order as they
// appear in the set_gas_and_aer_names_and_indices subroutine within the
// modal_aero_microp_species F90 module (assuming nsoa == 1). It looks like MAM4
// only tracks SOAG and H2SO4. We keep NH3 around because ternary nucleation
// requires it, is already ported, and may be required in the future.
enum class GasId {
  SOAG = 0,  // secondary organic aerosol precursor
  H2SO4 = 1, // sulfuric acid
  NH3 = 2,   // ammonia
};

/// Molecular weight of carbon dioxide [kg/mol]
static constexpr Real molec_weight_co2 = 0.0440095;
/// Molecular weight of methane @f$\text{CH}_4@f$
static constexpr Real molec_weight_ch4 = 0.0160425;
/// Molecular weight of trichlorofluoromethan @f$\text{CCl}_3\text{F}@f$
static constexpr Real molec_weight_ccl3f = 0.13736;
/// Molecular weight of dichlorofluoromethane @f$\texct{CHCl}_2F@f$
static constexpr Real molec_weight_chcl2f = 0.10292;
/// Molecular weight of hydrogen peroxide @f$\text{H}_2\text{O}_2@f$
static constexpr Real molec_weight_h2o2 = 0.034015;
/// Molecular weight of dimethylsulfide @f$\text{C}_2\text{H}_6\text{S}@f$
static constexpr Real molec_weight_dms = 0.06214;
/// Molecular weight of oxygen molecule @f$\text{O}_2@f$
static constexpr Real molec_weight_o2 = 0.0319988;
/// Molecular weight of nitrous oxide @f$\text{N}_2\text{O}@f$
static constexpr Real molec_weight_n2o = 0.044013;
/// Molecular weight of ozone @f$\text{O}_3@f$
static constexpr Real molec_weight_o3 = 0.0479982;
/// Molecular weight of sulfur dioxide @f$\text{SO}_2@f$
static constexpr Real molec_weight_so2 = 0.06407;

/// A list of gas species in MAM4.
KOKKOS_INLINE_FUNCTION GasSpecies gas_species(const int i) {
  static const GasSpecies species[13] = {
      {molec_weight_o3},               // ozone
      {molec_weight_h2o2},             // hydrogen peroxide
      {Constants::molec_weight_h2so4}, // sulfuric acid
      {molec_weight_so2},              // sulfur dioxide
      {molec_weight_dms},              // dimethylsulfide
      {Constants::molec_weight_c},     // secondary organic aerosol precursor
      {molec_weight_o2},               // oxygen
      {molec_weight_co2},              // carbon dioxide
      {molec_weight_n2o},              // nitrous oxide
      {molec_weight_ch4},              // methane
      {molec_weight_ccl3f},            // thrichlorofluoromethane
      {molec_weight_chcl2f},           // dichlorofluoromethane
      {Constants::molec_weight_nh3}    // ammonia
  };
  return species[i];
}

} // namespace mam4

#endif
