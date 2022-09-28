#ifndef MAM4XX_AERO_MODES_HPP
#define MAM4XX_AERO_MODES_HPP

#include <haero/aero_species.hpp>
#include <haero/gas_species.hpp>
#include <haero/math.hpp>
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

  // Default constructor needed to resize Kokkos Views on device before deep
  // copy.
  KOKKOS_INLINE_FUNCTION
  Mode()
      : min_diameter(0),
        nom_diameter(0),
        max_diameter(0),
        mean_std_dev(1),
        crystallization_pt(0),
        deliquescence_pt(0) {}

  /// Creates an aerosol particle mode.
  /// @param [in] min_diam The minimum diameter for particles that belong
  ///                      to this mode [m].
  /// @param [in] nom_diam The nominal diameter for particles that belong
  ///                      to this mode [m].
  /// @param [in] max_diam The maximum diameter for particles that belong
  ///                      to this mode [m].
  /// @param [in] sigma    The geometric standard deviation for this mode.
  /// @param [in] crystal_pt The crystallization point of the mode
  /// @param [in] deliq_pt The deliquescence point of the mode
  Mode(Real min_diam, Real nom_diam, Real max_diam, Real sigma, Real crystal_pt,
       Real deliq_pt)
      : min_diameter(min_diam),
        nom_diameter(nom_diam),
        max_diameter(max_diam),
        mean_std_dev(sigma),
        crystallization_pt(crystal_pt),
        deliquescence_pt(deliq_pt) {
    EKAT_ASSERT(max_diam > min_diam);
    EKAT_ASSERT(nom_diam > min_diam);
    EKAT_ASSERT(max_diam > nom_diam);
    EKAT_ASSERT(deliq_pt > crystal_pt);
    EKAT_ASSERT(sigma >= 1);
  }

  KOKKOS_INLINE_FUNCTION
  Mode(const Mode &m)
      : min_diameter(m.min_diameter),
        nom_diameter(m.nom_diameter),
        max_diameter(m.max_diameter),
        mean_std_dev(m.mean_std_dev),
        crystallization_pt(m.crystallization_pt),
        deliquescence_pt(m.deliquescence_pt) {}

  KOKKOS_INLINE_FUNCTION
  Mode &operator=(const Mode &m) {
    min_diameter = m.min_diameter;
    nom_diameter = m.nom_diameter;
    max_diameter = m.max_diameter;
    mean_std_dev = m.mean_std_dev;
    crystallization_pt = m.crystallization_pt;
    deliquescence_pt = m.deliquescence_pt;
    return *this;
  }

  KOKKOS_INLINE_FUNCTION
  ~Mode() {}

  /// The minimum diameter for particles that belong to this mode.
  Real min_diameter;

  /// The nominal diameter for particles that belong to this mode.
  Real nom_diameter;

  /// The maximum diameter for particles that belong to this mode.
  Real max_diameter;

  /// The geometric mean standard deviation for this mode.
  Real mean_std_dev;

  /// The crystallization point [rel. humidity] for this mode.
  Real crystallization_pt;

  /// The deliquescence point [rel. humidity] for this mode.
  Real deliquescence_pt;

  // Comparison operators.
  KOKKOS_INLINE_FUNCTION
  bool operator==(const Mode &other) const {
    return ((min_diameter == other.min_diameter) and
            (max_diameter == other.max_diameter) and
            (mean_std_dev == other.mean_std_dev) and
            (crystallization_pt == other.crystallization_pt) and
            (deliquescence_pt == other.deliquescence_pt));
  }
  KOKKOS_INLINE_FUNCTION
  bool operator!=(const Mode &other) const { return !(*this == other); }
};

/// Mode indices in MAM4
enum class ModeIndex {
  Accumulation,
  Aitken,
  Coarse,
  PrimaryCarbon,
};

/// A list of all modes within MAM4.
/// NOTE: Legacy MAM4 uses the same constant crystallization and deliquescence
/// NOTE: values for all modes & species.  See links for additional discussion:
/// NOTE:
/// https://eagles-project.atlassian.net/wiki/spaces/Computation/pages/1125515265/Aerosol+species+and+mode+data
/// NOTE:
/// https://eagles-project.atlassian.net/wiki/spaces/Computation/pages/354877515/Module+verifications
/// NOTE: These data are found on Anvil in
/// NOTE: /lcrc/group/acme/ccsm-data/inputdata/atm/cam/physprops/
static Mode modes[4] = {
    Mode(5.35e-8, 1.1e-7, 4.4e-7, 1.8, 0.35, 0.8),  // accumulation
    Mode(8.7e-9, 2.6e-8, 5.2e-8, 1.6, 0.35, 0.8),   // aitken
    Mode(1e-6, 2e-6, 4e-6, 1.8, 0.35, 0.8),         // coarse
    Mode(1e-8, 5e-8, 1e-7, 1.6, 0.35, 0.8)          // primary carbon
};

/// Identifiers for aerosol species that inhabit MAM4 modes.
enum class AeroId {
  SO4,   // sulphate
  POM,   // primary organic matter
  SOA,   // secondary organic aerosol
  BC,    // black carbon
  DST,   // dust
  NaCl,  // sodium chloride
  MOM,   // marine organic matter,
  None   // invalid aerosol species
};

// A list of aerosol species in MAM4.
static haero::AeroSpecies aero_species[7] = {
    haero::AeroSpecies(96.0, 1770.0, 0.507),    // sulphate
    haero::AeroSpecies(12.011, 1000.0, 1e-10),  // primary organic matter
    haero::AeroSpecies(12.011, 1000.0, 0.14),   // secondary organic aerosol
    haero::AeroSpecies(12.011, 1700.0, 1e-10),  // black carbon
    haero::AeroSpecies(135.065, 2600.0, 0.14),  // dust
    haero::AeroSpecies(58.4425, 1900.0, 1.16),  // sodium chloride
    haero::AeroSpecies(250093.0, 1601.0, 0.1)   // marine organic matter
};

/// Returns the index of the given aerosol species within the given mode, or
/// -1 if the species is not found within the mode.
KOKKOS_INLINE_FUNCTION
int aerosol_index_for_mode(ModeIndex mode, AeroId aero_id) {
  // A list of species within each mode for MAM4.
  static constexpr AeroId mode_aero_species[4][7] = {
      {// accumulation mode
       AeroId::SO4, AeroId::POM, AeroId::SOA, AeroId::BC, AeroId::DST,
       AeroId::NaCl, AeroId::MOM},
      {
          // aitken mode
          AeroId::SO4,
          AeroId::SOA,
          AeroId::NaCl,
          AeroId::MOM,
          AeroId::None,
          AeroId::None,
          AeroId::None,
      },
      {// coarse mode
       AeroId::DST, AeroId::NaCl, AeroId::SO4, AeroId::BC, AeroId::POM,
       AeroId::SOA, AeroId::MOM},
      {// primary carbon mode
       AeroId::POM, AeroId::BC, AeroId::MOM, AeroId::None, AeroId::None,
       AeroId::None, AeroId::None}};

  int mode_index = static_cast<int>(mode);
  for (int s = 0; s < 7; ++s) {
    if (aero_id == mode_aero_species[mode_index][s]) {
      return s;
    }
  }
  return -1;
}

// Identifiers for gas species in MAM4.
enum class GasId {
  O3,     // ozone
  H2O2,   // hydrogen peroxide
  H2SO4,  // sulfuric acid
  SO2,    // sulfur dioxide
  DMS,    // dimethylsulfide
  SOAG,   // secondary organic aerosol precursor
  O2,     // oxygen
  CO2,    // carbon dioxide
  N2O,    // nitrous oxide
  CH4,    // methane,
  CFC11,  // trichlorofluoromethane
  CFC12,  // dichlorodifluoromethane
  NH3     // ammonia
};

// A list of gas species in MAM4.
static haero::GasSpecies gas_species[13] = {
    haero::GasSpecies(47.9982),  // ozone
    haero::GasSpecies(34.0136),  // hydrogen peroxide
    haero::GasSpecies(98.0784),  // sulfuric acid
    haero::GasSpecies(64.0648),  // sulfur dioxide
    haero::GasSpecies(62.1324),  // dimethylsulfide
    haero::GasSpecies(12.011),   // secondary organic aerosol precursor
    haero::GasSpecies(31.988),   // oxygen
    haero::GasSpecies(44.009),   // carbon dioxide
    haero::GasSpecies(44.013),   // nitrous oxide
    haero::GasSpecies(16.04),    // methane
    haero::GasSpecies(137.73),   // thrichlorofluoromethane
    haero::GasSpecies(120.91),   // dichlorofluoromethane
    haero::GasSpecies(50.0)      // ammonia
};

}  // namespace mam4

#endif
