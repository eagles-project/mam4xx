// Copyright (c) 2021, National Technology & Engineering Solutions of Sandia,
// LLC (NTESS). Copyright (c) 2022, Battelle Memorial Institute
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MAM4XX_AERO_SPECIES_HPP
#define MAM4XX_AERO_SPECIES_HPP

#include "Kokkos_Macros.hpp"
#include "mam4_config.hpp"

#include <ekat_kokkos_types.hpp>

#include <string>

namespace mam4 {

/// @struct AeroSpecies
/// This type represents an aerosol species.
struct AeroSpecies final {
  // Molecular weight [kg/mol]
  Real molecular_weight;

  /// Material density [kg/m^3]
  Real density;

  /// Hygroscopicity
  Real hygroscopicity;
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

// default values for aerosol species properties
namespace defaults {

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

} // namespace defaults

namespace internal {

using DeviceType = ekat::KokkosTypes<ekat::DefaultDevice>;
extern typename DeviceType::view_1d<AeroSpecies> aero_species_d;

} // namespace internal

/// Returns the aerosol species associated with the given unique identifier.
/**
  Note that in MAM4 fortran, molecular weights are given as g/mol, rather than
  kg/mol.

  Here we use SI units for everything, so molecular weights are given as
  [kg/mol].

  When the variable is "universal" in the sense that it will be the same
  whether MAM4 is using or some other software package is using it, we
  use the external Constants value, which is sourced to the latest
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
KOKKOS_INLINE_FUNCTION const AeroSpecies aero_species(const AeroId id) {
  return internal::aero_species_d[int(id)];
}

//--------------------------------------------------------
// The following functions can only be called on the host
//--------------------------------------------------------

// Configures the set of aerosol species to be used by mam4xx. This may only be called once.
void configure_aero_species(const std::map<AeroId, AeroSpecies>& species);

// Configures the default set of aerosol species to be used by mam4xx. Call only once.
void configure_default_aero_species();

// Overrides the molecular weight [kg/mol] in the aerosol species with the given ID.
void set_aero_molecular_weight(const AeroId id, Real molecular_weight);

// Overrides the mass density [kg/m^3] in the aerosol species with the given ID.
void set_aero_density(const AeroId id, Real density);

// Overrides the hygroscopicity in the aerosol species with the given ID.
void set_aero_hygroscopicity(const AeroId id, Real hygroscopicity);

/// Maps an AeroId to the name of its species.
std::string aero_id_str(const AeroId id);

/// Maps an AeroId to a shortened name for its species.
std::string aero_id_short_name(const AeroId id);

} // namespace mam4

#endif
