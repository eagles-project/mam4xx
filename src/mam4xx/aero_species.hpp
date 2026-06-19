// Copyright (c) 2021, National Technology & Engineering Solutions of Sandia,
// LLC (NTESS). Copyright (c) 2022, Battelle Memorial Institute
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MAM4XX_AERO_SPECIES_HPP
#define MAM4XX_AERO_SPECIES_HPP

#include "mam4_config.hpp"

#include <ekat_kokkos_types.hpp>

#include <string>

namespace mam4 {

/// @struct AeroSpecies
/// This type represents an aerosol species.
struct AeroSpecies {
  // Molecular weight [kg/mol]
  Real molecular_weight;

  /// Material density [kg/m^3]
  Real density;

  /// Hygroscopicity
  Real hygroscopicity;
};

/// Identifiers for aerosol species that inhabit MAM4 modes.
enum class AeroId {
  SOA = 0,        // secondary organic aerosol
  SO4 = 1,        // sulphate
  POM = 2,        // primary organic matter
  BC = 3,         // black carbon
  NaCl = 4,       // sodium chloride
  DST = 5,        // dust
  MOM = 6,        // marine organic matter,
  NumSpecies = 7, // number of aerosol species
  None = 8        // invalid aerosol species
};

/// A device-side Kokkos View containing aerosol species.
using AeroSpeciesView =
    typename ekat::KokkosTypes<ekat::DefaultDevice>::view_1d<AeroSpecies>;

/// A host-side Kokkos View for configuring aerosol species.
using AeroSpeciesHostView =
    typename ekat::KokkosTypes<ekat::HostDevice>::view_1d<AeroSpecies>;

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

//--------------------------------------------------------
// The following functions can only be called on the host
//--------------------------------------------------------

/// Returns a newly-created view containing the default configuration for
/// aerosol species. Create this on the host, override properties as desired,
/// and copy to device with Kokkos::deep_copy.
AeroSpeciesHostView default_aero_species();

/// Maps an AeroId to the name of its species.
std::string aero_id_str(const AeroId id);

/// Maps an AeroId to a shortened name for its species.
std::string aero_id_short_name(const AeroId id);

} // namespace mam4

#endif
