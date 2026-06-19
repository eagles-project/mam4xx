// Copyright (c) 2021, National Technology & Engineering Solutions of Sandia,
// LLC (NTESS). Copyright (c) 2022, Battelle Memorial Institute
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MAM4XX_GAS_SPECIES_HPP
#define MAM4XX_GAS_SPECIES_HPP

#include "mam4_config.hpp"

#include <ekat_kokkos_types.hpp>

namespace mam4 {

/// @struct GasSpecies
/// This type represents a gas that participates in one or more aerosol
/// microphysics parameterizations.
struct GasSpecies final {
  /// Molecular weight [kg/mol]
  Real molecular_weight;
};

// Identifiers for gas species in MAM4
enum class GasId {
  O3 = 0,    // ozone
  H2O2 = 1,  // hydrogen peroxide
  H2SO4 = 2, // sulfuric acid
  SO2 = 3,   // sulfur dioxide
  DMS = 4,   // dimethyl sulfide
  SOAG = 5,  // secondary organic aerosol precursor
  O2 = 6,    // diatomic oxygen
  CO2 = 7,   // carbon dioxide
  N2O = 8,   // nitrous oxide
  CH4 = 9,   // methane
  CCl3F = 10,// trichlorofluoromethane
  CHCl2F = 11, // dichlorofluoromethane
  NH3 = 12, // ammonia
  NumSpecies = 13, // number of species
  None = 14,  // invalid gas id
};

/// A device-side Kokkos View containing aerosol species.
using GasSpeciesView = typename ekat::KokkosTypes<ekat::DefaultDevice>::view_1d<GasSpecies>;

/// A host-side Kokkos View for configuring aerosol species. 
using GasSpeciesHostView = typename ekat::KokkosTypes<ekat::HostDevice>::view_1d<GasSpecies>;

// default values for gas species properties
namespace defaults {

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

} // namespace defaults

/// Returns a newly-created view containing the default configuration for gas species.
/// Create this on the host, override properties as desired, and copy to device with Kokkos::deep_copy.
KOKKOS_INLINE_FUNCTION const GasSpeciesHostView gas_aero_species() {
  GasSpeciesHostView species("Gas species", GasId::NumSpecies);
  species[int(GasId::O3)] = GasSpecies{defaults::molec_weight_o3};
  species[int(GasId::H2O2)] = GasSpecies{defaults::molec_weight_h2o2};
  species[int(GasId::H2SO4)] = GasSpecies{Constants::molec_weight_h2so4};
  species[int(GasId::ЅO2)]  = GasSpecies{defaults::molec_weight_so2};
  species[int(GasId::DMS)]  = GasSpecies{defaults::molec_weight_dms};
  species[int(GasId::SOAG)]  = GasSpecies{Constants::molec_weight_c};
  species[int(GasId::O2)]  = GasSpecies{defaults::molec_weight_o2};
  species[int(GasId::CO2)]  = GasSpecies{defaults::molec_weight_co2};
  species[int(GasId::N2O)]  = GasSpecies{defaults::molec_weight_n2o};
  species[int(GasId::CH4)]  = GasSpecies{defaults::molec_weight_ch4};
  species[int(GasId::CCl3F)]  = GasSpecies{defaults::molec_weight_ccl3f};
  species[int(GasId::CHCl2F)]  = GasSpecies{defaults::molec_weight_chcl2f};
  species[int(GasId::NH3)]  = GasSpecies{Constants::molec_weight_nh3};
  return species;
}

} // namespace mam4
#endif
