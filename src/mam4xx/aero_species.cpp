#include "aero_species.hpp"
#include "mam4_constants.hpp"

#include <ekat_assert.hpp>
#include <Kokkos_Core.hpp>

namespace mam4 {

namespace internal {
typename DeviceType::view_1d<AeroSpecies> aero_species_d = {};
}

namespace {

static bool species_configured = false;

using HostType = ekat::KokkosTypes<ekat::HostDevice>;
using DeviceType = ekat::KokkosTypes<ekat::DefaultDevice>;

// Host and device representations of aerosol species
typename HostType::view_1d<AeroSpecies> aero_species_h = {};

// This function is called by Kokkos::finalize() and cleans up some related views.
void destroy_aerosol_species_views() {
  if (species_configured) {
    aero_species_h = decltype(aero_species_h)();
    internal::aero_species_d = decltype(internal::aero_species_d)();
    species_configured = false;
  }
}

}

void configure_aero_species(const std::map<AeroId, AeroSpecies>& species) {
  if (not species_configured) {
    // copy species data into place on the device
    aero_species_h = HostType::view_1d<AeroSpecies>("Aerosol species", 7);
    internal::aero_species_d = DeviceType::view_1d<AeroSpecies>("Aerosol species", 7);
    for (auto iter = species.begin(); iter != species.end(); ++iter) {
      aero_species_h[int(iter->first)] = iter->second;
    }
    Kokkos::deep_copy(internal::aero_species_d, aero_species_h);

    species_configured = true;
    Kokkos::push_finalize_hook(destroy_aerosol_species_views);
  } else {
    EKAT_ERROR_MSG("configure_aero_species cannot be called more than once!");
  }
}

void configure_default_aero_species() {
  std::map<AeroId, AeroSpecies> species = {
    {AeroId::SOA, AeroSpecies{Constants::molec_weight_c, defaults::mam4_density_soa, defaults::mam4_hyg_soa}},
    {AeroId::SO4, AeroSpecies{Constants::molec_weight_so4, defaults::mam4_density_so4, defaults::mam4_hyg_so4}},
    {AeroId::POM, AeroSpecies{Constants::molec_weight_c, defaults::mam4_density_pom, defaults::mam4_hyg_pom}},
    {AeroId::BC, AeroSpecies{Constants::molec_weight_c, defaults::mam4_density_bc, defaults::mam4_hyg_bc}},
    {AeroId::NaCl, AeroSpecies{Constants::molec_weight_nacl, defaults::mam4_density_nacl, defaults::mam4_hyg_nacl}},
    {AeroId::DST, AeroSpecies{defaults::mam4_molec_weight_dst, defaults::mam4_density_dst, defaults::mam4_hyg_dst}},
    {AeroId::MOM, AeroSpecies{defaults::mam4_molec_weight_mom, defaults::mam4_density_mom, defaults::mam4_hyg_mom}},
  };
  configure_aero_species(species);
}

void set_aero_molecular_weight(const AeroId id, Real molecular_weight) {
  EKAT_REQUIRE(species_configured, "configure_aero_species must be called before set_aero_molecular_weight");
  aero_species_h[int(id)].molecular_weight = molecular_weight;
  Kokkos::deep_copy(internal::aero_species_d, aero_species_h);
}

void set_aero_density(const AeroId id, Real density) {
  EKAT_REQUIRE(species_configured, "configure_aero_species must be called before set_aero_density");
  aero_species_h[int(id)].density = density;
  Kokkos::deep_copy(internal::aero_species_d, aero_species_h);
}

void set_aero_hygroscopicity(const AeroId id, Real hygroscopicity) {
  EKAT_REQUIRE(species_configured, "configure_aero_species must be called before set_aero_hygroscopicity");
  aero_species_h[int(id)].hygroscopicity = hygroscopicity;
  Kokkos::deep_copy(internal::aero_species_d, aero_species_h);
}

std::string aero_id_str(const AeroId id) {
  switch (id) {
  case (AeroId::SO4): {
    return "sulphate";
  }
  case (AeroId::POM): {
    return "primary_organic_matter";
  }
  case (AeroId::SOA): {
    return "secondary_organic_aerosol";
  }
  case (AeroId::BC): {
    return "black_carbon";
  }
  case (AeroId::DST): {
    return "dust";
  }
  case (AeroId::NaCl): {
    return "salt";
  }
  case (AeroId::MOM): {
    return "marine_organic_matter";
  }
  case (AeroId::None): {
    return "none";
  }
  default:
    return "invalid_aerosol_id";
  }
}

std::string aero_id_short_name(const AeroId id) {
  switch (id) {
  case (AeroId::SO4): {
    return "so4";
  }
  case (AeroId::POM): {
    return "pom";
  }
  case (AeroId::SOA): {
    return "soa";
  }
  case (AeroId::BC): {
    return "bc";
  }
  case (AeroId::DST): {
    return "dst";
  }
  case (AeroId::NaCl): {
    return "nacl";
  }
  case (AeroId::MOM): {
    return "mom";
  }
  case (AeroId::None): {
    return "none";
  }
  default:
    return "invalid_aerosol_id";
  }
}

} // namespace mam4
