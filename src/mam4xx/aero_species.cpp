// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include "aero_species.hpp"
#include "mam4_constants.hpp"

namespace mam4 {

AeroSpeciesHostView default_aero_species() {
  AeroSpeciesHostView species("Aerosol species", int(AeroId::NumSpecies));
  species[int(AeroId::SOA)] =
      AeroSpecies{Constants::molec_weight_c, defaults::mam4_density_soa,
                  defaults::mam4_hyg_soa};
  species[int(AeroId::SO4)] =
      AeroSpecies{Constants::molec_weight_so4, defaults::mam4_density_so4,
                  defaults::mam4_hyg_so4};
  species[int(AeroId::POM)] =
      AeroSpecies{Constants::molec_weight_c, defaults::mam4_density_pom,
                  defaults::mam4_hyg_pom};
  species[int(AeroId::BC)] =
      AeroSpecies{Constants::molec_weight_c, defaults::mam4_density_bc,
                  defaults::mam4_hyg_bc};
  species[int(AeroId::NaCl)] =
      AeroSpecies{Constants::molec_weight_nacl, defaults::mam4_density_nacl,
                  defaults::mam4_hyg_nacl};
  species[int(AeroId::DST)] =
      AeroSpecies{defaults::mam4_molec_weight_dst, defaults::mam4_density_dst,
                  defaults::mam4_hyg_dst};
  species[int(AeroId::MOM)] =
      AeroSpecies{defaults::mam4_molec_weight_mom, defaults::mam4_density_mom,
                  defaults::mam4_hyg_mom};
  return species;
}

AeroSpeciesView
aero_species_on_device(const AeroSpeciesHostView &species_on_host) {
  AeroSpeciesView species_on_device("On-device aerosol species",
                                    species_on_host.extent(0));
  Kokkos::deep_copy(species_on_device, species_on_host);
  return species_on_device;
}

AeroSpeciesHostView
aero_species_on_host(const AeroSpeciesView &species_on_device) {
  AeroSpeciesHostView species_on_host("On-host aerosol species",
                                      species_on_device.extent(0));
  Kokkos::deep_copy(species_on_host, species_on_device);
  return species_on_host;
}

std::string aero_id_str(const AeroId aid) {
  switch (aid) {
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

std::string aero_id_short_name(const AeroId aid) {
  switch (aid) {
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
